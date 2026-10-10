"""Fixed single-image recognition, derived from the current production algorithm.

Input BGR uint8 -> resize -> orientation -> color mask -> cleanup -> palm
reconstruction -> contour/finger geometry -> five TIMRP bits.
Each call is independent. No plugin registry, temporal memory or image labels.
Recognition parameters are preserved; this is a speed refactor, not retraining.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Optional

import cv2
import numpy as np
import finger_geometry as FG

IMAGE_MAX_WIDTH = 2560
IMAGE_MAX_HEIGHT = 1440

# Parameters inherited unchanged from the production still-image path.
ROI_WIDTH_VS_SHORT = 0.75
ROI_HEIGHT_RATIO = 0.8
ROI_CENTER_Y = 0.5
ROI_ENABLED = False
CR_LOW = 131
CR_HIGH = 173
CB_LOW = 77
CB_HIGH = 128
Y_LOW = 40
Y_HIGH = 245
SCENE_BASELINE_CR = 127.0
SCENE_BASELINE_CB = 128.0
SCENE_SHIFT_CLAMP = 22.0
SCENE_ESTIMATE_STEP = 4
CLEANUP_OPEN_KERNEL = 3
CLEANUP_MIN_AREA_RATIO = 0.002
CLEANUP_MIN_AREA_VS_LARGEST = 0.25
CLEANUP_FILL_HOLES = True
RECON_SEED_DT_FACTOR = 0.7
RECON_TOL_CR = 12
RECON_TOL_CB = 15
RECON_TOL_Y_DOWN = 75
RECON_MIN_SEED_PIXELS = 40
RECON_MIN_RADIUS_PX = 8
RECON_PALM_CLOSE_RATIO = 0.13
RECOG_WRIST_MIN_PIXELS = 12
RECOG_TRUNCATE_FACTOR = 1.5
RECOG_TIP_MIN_DISTANCE = 1.0
RECOG_TIP_PALM_THICK = 0.45
RECOG_TIP_PROTRUSION_MIN = 1.45
RECOG_TIP_PEAK_WINDOW = 0.45
RECOG_TIP_MIN_PROMINENCE = 0.2
RECOG_TIP_MIN_TURN_RATIO = 0.1
RECOG_TIP_MAX_ANGLE_DEG = 145.0
RECOG_TIP_THICKNESS_SAMPLE = 0.2
RECOG_TIP_THICKNESS_MAX = 0.45
RECOG_TIP_MIN_ANGLE_DEG = 20.0
RECOG_TIP_MIN_THICKNESS = 0.05
RECOG_TIP_DEDUP = 0.35
RECOG_TIP_MAX_COUNT = 6
RECOG_TIP_PALM_SIDE_MAX = 0.15
RECOG_GAP_MIN_DEPTH = 0.25
RECOG_GAP_MAX_ANGLE_DEG = 105.0
RECOG_GAP_MAX_COUNT = 5
FINGER_NAMES = ('thumb', 'index', 'middle', 'ring', 'pinky')
FINGER_LABELS_CN = ('拇指', '食指', '中指', '无名指', '小指')
FINGER_TEMPLATE_DEG = (47.0, -2.0, -19.5, -36.0, -60.0)
FINGER_THUMB_RANGE_DEG = (40.0, 95.0)
FINGER_SEARCH_SCALES = (0.8, 0.9, 1.0, 1.1, 1.2)
FINGER_CONTIGUOUS_PENALTY = 900.0
FINGER_PROFILE_MIN_DEG = -115.0
FINGER_PROFILE_MAX_DEG = 115.0
FINGER_PROFILE_STEP_DEG = 2.0
FINGER_TEMPLATE_LEN = (2.18, 2.85, 3.0, 2.84, 2.54)
FINGER_W_ANGLE = 1.0
FINGER_W_LEN = 0.0
FINGER_NONTHUMB_SHIFT_DEG = 25.0
FINGER_NONTHUMB_SHIFT_STEP = 2.5
FINGER_W_SCALE = 20000.0
FINGER_WEB_X = (0.72, 0.2, -0.34, -0.84)
FINGER_W_WEB = 3000.0
FINGER_WEB_COUNT_PENALTY = 1.0
FINGER_EXTENDED_MIN_R = 1.8
FINGER_PROBE_R_MIN = 0.5
FINGER_PROBE_R_MAX = 3.2
FINGER_PROBE_R_STEP = 0.05
FINGER_RUN_GAP_TOLERANCE = 0.25
RECON_PLUGIN_NAME = 'palm_reconstruct'
AUTO_ROTATE_DOWNSCALE = 320
AUTO_ROTATE_COLOR_SLACK = 0
AUTO_ROTATE_EDGE_MIN_PIXELS = 3
AUTO_ROTATE_EDGE_MAX_FRAC = 0.35
AUTO_ROTATE_MIN_AREA = 0.004


def _roi_size_pixels(shape_hw: tuple) -> tuple:
    height, width = (int(shape_hw[0]), int(shape_hw[1]))
    short = min(height, width)
    roi_w = int(round(float(ROI_WIDTH_VS_SHORT) * short))
    roi_h = int(round(float(ROI_HEIGHT_RATIO) * height))
    roi_w = max(1, min(roi_w, width))
    roi_h = max(1, min(roi_h, height))
    return (roi_w, roi_h)


def roi_rect_pixels(shape_hw: tuple, anchor=None) -> tuple:
    height, width = (int(shape_hw[0]), int(shape_hw[1]))
    roi_w, roi_h = _roi_size_pixels((height, width))
    if anchor is None:
        cx, cy = (width * 0.5, height * float(ROI_CENTER_Y))
    else:
        cx, cy = (float(anchor[0]), float(anchor[1]))
    x0 = int(round(cx - roi_w * 0.5))
    y0 = int(round(cy - roi_h * 0.5))
    x0 = max(0, min(x0, width - roi_w))
    y0 = max(0, min(y0, height - roi_h))
    x1 = min(width, x0 + roi_w)
    y1 = min(height, y0 + roi_h)
    return (x0, y0, x1, y1)


def roi_enabled_now(state):
    return state.flags['roi_enabled']


def _normalized_pair(low, high, label: str) -> tuple:
    low, high = (int(low), int(high))
    if low > high:
        print('[plugins] 警告：%s 的下限(%d) > 上限(%d)，已自动交换' % (label, low, high))
        low, high = (high, low)
    return (low, high)


def _finish_mask(mask: np.ndarray, state, method: str) -> np.ndarray:
    height, width = mask.shape[:2]
    stats = {}
    enabled = roi_enabled_now(state)
    if enabled:
        x0, y0, x1, y1 = roi_rect_pixels((height, width))
        mask[:y0] = 0
        mask[y1:] = 0
        mask[y0:y1, :x0] = 0
        mask[y0:y1, x1:] = 0
        stats['roi_rect'] = (x0, y0, x1, y1)
        stats['roi_area_ratio'] = float((x1 - x0) * (y1 - y0)) / float(height * width)
    stats['roi_enabled'] = enabled
    stats['method'] = method
    stats['foreground_ratio'] = float(np.count_nonzero(mask)) / float(mask.size)
    state.flags['segment_stats'] = stats
    return mask


def _estimate_scene_shift(ycrcb, state):
    step = max(1, int(SCENE_ESTIMATE_STEP))
    if roi_enabled_now(state):
        x0, y0, x1, y1 = roi_rect_pixels(ycrcb.shape[:2])
        x0 = (x0 + step - 1) // step * step
        y0 = (y0 + step - 1) // step * step
        sample = ycrcb[y0:y1:step, x0:x1:step]
    else:
        sample = ycrcb[::step, ::step]
    if not sample.size:
        return (0.0, 0.0)
    bg_cr = float(np.median(sample[:, :, 1].astype(np.float32)))
    bg_cb = float(np.median(sample[:, :, 2].astype(np.float32)))
    limit = float(SCENE_SHIFT_CLAMP)
    d_cr = max(-limit, min(limit, bg_cr - float(SCENE_BASELINE_CR)))
    d_cb = max(-limit, min(limit, bg_cb - float(SCENE_BASELINE_CB)))
    state.flags['scene_bg_estimate'] = {'mode': 'roi', 'tiles': 0, 'tile_cr': [], 'tile_cb': [], 'bg_cr': round(bg_cr, 1), 'bg_cb': round(bg_cb, 1), 'spread_cr': None, 'spread_cb': None}
    return (d_cr, d_cb)


def step_color_threshold(frame: np.ndarray, state) -> np.ndarray:
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    y_channel = ycrcb[:, :, 0]
    cr_channel = ycrcb[:, :, 1]
    cb_channel = ycrcb[:, :, 2]
    cr_lo, cr_hi = _normalized_pair(CR_LOW, CR_HIGH, 'CR_LOW/CR_HIGH')
    cb_lo, cb_hi = _normalized_pair(CB_LOW, CB_HIGH, 'CB_LOW/CB_HIGH')
    y_lo, y_hi = _normalized_pair(Y_LOW, Y_HIGH, 'Y_LOW/Y_HIGH')
    scene_dcr = scene_dcb = 0.0
    scene_dcr, scene_dcb = _estimate_scene_shift(ycrcb, state)
    cr_lo = int(round(max(0.0, min(255.0, cr_lo + scene_dcr))))
    cr_hi = int(round(max(0.0, min(255.0, cr_hi + scene_dcr))))
    cb_lo = int(round(max(0.0, min(255.0, cb_lo + scene_dcb))))
    cb_hi = int(round(max(0.0, min(255.0, cb_hi + scene_dcb))))
    cr_lo, cr_hi = _normalized_pair(cr_lo, cr_hi, '校准后 Cr')
    cb_lo, cb_hi = _normalized_pair(cb_lo, cb_hi, '校准后 Cb')
    mask = cv2.inRange(ycrcb, (y_lo, cr_lo, cb_lo), (y_hi, cr_hi, cb_hi))
    state.flags['scene_shift'] = (scene_dcr, scene_dcb)
    state.flags['scene_box'] = (cr_lo, cr_hi, cb_lo, cb_hi)
    state.flags['segment_ycrcb'] = ycrcb
    state.flags['segment_source_bgr'] = frame
    return _finish_mask(mask, state, 'color')


def _fill_internal_holes(binary: np.ndarray) -> np.ndarray:
    height, width = binary.shape[:2]
    padded = cv2.copyMakeBorder(binary, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    flood_mask = np.zeros((height + 4, width + 4), np.uint8)
    cv2.floodFill(padded, flood_mask, (0, 0), 255)
    holes = cv2.bitwise_not(padded)[1:-1, 1:-1]
    return cv2.bitwise_or(binary, holes)


def roi_area_pixels(shape_hw: tuple, enabled: bool=None) -> int:
    if enabled is None:
        enabled = ROI_ENABLED
    height, width = (int(shape_hw[0]), int(shape_hw[1]))
    if not enabled:
        return height * width
    x0, y0, x1, y1 = roi_rect_pixels((height, width))
    return int((x1 - x0) * (y1 - y0))


def step_mask_cleanup(frame: np.ndarray, state) -> np.ndarray:
    mask = frame
    height, width = mask.shape[:2]
    enabled = roi_enabled_now(state)
    kernel_size = int(CLEANUP_OPEN_KERNEL)
    if kernel_size >= 3:
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    removed_components = 0
    kept_components = 0
    min_area_used = 0.0
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if count > 1:
        areas = stats[1:, cv2.CC_STAT_AREA].astype(np.float64)
        largest = float(areas.max())
        absolute_min = float(CLEANUP_MIN_AREA_RATIO) * float(roi_area_pixels((height, width), enabled))
        relative_min = float(CLEANUP_MIN_AREA_VS_LARGEST) * largest
        min_area_used = max(1.0, absolute_min, relative_min)
        lut = np.zeros(count, np.uint8)
        for label in range(1, int(count)):
            if float(stats[label, cv2.CC_STAT_AREA]) >= min_area_used:
                lut[label] = 255
                kept_components += 1
            else:
                removed_components += 1
        mask = lut[labels] if kept_components else np.zeros_like(mask)
    filled_pixels = 0
    if CLEANUP_FILL_HOLES and kept_components:
        before = int(np.count_nonzero(mask))
        mask = _fill_internal_holes(mask)
        filled_pixels = int(np.count_nonzero(mask)) - before
    segment_stats = dict(state.flags.get('segment_stats') or {})
    segment_stats['cleaned'] = True
    segment_stats['foreground_ratio'] = float(np.count_nonzero(mask)) / float(mask.size)
    segment_stats['removed_components'] = removed_components
    segment_stats['kept_components'] = kept_components
    segment_stats['min_area_used'] = min_area_used
    segment_stats['filled_pixels'] = filled_pixels
    state.flags['segment_stats'] = segment_stats
    return mask


def _estimate_palm_robust(mask: np.ndarray):
    area = int(np.count_nonzero(mask))
    if area <= 0:
        return ((0, 0), 0.0)
    scale = float(np.sqrt(float(area) / np.pi))
    size = int(round(float(RECON_PALM_CLOSE_RATIO) * scale))
    if size >= 3:
        if size % 2 == 0:
            size += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        work = _close_for_palm(mask, kernel)
    else:
        work = mask
    distance = cv2.distanceTransform(work, cv2.DIST_L2, 5)
    _, radius, _, location = cv2.minMaxLoc(distance)
    if not np.isfinite(radius) or radius > mask.shape[0] + mask.shape[1]:
        return ((0, 0), 0.0)
    return ((int(location[0]), int(location[1])), float(radius))


def step_palm_reconstruct(frame: np.ndarray, state) -> np.ndarray:
    mask = frame
    height, width = mask.shape[:2]
    ycrcb = state.flags.get('segment_ycrcb')
    if ycrcb is None:
        state.flags['reconstruction_note'] = '[%s] 没有可用的 YCrCb 图（上游没跑颜色阈值？），跳过重建' % RECON_PLUGIN_NAME
        return frame
    distance = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    _, radius, _, location = cv2.minMaxLoc(distance)
    if not np.isfinite(radius) or radius > height + width:
        state.flags['failure_reason'] = 'invalid_segmentation_no_background'
        return frame
    if radius < float(RECON_MIN_RADIUS_PX):
        state.flags['reconstruction_note'] = '[%s] 掌心半径只有 %.1f，太小，跳过重建' % (RECON_PLUGIN_NAME, radius)
        return frame
    cx, cy = (int(location[0]), int(location[1]))
    seed = distance >= float(RECON_SEED_DT_FACTOR) * radius
    seed_pixels = int(np.count_nonzero(seed))
    if seed_pixels < int(RECON_MIN_SEED_PIXELS):
        state.flags['reconstruction_note'] = '[%s] 种子只有 %d px，太小，跳过重建' % (RECON_PLUGIN_NAME, seed_pixels)
        return frame

    def median_of(plane):
        return float(np.median(plane[seed]))
    cr_plane = ycrcb[:, :, 1]
    cb_plane = ycrcb[:, :, 2]
    y_plane = ycrcb[:, :, 0]
    cr0 = median_of(cr_plane)
    cb0 = median_of(cb_plane)
    y0 = median_of(y_plane)
    cr_lo = int(round(max(0.0, cr0 - float(RECON_TOL_CR))))
    cr_hi = int(round(min(255.0, cr0 + float(RECON_TOL_CR))))
    cb_lo = int(round(max(0.0, cb0 - float(RECON_TOL_CB))))
    cb_hi = int(round(min(255.0, cb0 + float(RECON_TOL_CB))))
    y_lo = int(round(max(0.0, y0 - float(RECON_TOL_Y_DOWN))))
    allowed = cv2.inRange(ycrcb, (y_lo, cr_lo, cb_lo), (255, cr_hi, cb_hi))
    cv2.bitwise_and(allowed, mask, dst=allowed)
    count, labels = cv2.connectedComponents(allowed, connectivity=8)
    seed_labels = set((int(v) for v in np.unique(labels[seed]))) - {0}
    if not seed_labels:
        state.flags['reconstruction_note'] = '[%s] 允许区和种子不相连（容差是不是调太小了？），跳过重建' % RECON_PLUGIN_NAME
        return frame
    lookup = np.zeros(count, np.uint8)
    for label in seed_labels:
        lookup[label] = 255
    result = lookup[labels]
    result = _fill_internal_holes(result)
    (palm_cx, palm_cy), palm_radius = _estimate_palm_robust(result)
    if palm_radius == 0.0:
        state.flags['failure_reason'] = 'invalid_palm_distance'
        return result
    pass
    segment_stats = dict(state.flags.get('segment_stats') or {})
    segment_stats['reconstructed'] = True
    segment_stats['palm_center'] = (palm_cx, palm_cy)
    segment_stats['palm_radius'] = float(palm_radius)
    segment_stats['palm_radius_raw'] = float(radius)
    segment_stats['seed_pixels'] = seed_pixels
    segment_stats['color_base'] = (cr0, cb0, y0)
    segment_stats['foreground_ratio'] = float(np.count_nonzero(result)) / float(result.size)
    state.flags['segment_stats'] = segment_stats
    return result


def _to_local_points(points, origin, up, scale) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if pts.size == 0:
        return np.zeros((0, 2), dtype=np.float64)
    right = np.array([-float(up[1]), float(up[0])], dtype=np.float64)
    delta = pts - np.asarray(origin, dtype=np.float64).reshape(1, 2)
    safe = float(scale) if abs(float(scale)) > 1e-06 else 1.0
    return np.stack([delta @ right / safe, -(delta @ up) / safe], axis=1)


def _contour_point(contour, index) -> np.ndarray:
    count = int(contour.shape[0])
    return contour[int(index) % count].reshape(2).astype(np.float64)


def _turning_ratio(contour, index: int, window: int) -> float:
    here = _contour_point(contour, index)
    before = _contour_point(contour, index - window)
    after = _contour_point(contour, index + window)
    chord = after - before
    chord_length = float(np.linalg.norm(chord))
    if chord_length < 1e-06:
        return 0.0
    offset = here - before
    perpendicular = abs(float(offset[0] * chord[1] - offset[1] * chord[0])) / chord_length
    return float(np.clip(perpendicular / chord_length, 0.0, 1.0))


def _contour_angle(contour, index: int, window: int) -> float:
    here = _contour_point(contour, index)
    before = _contour_point(contour, index - window) - here
    after = _contour_point(contour, index + window) - here
    na = float(np.linalg.norm(before))
    nb = float(np.linalg.norm(after))
    if na < 1e-06 or nb < 1e-06:
        return 180.0
    cos_value = float(np.clip(np.dot(before, after) / (na * nb), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_value)))


def _tip_sharpness(contour, index: int, window: int) -> tuple:
    windows = [max(2, int(round(window * factor))) for factor in (0.6, 1.0, 1.5)]
    ratios = [_turning_ratio(contour, index, w) for w in windows]
    angles = [_contour_angle(contour, index, w) for w in windows]
    return (float(np.max(ratios)), float(np.median(angles)))


def _sample_plane(plane: np.ndarray, point: np.ndarray) -> float:
    height, width = plane.shape[:2]
    x = int(np.clip(round(float(point[0])), 0, width - 1))
    y = int(np.clip(round(float(point[1])), 0, height - 1))
    return float(plane[y, x])


def _protrusion_length(thickness: np.ndarray, tip_point, palm_center, radius: float) -> float:
    tip = np.asarray(tip_point, dtype=np.float64).reshape(2)
    palm = np.asarray(palm_center, dtype=np.float64).reshape(2)
    total = float(np.linalg.norm(palm - tip))
    if total < 1e-06:
        return 0.0
    steps = int(np.clip(total, 8, 400))
    fractions = np.linspace(0.0, 1.0, steps)
    xs = tip[0] + (palm[0] - tip[0]) * fractions
    ys = tip[1] + (palm[1] - tip[1]) * fractions
    height, width = thickness.shape[:2]
    xi = np.clip(np.round(xs).astype(np.int32), 0, width - 1)
    yi = np.clip(np.round(ys).astype(np.int32), 0, height - 1)
    values = thickness[yi, xi]
    hit = np.nonzero(values >= float(RECOG_TIP_PALM_THICK) * float(radius))[0]
    if hit.size == 0:
        return total / float(radius)
    return float(fractions[int(hit[0])]) * total / float(radius)


def _estimate_wrist(mask: np.ndarray, palm_center, radius: float):
    return _estimate_wrist_legacy(mask, palm_center, radius)


def _estimate_wrist_legacy(mask: np.ndarray, palm_center, radius: float):
    height, width = mask.shape[:2]
    row_widths = np.count_nonzero(mask, axis=1)
    rows = np.nonzero(row_widths)[0]
    if rows.size == 0:
        return None
    bottom_row = int(rows[-1])
    cy = int(round(float(palm_center[1])))
    start = min(bottom_row, cy + int(round(0.8 * float(radius))))
    if bottom_row - start < 3:
        return None
    segment = row_widths[start:bottom_row + 1].astype(np.float32)
    if segment.size < 3:
        return None
    kernel = max(3, int(round(0.2 * float(radius))) | 1)
    smoothed = cv2.blur(segment.reshape(-1, 1), (1, kernel)).ravel()
    peak = float(smoothed.max())
    if peak <= 0.0:
        return None
    left = np.concatenate([[smoothed[0]], smoothed[:-1]])
    right = np.concatenate([smoothed[1:], [smoothed[-1]]])
    is_min = (smoothed <= left) & (smoothed <= right) & (smoothed < 0.88 * peak)
    candidates = np.nonzero(is_min)[0]
    best = int(candidates[0]) if candidates.size else int(np.argmin(smoothed))
    wrist_row = start + best
    xs = np.nonzero(mask[wrist_row])[0]
    if xs.size < int(RECOG_WRIST_MIN_PIXELS):
        return None
    center = np.array([0.5 * (float(xs.min()) + float(xs.max())), float(wrist_row)], dtype=np.float64)
    if center[1] <= float(palm_center[1]):
        return None
    return center


def _truncate_forearm(mask: np.ndarray, palm_center, up, radius: float) -> np.ndarray:
    height, width = mask.shape[:2]
    far = float(max(height, width)) * 2.0
    origin = np.asarray(palm_center, dtype=np.float64) - np.asarray(up, dtype=np.float64) * (float(RECOG_TRUNCATE_FACTOR) * float(radius))
    along = np.array([-float(up[1]), float(up[0])], dtype=np.float64)
    polygon = np.array([origin + along * far, origin - along * far, origin - along * far + np.asarray(up) * far, origin + along * far + np.asarray(up) * far], dtype=np.int32)
    allowed = np.zeros((height, width), dtype=np.uint8)
    cv2.fillConvexPoly(allowed, polygon, 255)
    return cv2.bitwise_and(mask, allowed)


def _detect_fingertips(contour, mask: np.ndarray, palm_center, radius: float, up):
    count = int(contour.shape[0])
    if count < 8:
        return ([], None)
    points = contour.reshape(count, 2).astype(np.float64)
    distances = np.linalg.norm(points - np.asarray(palm_center).reshape(1, 2), axis=1)
    window = max(2, int(round(RECOG_TIP_PEAK_WINDOW * float(radius))))
    padded = np.pad(distances, (window, window), mode='wrap').reshape(1, -1)
    kernel = np.ones((1, 2 * window + 1), dtype=np.uint8)
    floor_max = cv2.dilate(padded, kernel)[0, window:-window]
    floor_min = cv2.erode(padded, kernel)[0, window:-window]
    keep = distances >= RECOG_TIP_MIN_DISTANCE * float(radius)
    keep &= floor_max <= distances + 1e-09
    keep &= distances - floor_min >= RECOG_TIP_MIN_PROMINENCE * float(radius)
    candidate_indices = np.nonzero(keep)[0]
    if candidate_indices.size == 0:
        return ([], distances)
    thickness = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    candidates = []
    for raw_index in candidate_indices:
        index = int(raw_index)
        ratio, angle = _tip_sharpness(contour, index, window)
        inner = points[index] + (np.asarray(palm_center) - points[index]) * float(RECOG_TIP_THICKNESS_SAMPLE)
        thickness_ratio = _sample_plane(thickness, inner) / float(radius)
        if ratio < RECOG_TIP_MIN_TURN_RATIO and thickness_ratio > RECOG_TIP_THICKNESS_MAX:
            continue
        if angle > RECOG_TIP_MAX_ANGLE_DEG:
            continue
        if angle < RECOG_TIP_MIN_ANGLE_DEG and thickness_ratio < RECOG_TIP_MIN_THICKNESS:
            continue
        protrusion = _protrusion_length(thickness, points[index], palm_center, radius)
        if protrusion < float(RECOG_TIP_PROTRUSION_MIN):
            continue
        candidates.append({'point': points[index].copy(), 'distance': float(distances[index]), 'turn_ratio': float(ratio), 'angle_deg': float(angle), 'thickness_ratio': float(thickness_ratio), 'protrusion_ratio': float(protrusion)})
    candidates.sort(key=lambda item: -item['distance'])
    merged = []
    for item in candidates:
        if any((float(np.linalg.norm(item['point'] - kept['point'])) < RECOG_TIP_DEDUP * float(radius) for kept in merged)):
            continue
        merged.append(item)
        if len(merged) >= int(RECOG_TIP_MAX_COUNT):
            break
    result = []
    for item in merged:
        local = _to_local_points(item['point'].reshape(1, 2), palm_center, up, radius)[0]
        if float(local[1]) > float(RECOG_TIP_PALM_SIDE_MAX):
            continue
        item['local'] = (float(local[0]), float(local[1]))
        result.append(item)
    result.sort(key=lambda item: float(np.arctan2(item['local'][1], item['local'][0])))
    return (result, distances)


def _detect_gaps(contour, tips, palm_center, radius: float, up):
    if len(tips) < 2 or contour is None:
        return []
    count = int(contour.shape[0])
    points = contour.reshape(count, 2).astype(np.float64)
    palm_distances = np.linalg.norm(points - np.asarray(palm_center).reshape(1, 2), axis=1)
    gaps = []
    for first, second in zip(tips, tips[1:]):
        first_point = np.asarray(first['point'], dtype=np.float64)
        second_point = np.asarray(second['point'], dtype=np.float64)
        first_index = int(np.argmin(np.linalg.norm(points - first_point.reshape(1, 2), axis=1)))
        second_index = int(np.argmin(np.linalg.norm(points - second_point.reshape(1, 2), axis=1)))
        forward = (second_index - first_index) % count
        backward = (first_index - second_index) % count
        if forward <= backward:
            indices = [(first_index + step) % count for step in range(forward + 1)]
        else:
            indices = [(first_index - step) % count for step in range(backward + 1)]
        if len(indices) < 3:
            continue
        arc = points[indices]
        arc_distances = palm_distances[indices]
        inner = slice(1, -1)
        best = int(np.argmin(arc_distances[inner])) + 1
        valley = arc[best]
        chord = second_point - first_point
        span = float(np.linalg.norm(chord))
        if span < 1e-06:
            continue
        depth = abs(float(np.cross(chord / span, valley - first_point)))
        if depth < RECOG_GAP_MIN_DEPTH * float(radius):
            continue
        local = _to_local_points(valley.reshape(1, 2), palm_center, up, radius)[0]
        local_first = _to_local_points(first_point.reshape(1, 2), palm_center, up, radius)[0]
        local_second = _to_local_points(second_point.reshape(1, 2), palm_center, up, radius)[0]
        v1 = np.asarray(local_first, dtype=np.float64) - np.asarray(local, dtype=np.float64)
        v2 = np.asarray(local_second, dtype=np.float64) - np.asarray(local, dtype=np.float64)
        n1 = float(np.linalg.norm(v1))
        n2 = float(np.linalg.norm(v2))
        if n1 < 1e-06 or n2 < 1e-06:
            continue
        angle = float(np.degrees(np.arccos(float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)))))
        if angle > RECOG_GAP_MAX_ANGLE_DEG:
            continue
        gaps.append({'point': valley.copy(), 'local': (float(local[0]), float(local[1])), 'depth_ratio': float(depth) / float(radius), 'angle_deg': float(angle), 'pair': (tuple(np.round(first_point).astype(int)), tuple(np.round(second_point).astype(int)))})
    gaps.sort(key=lambda item: float(np.arctan2(item['local'][1], item['local'][0])))
    return gaps[:int(RECOG_GAP_MAX_COUNT)]


def _build_fingers(tips, gaps, palm_center, radius: float, up):
    fingers = []
    if not tips:
        return fingers
    for position, tip in enumerate(tips):
        tip_local = np.asarray(tip['local'], dtype=np.float64)
        left = gaps[position - 1]['point'] if 0 <= position - 1 < len(gaps) else None
        right = gaps[position]['point'] if 0 <= position < len(gaps) else None
        if left is not None and right is not None:
            base = (np.asarray(left, dtype=np.float64) + np.asarray(right, dtype=np.float64)) / 2.0
        elif left is not None:
            base = np.asarray(left, dtype=np.float64)
        elif right is not None:
            base = np.asarray(right, dtype=np.float64)
        else:
            direction = np.asarray(tip['point'], dtype=np.float64) - np.asarray(palm_center)
            norm = float(np.linalg.norm(direction))
            direction = direction / norm if norm > 1e-06 else np.asarray(up, dtype=np.float64)
            base = np.asarray(palm_center, dtype=np.float64) + direction * float(radius)
        length = float(np.linalg.norm(np.asarray(tip['point'], dtype=np.float64) - base))
        fingers.append({'index': position + 1, 'tip': np.asarray(tip['point'], dtype=np.float64).copy(), 'base': base.copy(), 'tip_local': (float(tip_local[0]), float(tip_local[1])), 'length_ratio': length / float(radius), 'angle_deg': float(np.degrees(np.arctan2(tip_local[1], tip_local[0])))})
    return fingers


def _recognize(mask: np.ndarray, state) -> Optional[dict]:
    if state.flags.get('failure_reason') or mask is None or int(np.count_nonzero(mask)) == 0:
        return None
    segment_stats = state.flags.get('segment_stats') or {}
    palm_center = segment_stats.get('palm_center')
    radius = segment_stats.get('palm_radius')
    if palm_center is None or not radius:
        palm_center, radius = _estimate_palm_robust(mask)
        if float(radius) < float(RECON_MIN_RADIUS_PX):
            return None
    palm_center = np.array([float(palm_center[0]), float(palm_center[1])], dtype=np.float64)
    radius = float(radius)
    wrist = _estimate_wrist(mask, palm_center, radius)
    if wrist is None:
        up = np.array([0.0, -1.0], dtype=np.float64)
    else:
        direction = palm_center - wrist
        norm = float(np.linalg.norm(direction))
        up = direction / norm if norm > 1e-06 else np.array([0.0, -1.0], dtype=np.float64)
    arm_axis = None
    hand = _truncate_forearm(mask, palm_center, up, radius)
    if int(np.count_nonzero(hand)) == 0:
        hand = mask
    contours, _ = cv2.findContours(hand, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if int(contour.shape[0]) < 8:
        return None
    tips, _profile = _detect_fingertips(contour, hand, palm_center, radius, up)
    gaps = _detect_gaps(contour, tips, palm_center, radius, up)
    fingers = _build_fingers(tips, gaps, palm_center, radius, up)
    hull = cv2.convexHull(contour)
    contour_area = float(cv2.contourArea(contour))
    hull_area = float(cv2.contourArea(hull)) if hull is not None else 0.0
    solidity = float(np.clip(contour_area / hull_area, 0.0, 1.0)) if hull_area > 0 else 0.0
    angle_span_deg = 0.0
    if len(tips) >= 2:
        tip_angles = np.array([float(np.arctan2(t['local'][1], t['local'][0])) for t in tips], dtype=np.float64)
        ordered = np.sort(tip_angles)
        circular_gaps = np.diff(np.concatenate([ordered, [ordered[0] + 2.0 * np.pi]]))
        angle_span_deg = float(np.degrees(2.0 * np.pi - float(np.max(circular_gaps))))
    if roi_enabled_now(state):
        rx0, ry0, rx1, ry1 = roi_rect_pixels((hand.shape[0], hand.shape[1]))
    else:
        rx0, ry0, rx1, ry1 = (0, 0, hand.shape[1], hand.shape[0])
    span_x = float(max(1, rx1 - rx0))
    span_y = float(max(1, ry1 - ry0))
    palm_rel = (float((palm_center[0] - rx0) / span_x), float((palm_center[1] - ry0) / span_y))
    mean_length = float(np.mean([f['length_ratio'] for f in fingers])) if fingers else 0.0
    result = {'palm_center': palm_center, 'palm_radius': radius, 'wrist': wrist, 'up': up, 'hand_mask': hand, 'contour': contour, 'hull': hull, 'solidity': solidity, 'angle_span_deg': angle_span_deg, 'palm_rel': palm_rel, 'mean_length_ratio': mean_length, 'tips': tips, 'gaps': gaps, 'fingers': fingers, 'finger_count': len(tips)}
    states, extensions, angles_used, match_detail, expect_len = _finger_states(hand, palm_center, up, radius, tips, gaps, contour=contour)
    result['finger_states'] = states
    result['finger_extensions'] = extensions
    result['finger_ray_angles'] = angles_used
    result['finger_expect_len'] = expect_len
    result['finger_match'] = match_detail
    result['finger_names'] = list(FINGER_NAMES)
    result['finger_text'] = finger_states_text(states)
    return result


def _radial_profile(mask: np.ndarray, palm_center, up, radius: float):
    up = np.asarray(up, dtype=np.float64).reshape(2)
    norm = float(np.linalg.norm(up))
    if norm > 1e-09:
        up = up / norm
    right = np.array([-up[1], up[0]], dtype=np.float64)
    origin = np.asarray(palm_center, dtype=np.float64).reshape(2)
    angles = np.arange(float(FINGER_PROFILE_MIN_DEG), float(FINGER_PROFILE_MAX_DEG) + 1e-09, float(FINGER_PROFILE_STEP_DEG), dtype=np.float64)
    r_step = max(0.001, float(FINGER_PROBE_R_STEP))
    rs = np.arange(float(FINGER_PROBE_R_MIN), float(FINGER_PROBE_R_MAX) + 1e-09, r_step, dtype=np.float64)
    t = np.radians(angles)[:, None]
    dirs = np.cos(t) * up[None, :] + np.sin(t) * right[None, :]
    pts = origin[None, None, :] + dirs[:, None, :] * (rs[None, :, None] * radius)
    height, width = mask.shape[:2]
    xs = np.round(pts[:, :, 0]).astype(np.int32)
    ys = np.round(pts[:, :, 1]).astype(np.int32)
    inside = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
    hits = inside & (mask[np.clip(ys, 0, height - 1), np.clip(xs, 0, width - 1)] > 0)
    gap_max = max(1, int(round(float(FINGER_RUN_GAP_TOLERANCE) / r_step)))
    count_a, count_r = hits.shape
    profile = np.zeros(count_a, dtype=np.float64)
    for a in range(count_a):
        row = hits[a]
        if not bool(row[0]):
            continue
        last, gap = (0, 0)
        for i in range(count_r):
            if row[i]:
                gap = 0
                last = i
            else:
                gap += 1
                if gap > gap_max:
                    break
        profile[a] = float(rs[last])
    return (angles, profile)


def _nominal_boundaries(combo):
    runs = []
    for i in combo:
        if runs and i == runs[-1][-1] + 1:
            runs[-1].append(i)
        else:
            runs.append([i])
    out = []
    for idx, run in enumerate(runs):
        if idx > 0:
            prev = runs[idx - 1]
            vals = [float(FINGER_WEB_X[k]) for k in range(prev[-1], run[0])]
            if vals:
                out.append(sum(vals) / len(vals))
        for a, _b in zip(run, run[1:]):
            out.append(float(FINGER_WEB_X[a]))
    return out


def _assign_finger_slots(tip_angles, tip_lengths=None, web_xs=None):
    template = np.asarray(FINGER_TEMPLATE_DEG, dtype=np.float64)
    order = sorted(range(len(tip_angles)), key=lambda j: -float(tip_angles[j]))
    ta = [float(tip_angles[j]) for j in order]
    count = len(ta)
    lens = [None] * count
    if tip_lengths is not None:
        for pos, j in enumerate(order):
            try:
                lens[pos] = float(tip_lengths[j])
            except (TypeError, ValueError, IndexError):
                pass
    lo, hi = map(float, FINGER_THUMB_RANGE_DEG)
    has_thumb = bool(count and lo < ta[0] < hi)
    anchor = 'thumb' if has_thumb else 'palm_axis'
    candidates = []
    for mask in range(32):
        combo = tuple((i for i in range(5) if mask >> i & 1))
        if len(combo) != count:
            continue
        if has_thumb and 0 not in combo:
            continue
        if not has_thumb and count < 5 and (0 in combo):
            continue
        candidates.append(combo)
    shifts = [None] if has_thumb else list(np.arange(-float(FINGER_NONTHUMB_SHIFT_DEG), float(FINGER_NONTHUMB_SHIFT_DEG) + 1e-09, float(FINGER_NONTHUMB_SHIFT_STEP)))
    if count == 0:
        shifts = [0.0]
    obs_webs = sorted([float(v) for v in web_xs or []], reverse=True)
    ranked = []
    best = None
    for combo in candidates:
        exp_webs = _nominal_boundaries(combo)
        web_sse = float(FINGER_WEB_COUNT_PENALTY) if len(obs_webs) != len(exp_webs) else 0.0
        web_sse += sum(((o - e) ** 2 for o, e in zip(obs_webs, exp_webs)))
        non_thumb = [i for i in combo if i != 0]
        penalty = float(FINGER_CONTIGUOUS_PENALTY) if not has_thumb and non_thumb and (non_thumb != list(range(1, 1 + len(non_thumb)))) else 0.0
        combo_best = None
        for scale in map(float, FINGER_SEARCH_SCALES):
            len_sse = sum(((ln - scale * float(FINGER_TEMPLATE_LEN[slot])) ** 2 for slot, ln in zip(combo, lens) if ln is not None))
            for shift in shifts:
                delta = ta[0] - scale * float(template[0]) if shift is None else float(shift)
                predicted = [delta + scale * float(template[i]) for i in combo]
                sse = float(sum(((a - p) ** 2 for a, p in zip(ta, predicted))))
                costs = {'angle': float(FINGER_W_ANGLE) * sse, 'length': float(FINGER_W_LEN) * len_sse, 'web': float(FINGER_W_WEB) * web_sse, 'scale': float(FINGER_W_SCALE) * (scale - 1.0) ** 2, 'contiguity': penalty}
                total = sum(costs.values())
                detail = {'delta_deg': delta, 'sse': sse, 'len_sse': len_sse, 'web_sse': web_sse, 'penalty': penalty, 'anchor': anchor, 'slots': list(combo), 'scale': scale, 'total': total, 'costs': costs, 'predicted_angles_deg': predicted, 'observed_web_x': obs_webs, 'expected_web_x': exp_webs}
                if combo_best is None or total < combo_best['total'] - 1e-09:
                    combo_best = detail
                if best is None or total < best['total'] - 1e-09:
                    best = detail
        if combo_best is not None:
            ranked.append(combo_best)
    if best is None:
        return ([False] * 5, order, anchor, {'delta_deg': 0.0, 'sse': None, 'valid_assignment': False, 'candidates': []})
    ranked.sort(key=lambda item: item['total'])
    detail = dict(best)
    detail['valid_assignment'] = True
    detail['candidates'] = ranked
    detail['score_margin'] = ranked[1]['total'] - ranked[0]['total'] if len(ranked) > 1 else None
    states = [i in best['slots'] for i in range(5)]
    return (states, order, anchor, detail)


def _finger_states(mask: np.ndarray, palm_center, up, radius: float, tips, web_gaps=None, contour=None):
    geometry_failure = None
    if contour is None:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contour = max(contours, key=cv2.contourArea) if contours else None
    measured, geometry_failure = FG.recognize_fingers(mask, contour, palm_center, up, radius, tips or [])
    if measured is not None:
        extensions = [0.0] * 5
        angles_used = [0.0] * 5
        for feature in measured['features']:
            slot = feature['slot']
            extensions[slot] = feature['length_r']
            x, y = feature['tip_local']
            angles_used[slot] = float(np.degrees(np.arctan2(x, -y)))
        measured['anchor'] = 'local_finger_axis'
        return (measured['states'], extensions, angles_used, measured, [0.0] * 5)
    tip_angles = []
    tip_lengths = []
    for t in tips or []:
        x = float(t['local'][0])
        y = float(t['local'][1])
        tip_angles.append(float(np.degrees(np.arctan2(x, -y))))
        tip_lengths.append(float(np.hypot(x, y)))
    web_xs = sorted([float(g['local'][0]) for g in web_gaps or []], reverse=True)
    states, _order, anchor, detail = _assign_finger_slots(tip_angles, tip_lengths, web_xs)
    best_total = float(detail.get('total', float('inf')))
    mirrored = False
    detail['mirrored'] = mirrored
    angles, profile = _radial_profile(mask, palm_center, up, radius)
    delta = float(detail.get('delta_deg', 0.0))
    scale_fit = float(detail.get('scale', 1.0))
    used = [delta + scale_fit * float(FINGER_TEMPLATE_DEG[i]) for i in range(5)]
    if mirrored:
        used = [-a for a in used]
    extensions = [float(np.interp(a, angles, profile)) for a in used]
    expected = [float(FINGER_TEMPLATE_LEN[i]) * scale_fit for i in range(5)]
    if anchor != 'thumb' and (not states[0]):
        if extensions[0] >= float(FINGER_EXTENDED_MIN_R):
            states[0] = True
            detail['thumb_from_profile'] = True
    detail['anchor'] = anchor
    detail['tip_angles_deg'] = [round(a, 1) for a in tip_angles]
    detail['expect_len'] = [round(v, 2) for v in expected]
    detail['method'] = 'angle_template_legacy'
    if geometry_failure is not None:
        detail['geometry_fallback_reason'] = geometry_failure
    return (states, extensions, used, detail, expected)


def finger_states_text(states) -> str:
    names = [FINGER_LABELS_CN[i] for i, s in enumerate(states) if s]
    return '+'.join(names) if names else '（全部蜷起）'


def detect_hand_rotation(frame: np.ndarray) -> float:
    h, w = frame.shape[:2]
    if h < 32 or w < 32:
        return 0.0
    scale = float(AUTO_ROTATE_DOWNSCALE) / max(1, w)
    small = cv2.resize(frame, (max(32, int(w * scale)), max(32, int(h * scale))), interpolation=cv2.INTER_AREA)
    ycc = cv2.cvtColor(small, cv2.COLOR_BGR2YCrCb)
    slack = float(AUTO_ROTATE_COLOR_SLACK)
    mask = cv2.inRange(ycc, (int(Y_LOW), int(CR_LOW - slack), int(CB_LOW - slack)), (int(Y_HIGH), int(CR_HIGH + slack), int(CB_HIGH + slack)))
    if int(np.count_nonzero(mask)) < 64:
        return 0.0
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if count <= 1:
        return 0.0
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    area = int(stats[biggest, cv2.CC_STAT_AREA])
    if area < float(AUTO_ROTATE_MIN_AREA) * mask.size:
        return 0.0
    blob = np.uint8(labels == biggest) * 255
    ys, xs = np.nonzero(blob)
    bh, bw = blob.shape[:2]
    edge_lines = {'left': blob[:, 0], 'right': blob[:, bw - 1], 'top': blob[0, :], 'bottom': blob[bh - 1, :]}
    best_edge, best_cnt, best_pos = (None, 0, 0.0)
    for name, line in edge_lines.items():
        idx = np.nonzero(line)[0]
        if idx.size > best_cnt:
            best_cnt = int(idx.size)
            best_edge = name
            best_pos = float(idx.mean())
    if best_edge is None or best_cnt < int(AUTO_ROTATE_EDGE_MIN_PIXELS):
        return 0.0
    if best_cnt > float(AUTO_ROTATE_EDGE_MAX_FRAC) * max(bh, bw):
        return 0.0
    centroid = np.array([float(xs.mean()), float(ys.mean())])
    if best_edge == 'left':
        target = np.array([0.0, best_pos])
    elif best_edge == 'right':
        target = np.array([float(bw - 1), best_pos])
    elif best_edge == 'top':
        target = np.array([best_pos, 0.0])
    else:
        target = np.array([best_pos, float(bh - 1)])
    arm = target - centroid
    n = float(np.linalg.norm(arm))
    if n < 1e-06:
        return 0.0
    arm = arm / n
    cur = float(np.degrees(np.arctan2(arm[0], arm[1])))
    while cur > 180.0:
        cur -= 360.0
    while cur < -180.0:
        cur += 360.0
    return float(-cur)


def auto_rotate_frame(frame):
    angle = detect_hand_rotation(frame)
    if abs(angle) < 1.0:
        return (frame, angle)
    h, w = frame.shape[:2]
    center = (w * 0.5, h * 0.5)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos_a, sin_a = (abs(matrix[0, 0]), abs(matrix[0, 1]))
    nw = int(h * sin_a + w * cos_a)
    nh = int(h * cos_a + w * sin_a)
    matrix[0, 2] += nw * 0.5 - center[0]
    matrix[1, 2] += nh * 0.5 - center[1]
    return (cv2.warpAffine(frame, matrix, (nw, nh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE), angle)


def _close_for_palm(mask, kernel):
    """Close only the foreground neighbourhood, preserving original borders.

    Dilation followed by erosion has a dependency radius of 2*r. Keep 2*r+1
    pixels of zero halo, clip only at actual image edges, then restore a full
    canvas. The following distance transform keeps its original dimensions
    and floating-point execution path.
    """
    x, y, width, height = cv2.boundingRect(mask)
    if not width or not height:
        return mask.copy()
    pad = kernel.shape[0]
    x0, y0 = (max(0, x - pad), max(0, y - pad))
    x1 = min(mask.shape[1], x + width + pad)
    y1 = min(mask.shape[0], y + height + pad)
    if x0 == 0 and y0 == 0 and (x1 == mask.shape[1]) and (y1 == mask.shape[0]):
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    work = np.zeros_like(mask)
    work[y0:y1, x0:x1] = cv2.morphologyEx(mask[y0:y1, x0:x1], cv2.MORPH_CLOSE, kernel)
    return work

@dataclass
class _Frame:
    flags: dict = field(default_factory=dict)


@dataclass
class RecognitionResult:
    """Coordinates refer to image, after the same resize/rotation as before."""
    image: np.ndarray
    mask: np.ndarray
    recognition: Optional[dict]
    rotation_deg: float
    stats: dict
    timings_ms: dict
    failure_reason: Optional[str] = None

    @property
    def finger_states(self):
        return None if self.recognition is None else self.recognition["finger_states"]

    @property
    def prediction(self):
        states = self.finger_states
        return None if states is None else "".join(c for c, on in zip("TIMRP", states) if on)

    def to_dict(self):
        rec = self.recognition
        return {
            "status": "no_hand" if rec is None else "recognized",
            "finger_order": "TIMRP",
            "finger_states": self.finger_states,
            "prediction": self.prediction,
            "finger_text": None if rec is None else rec["finger_text"],
            "method": None if rec is None else rec["finger_match"]["method"],
            "fallback_reason": None if rec is None else rec["finger_match"].get("geometry_fallback_reason"),
            "failure_reason": self.failure_reason,
            "rotation_deg": self.rotation_deg,
            "processed_size": [int(self.image.shape[1]), int(self.image.shape[0])],
            "timings_ms": self.timings_ms,
        }


def recognize(image: np.ndarray, *, roi_enabled: bool = False) -> RecognitionResult:
    """Process one BGR uint8 image once; never modify the caller's input.

    ROI defaults to the current camera setup (off). roi_enabled=True reproduces
    the old phone-photo setup, as an explicit input parameter, not a file-name
    rule. Missing hand returns None states; a fist returns five False values.
    Timings exclude image decoding, JSON encoding and optional visualization.
    """
    if (not isinstance(image, np.ndarray) or image.dtype != np.uint8 or
            image.ndim != 3 or image.shape[2] != 3 or not image.shape[0] or not image.shape[1]):
        raise ValueError("Expected a nonempty H x W x 3 BGR uint8 image")
    start = perf_counter()
    height, width = image.shape[:2]
    scale = min(float(IMAGE_MAX_WIDTH) / width, float(IMAGE_MAX_HEIGHT) / height)
    if scale < 1.0:
        image = cv2.resize(image, (max(1, int(round(width * scale))),
                                  max(1, int(round(height * scale)))), interpolation=cv2.INTER_AREA)
    resized = perf_counter()
    image, rotation = auto_rotate_frame(image)
    oriented = perf_counter()
    frame = _Frame({"roi_enabled": bool(roi_enabled)})
    mask = step_color_threshold(image, frame)
    segmented = perf_counter()
    mask = step_mask_cleanup(mask, frame)
    cleaned = perf_counter()
    mask = step_palm_reconstruct(mask, frame)
    reconstructed = perf_counter()
    result = _recognize(mask, frame)
    finished = perf_counter()
    timings = {
        "resize": (resized - start) * 1000,
        "rotation": (oriented - resized) * 1000,
        "segmentation": (segmented - oriented) * 1000,
        "cleanup": (cleaned - segmented) * 1000,
        "reconstruction": (reconstructed - cleaned) * 1000,
        "recognition": (finished - reconstructed) * 1000,
        "total": (finished - start) * 1000,
    }
    return RecognitionResult(image, mask, result, float(rotation),
                             frame.flags.get("segment_stats", {}), timings,
                             frame.flags.get("failure_reason"))
