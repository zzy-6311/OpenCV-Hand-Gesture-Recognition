"""Fixed video-frame recognition and drawing.

The path is always resize -> board crop -> orientation -> skin segmentation
-> mask cleanup -> palm reconstruction -> finger identity -> original-frame
drawing. The current development algorithm and its thresholds are preserved.
Each call owns its data: no plugin registry, frame skipping, tracking, saved
decisions, training labels, or imports from the development directory.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from time import perf_counter
from types import SimpleNamespace
from typing import Optional
import os

import cv2
import numpy as np
import finger_geometry as FG

VIDEO_MAX_LONG_SIDE = 1280

ROI_WIDTH_VS_SHORT = 0.75

ROI_HEIGHT_RATIO = 0.8

ROI_CENTER_Y = 0.5

ROI_ENABLED = True

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

RECON_SEED_DT_FACTOR = 0.7

RECON_TOL_CR = 12

RECON_TOL_CB = 15

RECON_TOL_Y_DOWN = 75

RECON_MIN_SEED_PIXELS = 40

RECON_MIN_RADIUS_PX = 8

RECON_PALM_CLOSE_RATIO = 0.13

RECOG_WRIST_MIN_PIXELS = 12

RECOG_TRUNCATE_FACTOR = 1.5

RECOG_TRUNCATE_LINE_HALF = 1.6

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

FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")

FINGER_LABELS_CN = ("拇指", "食指", "中指", "无名指", "小指")

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

COLOR_BOUNDARY = (0, 200, 255)

COLOR_TEXT = (255, 255, 255)

COLOR_PALM = (0, 165, 255)

COLOR_WRIST = (0, 255, 255)

COLOR_TIP = (0, 0, 255)

COLOR_BASE = (255, 128, 0)

COLOR_GAP = (255, 0, 255)

COLOR_LABEL = (0, 255, 0)

COLOR_OUTLINE = (0, 0, 0)

HUD_MARGIN = 14

HUD_PANEL_ALPHA = 0.62

HUD_PANEL_BORDER = (95, 95, 95)

HUD_LINE_GAP = 8

CJK_FONT_CANDIDATES = (
    "C:\\Windows\\Fonts\\msyh.ttc",
    "C:\\Windows\\Fonts\\msyhbd.ttc",
    "C:\\Windows\\Fonts\\simhei.ttf",
    "C:\\Windows\\Fonts\\simsun.ttc",
    "C:\\Windows\\Fonts\\Deng.ttf",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
)

_CJK_FONT_CACHE: dict = {}

FINGER_BOX_SIZE = 54

FINGER_BOX_GAP = 12

FINGER_BOX_THICKNESS = 4

COLOR_FINGER_ON = (0, 220, 0)

COLOR_FINGER_OFF = (120, 120, 120)

FINGER_BOX_TAGS = ("T", "I", "M", "R", "P")

FINGER_DEBUG_RAY_LEN_R = 3.6

AUTO_ROTATE_DOWNSCALE = 320

AUTO_ROTATE_COLOR_SLACK = 0

AUTO_ROTATE_EDGE_MIN_PIXELS = 3

AUTO_ROTATE_EDGE_MAX_FRAC = 0.35

AUTO_ROTATE_MIN_AREA = 0.004


# Frame measurements and display helpers.


def _roi_size_pixels(shape_hw: tuple) -> tuple:
    """按【短边】定宽、按【画面高度】定高，返回 ``(宽, 高)``。

    ★ 宽度按短边算是这套自适应的关键：写死"画面宽度的百分之几"的话，
      竖幅画面（810x1440）只能拿到 405 px，手一偏就被切掉。
    """
    height, width = (int(shape_hw[0]), int(shape_hw[1]))
    short = min(height, width)
    roi_w = int(round(float(ROI_WIDTH_VS_SHORT) * short))
    roi_h = int(round(float(ROI_HEIGHT_RATIO) * height))
    roi_w = max(1, min(roi_w, width))
    roi_h = max(1, min(roi_h, height))
    return (roi_w, roi_h)


def roi_rect_pixels(shape_hw):
    """Centered fallback ROI; a detected board replaces this ROI entirely."""
    height, width = (int(shape_hw[0]), int(shape_hw[1]))
    roi_w, roi_h = _roi_size_pixels((height, width))
    cx, cy = (width * 0.5, height * float(ROI_CENTER_Y))
    x0 = max(0, min(int(round(cx - roi_w * 0.5)), width - roi_w))
    y0 = max(0, min(int(round(cy - roi_h * 0.5)), height - roi_h))
    return (x0, y0, min(width, x0 + roi_w), min(height, y0 + roi_h))


def roi_enabled_now(state):
    """Disable the centered fallback after successful board localization."""
    flags = state.flags
    if flags.get("calibrating_board", False):
        return False
    return not flags.get("preprocess", {}).get("board", {}).get("detected", False)


def build_roi_mask(shape_hw, enabled=True):
    """Build this frame's mask without any tracking or temporal cache."""
    height, width = (int(shape_hw[0]), int(shape_hw[1]))
    mask = np.zeros((height, width), dtype=np.uint8)
    if enabled:
        x0, y0, x1, y1 = roi_rect_pixels((height, width))
        mask[y0:y1, x0:x1] = 255
    else:
        mask[:] = 255
    return mask


def _normalized_pair(low, high, label: str) -> tuple:
    """把一对上下限规整好；写反了自动交换并提示，省得对着全黑画面找原因。"""
    low, high = (int(low), int(high))
    if low > high:
        low, high = (high, low)
    return (low, high)


def _finish_mask(mask: np.ndarray, state, method: str) -> np.ndarray:
    """Apply the fallback ROI and record this frame’s segmentation statistics."""
    height, width = mask.shape[:2]
    stats = {}
    enabled = roi_enabled_now(state)
    if enabled:
        cv2.bitwise_and(mask, build_roi_mask((height, width), enabled), dst=mask)
        x0, y0, x1, y1 = roi_rect_pixels((height, width))
        stats["roi_rect"] = (x0, y0, x1, y1)
        stats["roi_area_ratio"] = float((x1 - x0) * (y1 - y0)) / float(height * width)
    stats["roi_enabled"] = enabled
    stats["method"] = method
    stats["foreground_ratio"] = float(np.count_nonzero(mask)) / float(mask.size)
    state.flags["segment_stats"] = stats
    return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)


def _threshold_source(frame, state):
    return frame


def _estimate_scene_shift(ycrcb: np.ndarray, state) -> tuple:
    """Measure background chroma from this frame’s ROI; reuse its board measurement when available."""
    height, width = ycrcb.shape[:2]
    step = max(1, int(SCENE_ESTIMATE_STEP))
    enabled = roi_enabled_now(state)
    measured = state.flags.get("preprocess", {}).get("scene_calibration")
    if measured is not None and (not enabled):
        state.flags["scene_bg_estimate"] = dict(measured["background"])
        return tuple((float(v) for v in measured["shift"]))
    roi_rect = roi_rect_pixels((height, width)) if enabled else (0, 0, width, height)
    x0, y0, x1, y1 = roi_rect
    bg_cr = bg_cb = float("nan")
    tile_cr: list = []
    tile_cb: list = []
    mode_used = "roi"
    if mode_used == "roi":
        roi = build_roi_mask((height, width), enabled) > 0
        sub_roi = roi[::step, ::step]
        sub_roi &= ycrcb[::step, ::step, 0] > 0
        if not np.any(sub_roi):
            state.flags.pop("scene_bg_estimate", None)
            return (0.0, 0.0)
        bg_cr = float(np.median(ycrcb[::step, ::step, 1][sub_roi].astype(np.float32)))
        bg_cb = float(np.median(ycrcb[::step, ::step, 2][sub_roi].astype(np.float32)))
    d_cr = bg_cr - float(SCENE_BASELINE_CR)
    d_cb = bg_cb - float(SCENE_BASELINE_CB)
    limit = float(SCENE_SHIFT_CLAMP)
    d_cr = max(-limit, min(limit, d_cr))
    d_cb = max(-limit, min(limit, d_cb))
    spread_cr = float(np.max(tile_cr) - np.min(tile_cr)) if len(tile_cr) >= 2 else float("nan")
    spread_cb = float(np.max(tile_cb) - np.min(tile_cb)) if len(tile_cb) >= 2 else float("nan")
    state.flags["scene_bg_estimate"] = {
        "mode": mode_used,
        "tiles": len(tile_cr),
        "tile_cr": [round(v, 1) for v in tile_cr],
        "tile_cb": [round(v, 1) for v in tile_cb],
        "bg_cr": round(bg_cr, 1),
        "bg_cb": round(bg_cb, 1),
        "spread_cr": round(spread_cr, 1) if spread_cr == spread_cr else None,
        "spread_cb": round(spread_cb, 1) if spread_cb == spread_cb else None,
    }
    return (d_cr, d_cb)


def _skin_color_bounds(ycrcb: np.ndarray, state, *, calibrate=True) -> tuple:
    """Shared scene calibration and Y/Cr/Cb bounds for rotation and segmentation."""
    cr_lo, cr_hi = _normalized_pair(CR_LOW, CR_HIGH, "CR_LOW/CR_HIGH")
    cb_lo, cb_hi = _normalized_pair(CB_LOW, CB_HIGH, "CB_LOW/CB_HIGH")
    y_lo, y_hi = _normalized_pair(Y_LOW, Y_HIGH, "Y_LOW/Y_HIGH")
    scene_dcr = scene_dcb = 0.0
    if calibrate:
        scene_dcr, scene_dcb = _estimate_scene_shift(ycrcb, state)
        cr_lo = int(round(max(0.0, min(255.0, cr_lo + scene_dcr))))
        cr_hi = int(round(max(0.0, min(255.0, cr_hi + scene_dcr))))
        cb_lo = int(round(max(0.0, min(255.0, cb_lo + scene_dcb))))
        cb_hi = int(round(max(0.0, min(255.0, cb_hi + scene_dcb))))
        cr_lo, cr_hi = _normalized_pair(cr_lo, cr_hi, "校准后 Cr")
        cb_lo, cb_hi = _normalized_pair(cb_lo, cb_hi, "校准后 Cb")
    else:
        state.flags.pop("scene_bg_estimate", None)
    state.flags["scene_shift"] = (scene_dcr, scene_dcb)
    state.flags["scene_box"] = (cr_lo, cr_hi, cb_lo, cb_hi)
    return ((y_lo, cr_lo, cb_lo), (y_hi, cr_hi, cb_hi))


def _board_valid_for_threshold(shape_hw, state):
    """Board validity is already in this frame's prepared coordinates."""
    info = state.flags.get("preprocess", {})
    if not info.get("board", {}).get("detected", False):
        return None
    valid = info.get("board_valid_mask")
    if isinstance(valid, np.ndarray) and valid.ndim == 2 and (valid.shape == tuple(shape_hw)):
        return valid
    return None


def segment_skin(frame: np.ndarray, state) -> np.ndarray:
    """Segment skin with scene-calibrated Y/Cr/Cb bounds and measured board validity."""
    ycrcb = cv2.cvtColor(_threshold_source(frame, state), cv2.COLOR_BGR2YCrCb)
    y_channel = ycrcb[:, :, 0]
    cr_channel = ycrcb[:, :, 1]
    cb_channel = ycrcb[:, :, 2]
    lower, upper = _skin_color_bounds(ycrcb, state)
    y_lo, cr_lo, cb_lo = lower
    y_hi, cr_hi, cb_hi = upper
    mask = cv2.inRange(cr_channel, cr_lo, cr_hi)
    cv2.bitwise_and(mask, cv2.inRange(cb_channel, cb_lo, cb_hi), dst=mask)
    luminance = cv2.inRange(y_channel, y_lo, y_hi)
    board_valid = _board_valid_for_threshold(y_channel.shape, state)
    if board_valid is not None:
        observed = cv2.inRange(y_channel, 1, y_hi)
        cv2.bitwise_and(observed, board_valid, dst=observed)
        cv2.bitwise_or(luminance, observed, dst=luminance)
    cv2.bitwise_and(mask, luminance, dst=mask)
    state.flags["segment_ycrcb"] = ycrcb
    state.flags["segment_source_bgr"] = _threshold_source(frame, state)
    result = _finish_mask(mask, state, "color")
    state.flags["segment_stats"]["board_low_light_recovery"] = board_valid is not None
    return result


def _fill_internal_holes(binary: np.ndarray) -> np.ndarray:
    """填掉前景**内部**的孔洞，完全不动外轮廓。

    做法：往外补一圈背景，再从角上 floodFill 把「外部背景」涂成 255。
    补的那一圈保证「所有和图像边界相连的背景」都是连通的，
    于是 floodFill 之后仍然是 0 的像素，就是被前景完全包住的内部空洞。

    ⚠️ 为什么要补一圈：如果只在原图上从 (0,0) floodFill，而前景恰好把
    画面上下切断（手贴到 ROI 左右边时就会这样），下半部分的背景就到不了，
    会被误判成"空洞"整片填白。补一圈就彻底避免了这个坑。
    """
    height, width = binary.shape[:2]
    padded = cv2.copyMakeBorder(binary, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    flood_mask = np.zeros((height + 4, width + 4), np.uint8)
    cv2.floodFill(padded, flood_mask, (0, 0), 255)
    holes = cv2.bitwise_not(padded)[1:-1, 1:-1]
    return cv2.bitwise_or(binary, holes)


def roi_area_pixels(shape_hw: tuple, enabled: bool = None) -> int:
    """ROI 的像素面积（用来算绝对面积门槛）。ROI 关掉时就是整幅图。"""
    if enabled is None:
        enabled = ROI_ENABLED
    height, width = (int(shape_hw[0]), int(shape_hw[1]))
    if not enabled:
        return height * width
    x0, y0, x1, y1 = roi_rect_pixels((height, width))
    return int((x1 - x0) * (y1 - y0))


def clean_mask(frame: np.ndarray, state) -> np.ndarray:
    """Remove small components and fill holes, preserving the development thresholds."""
    single = frame[:, :, 0] if frame.ndim == 3 else frame
    mask = single.copy()
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
    unsupported_board_components = 0
    board_valid = _board_valid_for_threshold((height, width), state)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if count > 1:
        eligible = np.ones(count, dtype=bool)
        eligible[0] = False
        if board_valid is not None:
            # Slanted board corners and replicated rotation borders can contain
            # room background. Components crossing the measured board edge
            # need a palm-sized core inside it before hole filling. Keep a
            # supported hand whole so its forearm and wrist axes do not change.
            inside = cv2.bitwise_and(mask, board_valid)
            padded = cv2.copyMakeBorder(
                inside, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0
            )
            distance = cv2.distanceTransform(padded, cv2.DIST_L2, 5)[1:-1, 1:-1]
            outside_labels = np.unique(labels[board_valid == 0])
            core_labels = np.unique(labels[distance >= float(RECON_MIN_RADIUS_PX)])
            eligible[outside_labels] = False
            eligible[core_labels] = True
            eligible[0] = False
            unsupported_board_components = int(np.count_nonzero(~eligible[1:]))
        # Rejected outside blobs must not suppress a smaller supported hand
        # by raising the relative area threshold.
        areas = stats[eligible, cv2.CC_STAT_AREA].astype(np.float64)
        largest = float(areas.max()) if areas.size else 0.0
        absolute_min = float(CLEANUP_MIN_AREA_RATIO) * float(
            roi_area_pixels((height, width), enabled)
        )
        relative_min = float(CLEANUP_MIN_AREA_VS_LARGEST) * largest
        min_area_used = max(1.0, absolute_min, relative_min)
        lut = np.zeros(count, np.uint8)
        for label in range(1, int(count)):
            if eligible[label] and float(stats[label, cv2.CC_STAT_AREA]) >= min_area_used:
                lut[label] = 255
                kept_components += 1
            else:
                removed_components += 1
        mask = lut[labels] if kept_components else np.zeros_like(mask)
    filled_pixels = 0
    if kept_components:
        before = int(np.count_nonzero(mask))
        mask = _fill_internal_holes(mask)
        filled_pixels = int(np.count_nonzero(mask)) - before
    segment_stats = dict(state.flags.get("segment_stats") or {})
    segment_stats["cleaned"] = True
    segment_stats["board_component_filter"] = board_valid is not None
    segment_stats["unsupported_board_components"] = unsupported_board_components
    segment_stats["foreground_ratio"] = float(np.count_nonzero(mask)) / float(mask.size)
    segment_stats["removed_components"] = removed_components
    segment_stats["kept_components"] = kept_components
    segment_stats["min_area_used"] = min_area_used
    segment_stats["filled_pixels"] = filled_pixels
    state.flags["segment_stats"] = segment_stats
    return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)


def _reject_board_shadow_branches(mask, ycrcb, board_valid, background_crcb,
                                  palm_seed, palm_radius):
    """Remove thinly attached board shadows using this frame's two chroma centers.

    Opening supplies connectivity evidence; it never replaces the hand mask.
    Palm-supported components and skin-colored thin fingers retain their
    original outline. Only board-colored detached branches are removed.
    """
    diagnostic = dict(applied=False, rejected_branches=0, removed_pixels=0)
    if (ycrcb is None or board_valid is None or background_crcb is None or
            palm_radius <= 0 or not np.any(palm_seed)):
        return mask, diagnostic
    foreground = np.median(ycrcb[:, :, 1:][palm_seed].astype(np.float32), axis=0)
    background = np.asarray(background_crcb, np.float32)
    direction = foreground - background
    separation = float(direction @ direction)
    if not np.isfinite(separation) or separation <= 0:
        return mask, diagnostic
    # The existing minimum flesh support defines a scale-relative thin neck.
    size = max(3, int(round(palm_radius * RECOG_TIP_MIN_THICKNESS)) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(opened, 8)
    diagnostic.update(applied=True, kernel=size, components=count - 1)
    if count <= 2:
        return mask, diagnostic
    protected = np.zeros(count, bool)
    protected[np.unique(labels[palm_seed])] = True
    protected[0] = False
    outside_votes = np.bincount(labels[board_valid == 0], minlength=count)
    possible = (~protected) & (outside_votes == 0)
    possible[0] = False
    if not np.any(possible):
        return mask, diagnostic
    # Only detached, board-contained components need a color vote. Avoid
    # converting/scoring every background and palm pixel of the full frame.
    selected = possible[labels]
    selected_labels = labels[selected]
    score = (ycrcb[:, :, 1:][selected].astype(np.float32) - background) @ direction
    foreground_votes = np.bincount(
        selected_labels[score >= 0.5 * separation], minlength=count)
    rejected = possible & (foreground_votes * 2 < stats[:, cv2.CC_STAT_AREA])
    diagnostic["rejected_branches"] = int(rejected.sum())
    if not np.any(rejected):
        return mask, diagnostic
    remove = cv2.dilate((rejected.astype(np.uint8) * 255)[labels], kernel)
    remove[protected[labels]] = 0
    erased = (remove > 0) & (mask > 0)
    diagnostic["removed_pixels"] = int(erased.sum())
    result = mask.copy()
    result[erased] = 0
    return result, diagnostic


def _estimate_palm_robust(mask: np.ndarray, valid_mask=None):
    """估掌心 (C, R)：先对**一份副本**做闭运算，再取最大内切圆。

    为什么这么做（完整理由见配置区 RECON_PALM_CLOSE_RATIO 的注释）：
    掩膜可能被割出口子，直接用会让最大内切圆只能塞进半块掌、R 被严重低估。
    闭运算把口子合上，R 就准了。

    ★ **不会修改传入的 mask** —— 闭运算只在内部副本上做，
      所以调用方手里的掩膜（拿去找指尖的）仍然是"指缝锋利"的原样。

    ★ **核大小按掩膜面积算，不按 R 算**：R 正是可能被低估的量，用它定核会
      陷入"核不够大 -> 口子合不上 -> R 还是小的"鸡生蛋。面积则几乎不受割口影响。

    返回 ``((cx, cy), R)``。
    """
    area = int(np.count_nonzero(mask))
    if area <= 0:
        return (0, 0), 0.0
    scale = float(np.sqrt(float(area) / np.pi))       # 等效半径，抗割口
    size = int(round(float(RECON_PALM_CLOSE_RATIO) * scale))
    if size >= 3:
        if size % 2 == 0:
            size += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        work = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    else:
        work = mask
    distance = cv2.distanceTransform(work, cv2.DIST_L2, 5)
    # Keep the complete silhouette for wrist estimation, but a board-backed
    # palm cannot be centered on replicated forearm/background outside it.
    _, radius, _, location = cv2.minMaxLoc(distance, mask=valid_mask)
    return (int(location[0]), int(location[1])), float(radius)


def reconstruct_palm(frame: np.ndarray, state) -> np.ndarray:
    """Reconstruct components connected to the palm seed under local colour constraints."""
    single = frame[:, :, 0] if frame.ndim == 3 else frame
    mask = single.copy()
    height, width = mask.shape[:2]
    ycrcb = state.flags.get("segment_ycrcb")
    if ycrcb is None:
        return frame
    board_valid = _board_valid_for_threshold(mask.shape, state)
    distance = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    _, radius, _, location = cv2.minMaxLoc(distance, mask=board_valid)
    if radius < float(RECON_MIN_RADIUS_PX):
        return frame
    cx, cy = (int(location[0]), int(location[1]))
    seed = distance >= float(RECON_SEED_DT_FACTOR) * radius
    if board_valid is not None:
        seed &= board_valid > 0
    seed_pixels = int(np.count_nonzero(seed))
    if seed_pixels < int(RECON_MIN_SEED_PIXELS):
        return frame

    def median_of(plane):
        return float(np.median(plane[seed]))

    cr_plane = ycrcb[:, :, 1]
    cb_plane = ycrcb[:, :, 2]
    y_plane = ycrcb[:, :, 0]
    cr0 = median_of(cr_plane)
    cb0 = median_of(cb_plane)
    y0 = median_of(y_plane)
    background = state.flags.get("scene_bg_estimate")
    background_crcb = ((background["bg_cr"], background["bg_cb"])
                       if background else None)
    mask, shadow_filter = _reject_board_shadow_branches(
        mask, ycrcb, board_valid, background_crcb, seed, radius)
    cr_lo = int(round(max(0.0, cr0 - float(RECON_TOL_CR))))
    cr_hi = int(round(min(255.0, cr0 + float(RECON_TOL_CR))))
    cb_lo = int(round(max(0.0, cb0 - float(RECON_TOL_CB))))
    cb_hi = int(round(min(255.0, cb0 + float(RECON_TOL_CB))))
    y_lo = int(round(max(0.0, y0 - float(RECON_TOL_Y_DOWN))))
    allowed = cv2.inRange(cr_plane, cr_lo, cr_hi)
    cv2.bitwise_and(allowed, cv2.inRange(cb_plane, cb_lo, cb_hi), dst=allowed)
    reconstruction_y = cv2.inRange(y_plane, y_lo, 255)
    if board_valid is not None:
        cv2.bitwise_or(reconstruction_y, board_valid, dst=reconstruction_y)
    cv2.bitwise_and(allowed, reconstruction_y, dst=allowed)
    cv2.bitwise_and(allowed, mask, dst=allowed)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(allowed, 8)
    seed_labels = set((int(v) for v in np.unique(labels[seed]))) - {0}
    if not seed_labels:
        return frame
    lookup = np.zeros(count, np.uint8)
    for label in seed_labels:
        lookup[label] = 255
    result = lookup[labels]
    result = _fill_internal_holes(result)
    (palm_cx, palm_cy), palm_radius = _estimate_palm_robust(result, board_valid)
    segment_stats = dict(state.flags.get("segment_stats") or {})
    segment_stats["reconstructed"] = True
    segment_stats["palm_center"] = (palm_cx, palm_cy)
    segment_stats["palm_radius"] = float(palm_radius)
    segment_stats["palm_radius_raw"] = float(radius)
    segment_stats["seed_pixels"] = seed_pixels
    segment_stats["color_base"] = (cr0, cb0, y0)
    segment_stats["shadow_filter"] = shadow_filter
    segment_stats["foreground_ratio"] = float(np.count_nonzero(result)) / float(result.size)
    state.flags["segment_stats"] = segment_stats
    return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)


def _to_local_points(points, origin, up, scale) -> np.ndarray:
    """把点转到手部局部坐标系并按 R 归一化。

    x 为手掌左右方向，**y 为指向手腕为正**（所以指尖的 y 为负）。
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if pts.size == 0:
        return np.zeros((0, 2), dtype=np.float64)
    right = np.array([-float(up[1]), float(up[0])], dtype=np.float64)
    delta = pts - np.asarray(origin, dtype=np.float64).reshape(1, 2)
    safe = float(scale) if abs(float(scale)) > 1e-06 else 1.0
    return np.stack([delta @ right / safe, -(delta @ up) / safe], axis=1)


def _contour_point(contour, index) -> np.ndarray:
    """按环形索引取轮廓点（float64）。"""
    count = int(contour.shape[0])
    return contour[int(index) % count].reshape(2).astype(np.float64)


def _turning_ratio(contour, index: int, window: int) -> float:
    """轮廓在该点的「转折比」= 弯曲深度 / 弦长。

    直线段接近 0，指尖圆帽明显更大。**不依赖像素尺度**，所以不用绝对阈值。
    """
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
    """轮廓在该点的夹角（度）。越接近 180 越平，越小越尖。"""
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
    """多尺度地量"这里尖不尖"，返回（转折比, 夹角）。"""
    windows = [max(2, int(round(window * factor))) for factor in (0.6, 1.0, 1.5)]
    ratios = [_turning_ratio(contour, index, w) for w in windows]
    angles = [_contour_angle(contour, index, w) for w in windows]
    return (float(np.max(ratios)), float(np.median(angles)))


def _circular_window_extrema(values, half_window):
    """Exact min/max of the established circular contour-index window.

    The existing peak thresholds are defined against a fixed number of
    CHAIN_APPROX_NONE points. Keep that support unchanged; only replace the
    repeated offset scans with monotone queues, using O(N + window) work.
    """
    half_window = int(half_window)
    if half_window < 0:
        raise ValueError("The contour window must be nonnegative")
    minima, maxima = (np.empty_like(values), np.empty_like(values))
    if len(values) == 0:
        return (minima, maxima)
    repeated = np.pad(values, (half_window, half_window), mode="wrap")
    low, high = (deque(), deque())
    right = 0
    for index in range(len(values)):
        while right <= index + 2 * half_window:
            while low and repeated[low[-1]] >= repeated[right]:
                low.pop()
            while high and repeated[high[-1]] <= repeated[right]:
                high.pop()
            low.append(right)
            high.append(right)
            right += 1
        while low and low[0] < index:
            low.popleft()
        while high and high[0] < index:
            high.popleft()
        minima[index], maxima[index] = (repeated[low[0]], repeated[high[0]])
    return (minima, maxima)


def _sample_plane(plane: np.ndarray, point: np.ndarray) -> float:
    """取某个浮点坐标处的像素值（就近取整，越界夹回）。"""
    height, width = plane.shape[:2]
    x = int(np.clip(round(float(point[0])), 0, width - 1))
    y = int(np.clip(round(float(point[1])), 0, height - 1))
    return float(plane[y, x])


def _protrusion_length(thickness: np.ndarray, tip_point, palm_center, radius: float) -> float:
    """量"这根东西从指尖凸出来了多长"，按 R 归一化。

    做法：从指尖沿直线朝掌心走，一路采样内部距离变换的厚度值，
    找到**第一个"厚度 >= RECOG_TIP_PALM_THICK * R"** 的位置 —— 那里就算
    已经进入掌区了。指尖到那里的距离就是凸出长度。

    为什么需要它：真手指（哪怕小指这种很短的）都是一条**细长凸出**，
    而拳头的指节凸起是**圆钝鼓包** —— 一出指尖厚度就上去了。
    这个量是尺度无关的（按 R 归一化），而且语义正好就是"手指伸出来多少"。
    """
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
    """手腕 W = 手掌**下方**横截面最窄的那一行。

    ★★ 这里**故意写死"小臂从画面下方垂直入画"** ★★
      因为整条流水线（截前臂、掌轴、所有角度判据）都是围绕这个前提标定的，
      而它在"手指朝上"的图上经过验证（老三批 28 张就是靠它）✓

      手转到别的方向怎么办？**不在这里做** —— 由前处理
      ``detect_hand_rotation()`` 先把画面**转成"手指朝上"**，再进这条流水线 ✓
      （试过让这里"方向无关"，实测反而更差：手大、小臂被画面切掉时
        "找走得远又细的方向"会误选到手指 ✗）

    ⚠️ 早期那版（已删除的 solution.py）用的是"手部底端带状区域的质心"，但那个做法
    默认掩膜在手腕处就结束了 —— 而我们的掩膜里含**整条小臂**，所以底端质心
    取到的是小臂尖，实测跑到 (1522,1265)，离真正的手腕差了 400 多像素。

    改成找**宽度剖面的局部极小**：手掌往下、宽度会先收到最窄（手腕束腰），
    再进小臂又变宽。取那个"离掌心最近的、明显比最宽处窄的"局部极小即可。

    ``mask`` 必须是**含小臂**的（截断之前），否则找不到束腰。
    """
    return _estimate_wrist_legacy(mask, palm_center, radius)


def _estimate_wrist_legacy(mask: np.ndarray, palm_center, radius: float):
    """手腕 W = 手掌下方**横截面最窄**的那一行（老做法，**只适合小臂垂直入画**）。

    ⚠️ 保留它只作为兜底：当画面里找不到"细长的小臂"（比如手伸到画面外、
    或者掩膜没含小臂）时用它。**不要再把它当主路径** —— 它写死了小臂方向 ✗

    原始注释（仍然有效）：早期那版用的是"手部底端带状区域的质心"，
    但那个做法默认掩膜在手腕处就结束了 —— 而我们的掩膜里含**整条小臂**，
    所以底端质心取到的是小臂尖。改成找**宽度剖面的局部极小**。
    """
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
    segment = row_widths[start : bottom_row + 1].astype(np.float32)
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
    center = np.array(
        [0.5 * (float(xs.min()) + float(xs.max())), float(wrist_row)], dtype=np.float64
    )
    if center[1] <= float(palm_center[1]):
        return None
    return center


def _truncate_forearm(mask: np.ndarray, palm_center, up, radius: float) -> np.ndarray:
    """前臂截断：只保留"掌心 + 手指那一侧"的**半平面**。

    ⚠️ 这里踩过一个严重的坑，务必保留这个写法：
       最初把多边形边距写成 `4R`（约 594 px）。结果多边形在**手指方向**也只
       延伸了 594 px，而指尖离截断线有 601 px —— **整片指尖被削平了**。
       表现在掩膜上就是出现一条水平直边，那条直边上的毛刺被当成了 4 个"指尖"，
       于是指尖数从 5 变成 6，而且指根/长度全乱。
       判据：手指区每行游程数在 y=330~390 全是 0，y=400 才突然冒出 3~5 个。

       所以沿**两个方向**都要延伸到远大于图像尺寸 —— 等价于一个半平面，
       这样无论手举多高、指尖离掌心多远都不会被误切。
    """
    height, width = mask.shape[:2]
    far = float(max(height, width)) * 2.0
    origin = np.asarray(palm_center, dtype=np.float64) - np.asarray(up, dtype=np.float64) * (
        float(RECOG_TRUNCATE_FACTOR) * float(radius)
    )
    along = np.array([-float(up[1]), float(up[0])], dtype=np.float64)
    polygon = np.array(
        [
            origin + along * far,
            origin - along * far,
            origin - along * far + np.asarray(up) * far,
            origin + along * far + np.asarray(up) * far,
        ],
        dtype=np.int32,
    )
    allowed = np.zeros((height, width), dtype=np.uint8)
    cv2.fillConvexPoly(allowed, polygon, 255)
    return cv2.bitwise_and(mask, allowed)


def _detect_fingertips(contour, mask: np.ndarray, palm_center, radius: float, up,
                       *, closed_candidates=None):
    """在轮廓上找指尖。返回（候选列表, 距离剖面）。

    判据（全部按 R 归一化）：
      1. 到掌心的距离 >= 1.25R
      2. 是该距离剖面的**局部极大**（环形邻域窗口）
      3. **突出度**足够（极大值要比邻域低谷高出 >= 0.2R）
      4. 形态要"尖"：转折比够大，或夹角够小
      5. 内部厚度不能太厚（排除掌缘/前臂那种厚凸起）
      6. 必须落在**掌上方**（局部系里 y <= 0.15）
    """
    count = int(contour.shape[0])
    if count < 8:
        return [], None
    points = contour.reshape(count, 2).astype(np.float64)
    distances = np.linalg.norm(points - np.asarray(palm_center).reshape(1, 2), axis=1)

    window = max(2, int(round(RECOG_TIP_PEAK_WINDOW * float(radius))))
    floor_min, floor_max = _circular_window_extrema(distances, window)

    # 先用纯向量化的三条门槛筛掉绝大多数轮廓点（快）
    keep = distances >= RECOG_TIP_MIN_DISTANCE * float(radius)
    keep &= floor_max <= distances + 1e-9
    keep &= (distances - floor_min) >= RECOG_TIP_MIN_PROMINENCE * float(radius)
    local_points = _to_local_points(points, palm_center, up, radius)
    # Reject wrist-cut corners before ranking/capping the candidates, so
    # invalid distant points cannot consume the slots of real fingertips.
    keep &= local_points[:, 1] <= float(RECOG_TIP_PALM_SIDE_MAX)
    candidate_indices = np.nonzero(keep)[0]
    if candidate_indices.size == 0:
        return [], distances

    # 剩下少数候选才算形态和厚度（贵的那部分）
    thickness = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    candidates = []
    for raw_index in candidate_indices:
        index = int(raw_index)
        ratio, angle = _tip_sharpness(contour, index, window)
        inner = points[index] + (np.asarray(palm_center) - points[index]) * \
            float(RECOG_TIP_THICKNESS_SAMPLE)
        thickness_ratio = _sample_plane(thickness, inner) / float(radius)
        if ratio < RECOG_TIP_MIN_TURN_RATIO and \
                thickness_ratio > RECOG_TIP_THICKNESS_MAX:
            continue
        if angle > RECOG_TIP_MAX_ANGLE_DEG:
            continue
        # 针状毛刺：夹角和厚度同时极端小 -> 不是指尖（真指尖是圆帽，两者都不会这么极端）
        if angle < RECOG_TIP_MIN_ANGLE_DEG and \
                thickness_ratio < RECOG_TIP_MIN_THICKNESS:
            # The radial sample can cross a finger gap. Rescue only a long
            # local shaft with opposing sides that do not markedly taper.
            shaft = FG._measure_tip(contour, points[index],
                                    np.asarray(palm_center, dtype=float),
                                    np.asarray(up, dtype=float), radius, FG.DEFAULT_CONFIG)
            if not (shaft["width_r"] >= 2.0 * RECOG_TIP_MIN_THICKNESS and
                    shaft["length_r"] >= FG.DEFAULT_CONFIG.axis_end_r and
                    shaft["aspect"] >= FG.DEFAULT_CONFIG.curled_aspect_max and
                    shaft["shaft_width_ratio"] <=
                    FG.DEFAULT_CONFIG.base_width_factor):
                continue
        # ★主力判据：凸出长度。真手指是细长凸出，指节凸起是圆钝鼓包。
        #   这一条同时管住了两头 —— 既排掉拳头的指节凸起，又保住很短的小指。
        protrusion = _protrusion_length(thickness, points[index], palm_center, radius)
        closed = protrusion < float(RECOG_TIP_PROTRUSION_MIN)
        if closed and closed_candidates is None:
            continue
        candidates.append({
            "closed_support": closed,
            "point": points[index].copy(),
            "local": (float(local_points[index, 0]), float(local_points[index, 1])),
            "distance": float(distances[index]),
            "turn_ratio": float(ratio),
            "angle_deg": float(angle),
            "thickness_ratio": float(thickness_ratio),
            "protrusion_ratio": float(protrusion),
        })

    # Keep the original strong-tip ranking independent of closed caps. A
    # knuckle cannot displace a real tip or change the extended-finger count.
    def merge(items):
        items = sorted(items, key=lambda item: -item["distance"])
        merged = []
        for item in items:
            if any(float(np.linalg.norm(item["point"] - kept["point"]))
                   < RECOG_TIP_DEDUP * float(radius) for kept in merged):
                continue
            merged.append(item)
            if len(merged) >= int(RECOG_TIP_MAX_COUNT):
                break
        merged.sort(key=lambda item: float(np.arctan2(item["local"][1], item["local"][0])))
        return merged

    strong = merge([item for item in candidates if not item["closed_support"]])
    if closed_candidates is not None:
        caps = [item for item in merge(candidates) if item["closed_support"]]
        promoted, remaining = [], []
        for cap in caps:
            # The mixed candidate merge can prefer a farther closed cap over
            # a nearby strong tip. Keep the strong observation and avoid
            # counting that same local contour cap again after promotion.
            if any(float(np.linalg.norm(cap["point"] - tip["point"])) <
                   RECOG_TIP_DEDUP * float(radius) for tip in strong):
                continue
            feature = FG._measure_tip(contour, np.asarray(cap["point"], dtype=float),
                                      np.asarray(palm_center, dtype=float),
                                      np.asarray(up, dtype=float), radius, FG.DEFAULT_CONFIG)
            # Radial reach can underestimate a spread little finger. Require
            # a real narrow shaft spanning the full existing axis window;
            # aspect alone would also accept a tiny, needle-like mask burr.
            if (feature["extended"] and feature["has_branch_extent"] and
                    feature["length_r"] >= FG.DEFAULT_CONFIG.axis_end_r and
                    (feature["aspect"] >= FG.DEFAULT_CONFIG.curled_aspect_max or
                     FG._is_lateral_thumb_branch(
                         feature, np.asarray(palm_center, dtype=float),
                         np.asarray(up, dtype=float), radius, FG.DEFAULT_CONFIG))):
                promoted.append(dict(cap, closed_support=False,
                                     promotion="contour_shaft_extent"))
            else:
                remaining.append(cap)
        if len(strong) + len(promoted) <= 5:
            strong.extend(promoted)
            strong.sort(key=lambda item: float(np.arctan2(item["local"][1], item["local"][0])))
            closed_candidates.extend(remaining)
        else:
            # Conflicting evidence cannot introduce more anatomical fingers.
            closed_candidates.extend(caps)
    return strong, distances


def _detect_gaps(contour, tips, palm_center, radius: float, up):
    """在每一对**相邻指尖**之间找最深的凹谷，就是指缝。

    找法：取两个指尖之间那段轮廓，找其中**离掌心最近**的那个点 ——
    它天然落在两指之间的谷底。深度用"该点到两指尖连线的垂距"来算，
    含义和凸缺陷深度一致。
    """
    if len(tips) < 2 or contour is None:
        return []
    count = int(contour.shape[0])
    points = contour.reshape(count, 2).astype(np.float64)
    palm_distances = np.linalg.norm(points - np.asarray(palm_center).reshape(1, 2), axis=1)
    gaps = []
    for first, second in zip(tips, tips[1:]):
        first_point = np.asarray(first["point"], dtype=np.float64)
        second_point = np.asarray(second["point"], dtype=np.float64)
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
        gaps.append(
            {
                "point": valley.copy(),
                "local": (float(local[0]), float(local[1])),
                "depth_ratio": float(depth) / float(radius),
                "angle_deg": float(angle),
                "pair": (
                    tuple(np.round(first_point).astype(int)),
                    tuple(np.round(second_point).astype(int)),
                ),
            }
        )
    gaps.sort(key=lambda item: float(np.arctan2(item["local"][1], item["local"][0])))
    return gaps[: int(RECOG_GAP_MAX_COUNT)]


def _build_fingers(tips, gaps, palm_center, radius: float, up):
    """给每根手指算「指根」：取它两侧指缝的中点。

    - 两侧都有指缝 -> 取中点
    - 只有一侧     -> 就用那一侧
    - 一侧都没有   -> 回退成"掌心沿该指尖方向 1.0R 处"
    指根到指尖连成一条线，就代表这根手指的**位置、方向、长度**。
    """
    fingers = []
    if not tips:
        return fingers
    for position, tip in enumerate(tips):
        tip_local = np.asarray(tip["local"], dtype=np.float64)
        left = gaps[position - 1]["point"] if 0 <= position - 1 < len(gaps) else None
        right = gaps[position]["point"] if 0 <= position < len(gaps) else None
        if left is not None and right is not None:
            base = (np.asarray(left, dtype=np.float64) + np.asarray(right, dtype=np.float64)) / 2.0
        elif left is not None:
            base = np.asarray(left, dtype=np.float64)
        elif right is not None:
            base = np.asarray(right, dtype=np.float64)
        else:
            direction = np.asarray(tip["point"], dtype=np.float64) - np.asarray(palm_center)
            norm = float(np.linalg.norm(direction))
            direction = direction / norm if norm > 1e-06 else np.asarray(up, dtype=np.float64)
            base = np.asarray(palm_center, dtype=np.float64) + direction * float(radius)
        length = float(np.linalg.norm(np.asarray(tip["point"], dtype=np.float64) - base))
        fingers.append(
            {
                "index": position + 1,
                "tip": np.asarray(tip["point"], dtype=np.float64).copy(),
                "base": base.copy(),
                "tip_local": (float(tip_local[0]), float(tip_local[1])),
                "length_ratio": length / float(radius),
                "angle_deg": float(np.degrees(np.arctan2(tip_local[1], tip_local[0]))),
            }
        )
    return fingers


def _recognize(mask: np.ndarray, state) -> Optional[dict]:
    """Measure the wrist, palm, contour, finger candidates and five anatomical finger states."""
    if mask is None or int(np.count_nonzero(mask)) == 0:
        return None
    segment_stats = state.flags.get("segment_stats") or {}
    palm_center = segment_stats.get("palm_center")
    radius = segment_stats.get("palm_radius")
    if palm_center is None or not radius:
        palm_center, radius = _estimate_palm_robust(
            mask, _board_valid_for_threshold(mask.shape, state))
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
    closed_caps = []
    tips, _profile = _detect_fingertips(
        contour, hand, palm_center, radius, up, closed_candidates=closed_caps
    )
    gaps = _detect_gaps(contour, tips, palm_center, radius, up)
    fingers = _build_fingers(tips, gaps, palm_center, radius, up)
    hull = cv2.convexHull(contour)
    contour_area = float(cv2.contourArea(contour))
    hull_area = float(cv2.contourArea(hull)) if hull is not None else 0.0
    solidity = float(np.clip(contour_area / hull_area, 0.0, 1.0)) if hull_area > 0 else 0.0
    angle_span_deg = 0.0
    if len(tips) >= 2:
        tip_angles = np.array(
            [float(np.arctan2(t["local"][1], t["local"][0])) for t in tips], dtype=np.float64
        )
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
    mean_length = float(np.mean([f["length_ratio"] for f in fingers])) if fingers else 0.0
    result = {
        "palm_center": palm_center,
        "palm_radius": radius,
        "wrist": wrist,
        "up": up,
        "hand_mask": hand,
        "contour": contour,
        "hull": hull,
        "solidity": solidity,
        "angle_span_deg": angle_span_deg,
        "palm_rel": palm_rel,
        "mean_length_ratio": mean_length,
        "tips": tips,
        "gaps": gaps,
        "fingers": fingers,
        "finger_count": len(tips),
    }
    states, extensions, angles_used, match_detail, expect_len = _finger_states(
        hand, palm_center, up, radius, tips, gaps, contour=contour, closed_candidates=closed_caps
    )
    result["finger_states"] = states
    result["finger_extensions"] = extensions
    result["finger_ray_angles"] = angles_used
    result["finger_expect_len"] = expect_len
    result["finger_match"] = match_detail
    result["finger_names"] = list(FINGER_NAMES)
    result["finger_text"] = finger_states_text(states)
    return result


def _radial_profile(mask: np.ndarray, palm_center, up, radius: float):
    """算**径向剖面**：每个角度上，掩膜从掌心往外连续延伸多长（× R）。

    返回 ``(angles_deg, profile)``，两个等长的一维数组。
    角度以掌轴为 0，**正方向 = 拇指侧**（右手 + 掌心朝镜头）。
    """
    up = np.asarray(up, dtype=np.float64).reshape(2)
    norm = float(np.linalg.norm(up))
    if norm > 1e-09:
        up = up / norm
    right = np.array([-up[1], up[0]], dtype=np.float64)
    origin = np.asarray(palm_center, dtype=np.float64).reshape(2)
    angles = np.arange(
        float(FINGER_PROFILE_MIN_DEG),
        float(FINGER_PROFILE_MAX_DEG) + 1e-09,
        float(FINGER_PROFILE_STEP_DEG),
        dtype=np.float64,
    )
    r_step = max(0.001, float(FINGER_PROBE_R_STEP))
    rs = np.arange(
        float(FINGER_PROBE_R_MIN), float(FINGER_PROBE_R_MAX) + 1e-09, r_step, dtype=np.float64
    )
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
    """给定槽位组合，返回它**期望有几条指缝、各自在哪**（名义横向位置，从拇指侧排到小指侧）。

    规则：把组合切成"连续段"；段内相邻两指之间有缝，段与段之间是一条"跨缝"，
    跨缝的位置取中间那几条名义缝的均值。
        例：{拇,食}          -> [T-I]                      = [+0.72]
            {拇,中}          -> [T-I 与 I-M 的均值]         = [+0.46]
            {食,中}          -> [I-M]                      = [+0.20]
            {拇,食,小}       -> [T-I, (I-M,M-R,R-P)的均值]  = [+0.72, -0.33]
    """
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
    """Match observed tips to slots, preserving anatomical order.

    Every (slot subset, angular spread, rotation) must be scored INSIDE
    all three loops. Return each subset's best fit for diagnosis. The cost
    margin is a score difference, not a calibrated probability.

    Existing weights and priors are retained for a controlled bug fix.
    In particular, free rotation can still make adjacent identities ambiguous.
    """
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
    anchor = "thumb" if has_thumb else "palm_axis"
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
    shifts = (
        [None]
        if has_thumb
        else list(
            np.arange(
                -float(FINGER_NONTHUMB_SHIFT_DEG),
                float(FINGER_NONTHUMB_SHIFT_DEG) + 1e-09,
                float(FINGER_NONTHUMB_SHIFT_STEP),
            )
        )
    )
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
        penalty = (
            float(FINGER_CONTIGUOUS_PENALTY)
            if not has_thumb and non_thumb and (non_thumb != list(range(1, 1 + len(non_thumb))))
            else 0.0
        )
        combo_best = None
        for scale in map(float, FINGER_SEARCH_SCALES):
            len_sse = sum(
                (
                    (ln - scale * float(FINGER_TEMPLATE_LEN[slot])) ** 2
                    for slot, ln in zip(combo, lens)
                    if ln is not None
                )
            )
            for shift in shifts:
                delta = ta[0] - scale * float(template[0]) if shift is None else float(shift)
                predicted = [delta + scale * float(template[i]) for i in combo]
                sse = float(sum(((a - p) ** 2 for a, p in zip(ta, predicted))))
                costs = {
                    "angle": float(FINGER_W_ANGLE) * sse,
                    "length": float(FINGER_W_LEN) * len_sse,
                    "web": float(FINGER_W_WEB) * web_sse,
                    "scale": float(FINGER_W_SCALE) * (scale - 1.0) ** 2,
                    "contiguity": penalty,
                }
                total = sum(costs.values())
                detail = {
                    "delta_deg": delta,
                    "sse": sse,
                    "len_sse": len_sse,
                    "web_sse": web_sse,
                    "penalty": penalty,
                    "anchor": anchor,
                    "slots": list(combo),
                    "scale": scale,
                    "total": total,
                    "costs": costs,
                    "predicted_angles_deg": predicted,
                    "observed_web_x": obs_webs,
                    "expected_web_x": exp_webs,
                }
                if combo_best is None or total < combo_best["total"] - 1e-09:
                    combo_best = detail
                if best is None or total < best["total"] - 1e-09:
                    best = detail
        if combo_best is not None:
            ranked.append(combo_best)
    if best is None:
        return (
            [False] * 5,
            order,
            anchor,
            {"delta_deg": 0.0, "sse": None, "valid_assignment": False, "candidates": []},
        )
    ranked.sort(key=lambda item: item["total"])
    detail = dict(best)
    detail["valid_assignment"] = True
    detail["candidates"] = ranked
    detail["score_margin"] = ranked[1]["total"] - ranked[0]["total"] if len(ranked) > 1 else None
    states = [i in best["slots"] for i in range(5)]
    return (states, order, anchor, detail)


def _finger_states(
    mask: np.ndarray,
    palm_center,
    up,
    radius: float,
    tips,
    web_gaps=None,
    contour=None,
    closed_candidates=None,
):
    """Return five states and diagnostic measurements.

    Geometry mode projects local finger axes onto a transverse palm reference
    line to assign identity. Local width and aspect determine extension, so a
    detected candidate can be assigned to a curled finger. This reference-line
    intersection is a positioning feature, not an anatomical joint estimate.

    If geometric measurements are unavailable, retain the legacy angle method
    and record its fallback reason. Neither method reads image labels.
    The extension diagnostics mean local branch length in geometry mode and
    radial reach in legacy mode. Rendering distinguishes these two meanings.
    """
    geometry_failure = None
    if contour is None:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contour = max(contours, key=cv2.contourArea) if contours else None
    measured, geometry_failure = FG.recognize_fingers(
        mask, contour, palm_center, up, radius, tips or [], closed_candidates=closed_candidates
    )
    if measured is not None:
        extensions = [0.0] * 5
        angles_used = [0.0] * 5
        for feature in measured["features"]:
            slot = feature["slot"]
            extensions[slot] = feature["length_r"]
            x, y = feature["tip_local"]
            angles_used[slot] = float(np.degrees(np.arctan2(x, -y)))
        measured["anchor"] = "local_finger_axis"
        return (measured["states"], extensions, angles_used, measured, [0.0] * 5)
    tip_angles = []
    tip_lengths = []
    for t in tips or []:
        x = float(t["local"][0])
        y = float(t["local"][1])
        tip_angles.append(float(np.degrees(np.arctan2(x, -y))))
        tip_lengths.append(float(np.hypot(x, y)))
    web_xs = sorted([float(g["local"][0]) for g in web_gaps or []], reverse=True)
    states, _order, anchor, detail = _assign_finger_slots(tip_angles, tip_lengths, web_xs)
    best_total = float(detail.get("total", float("inf")))
    mirrored = False
    detail["mirrored"] = mirrored
    angles, profile = _radial_profile(mask, palm_center, up, radius)
    delta = float(detail.get("delta_deg", 0.0))
    scale_fit = float(detail.get("scale", 1.0))
    used = [delta + scale_fit * float(FINGER_TEMPLATE_DEG[i]) for i in range(5)]
    if mirrored:
        used = [-a for a in used]
    extensions = [float(np.interp(a, angles, profile)) for a in used]
    expected = [float(FINGER_TEMPLATE_LEN[i]) * scale_fit for i in range(5)]
    if anchor != "thumb" and (not states[0]):
        if extensions[0] >= float(FINGER_EXTENDED_MIN_R):
            states[0] = True
            detail["thumb_from_profile"] = True
    detail["anchor"] = anchor
    detail["tip_angles_deg"] = [round(a, 1) for a in tip_angles]
    detail["expect_len"] = [round(v, 2) for v in expected]
    detail["method"] = "angle_template_legacy"
    if geometry_failure is not None:
        detail["geometry_fallback_reason"] = geometry_failure
    return (states, extensions, used, detail, expected)


def finger_states_text(states) -> str:
    """把 5 位状态拼成人类可读的一行，例如 ``拇指+食指+小指`` 或 ``（全部蜷起）``。"""
    names = [FINGER_LABELS_CN[i] for i, s in enumerate(states) if s]
    return "+".join(names) if names else "（全部蜷起）"


def _load_cjk_font(size: int):
    """按需加载中文字体。找不到就返回 None（调用方必须能接受 None）。"""
    if size in _CJK_FONT_CACHE:
        return _CJK_FONT_CACHE[size]
    font = None
    try:
        from PIL import ImageFont
    except Exception:
        ImageFont = None
    if ImageFont is not None:
        for path in CJK_FONT_CANDIDATES:
            if os.path.exists(path):
                try:
                    font = ImageFont.truetype(path, int(size))
                    break
                except Exception:
                    font = None
    _CJK_FONT_CACHE[size] = font
    return font


def text_width(text: str, size: int) -> int:
    """量一段文字有多宽（像素）。面板尺寸和右对齐都靠它算准。"""
    font = _load_cjk_font(size)
    if font is not None:
        try:
            return int(font.getlength(text))
        except Exception:
            pass
    return int(len(text) * size * 0.62)


def _put_text_outlined(image, text, origin, scale=0.7, color=COLOR_LABEL, thickness=2):
    """带黑描边的**英文/数字**文字（cv2 版，快）。

    ⚠️ 只能画 ASCII。要画中文请用 draw_text_lines()。
    """
    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        COLOR_OUTLINE,
        thickness + 3,
        cv2.LINE_AA,
    )
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def draw_panel(image: np.ndarray, box, alpha: float = None, border=None) -> None:
    """画一块半透明深色面板，给 HUD 文字当底衬（否则白底/花背景上字看不清）。"""
    x0, y0, x1, y1 = (int(v) for v in box)
    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(image.shape[1], x1)
    y1 = min(image.shape[0], y1)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return
    region = image[y0:y1, x0:x1]
    dark = np.full_like(region, 16)
    cv2.addWeighted(
        region,
        1.0 - (HUD_PANEL_ALPHA if alpha is None else alpha),
        dark,
        HUD_PANEL_ALPHA if alpha is None else alpha,
        0.0,
        dst=region,
    )
    cv2.rectangle(
        image, (x0, y0), (x1 - 1, y1 - 1), HUD_PANEL_BORDER if border is None else border, 1
    )


def draw_text_lines(image: np.ndarray, lines, origin, panel: bool = False) -> int:
    """画多行文字，**支持中文**。返回整块画完后底部的 y 坐标。

    lines: [(文字, 字号px, BGR颜色), ...]；origin 是第一行的左上角。
    panel: True = 先在文字底下铺一块半透明面板再写字。

    ★ 什么时候该用 panel：
      小字号的中文笔画很密，**8 方向黑描边会把字内空隙填满，糊成一团黑块**。
      这种情况用"底衬面板 + 不描边"才清楚（ROI 里那几行就是这么处理的）。
      大字号（>=22px）可以不用面板，靠描边就够。

    ★ 性能：整块只做一次 PIL 往返，而且**只转换文字覆盖的那一小块区域**。
      如果对整幅 1920x1440 反复做 BGR<->RGB 转换，每行都要好几毫秒，
      一屏七八行就吃掉几十毫秒 —— 视频流下这是不能接受的。
    """
    lines = [ln for ln in lines if ln and str(ln[0])]
    if not lines:
        return int(origin[1])
    x0, y0 = (int(origin[0]), int(origin[1]))
    sizes = [int(ln[1]) for ln in lines]
    block_w = max((text_width(str(ln[0]), int(ln[1])) for ln in lines)) + 10
    block_h = sum(sizes) + HUD_LINE_GAP * (len(lines) - 1) + 10
    bx0 = max(0, x0 - 4)
    by0 = max(0, y0 - 4)
    bx1 = min(image.shape[1], x0 + block_w + 4)
    by1 = min(image.shape[0], y0 + block_h + 4)
    if bx1 - bx0 < 2 or by1 - by0 < 2:
        return y0 + block_h
    use_pil = True
    try:
        from PIL import Image, ImageDraw
    except Exception:
        use_pil = False
    if use_pil and any((_load_cjk_font(s) is None for s in sizes)):
        use_pil = False
    if panel:
        draw_panel(image, (bx0, by0, bx1, by1), alpha=0.68)
    if use_pil:
        sub = image[by0:by1, bx0:bx1]
        pil = Image.fromarray(cv2.cvtColor(sub, cv2.COLOR_BGR2RGB))
        painter = ImageDraw.Draw(pil)
        cursor = y0 - by0
        for text, size, color in lines:
            font = _load_cjk_font(int(size))
            px, py = (x0 - bx0, cursor)
            rgb = (int(color[2]), int(color[1]), int(color[0]))
            if not panel:
                for dx, dy in ((1, 0), (0, 1), (1, 1)):
                    painter.text((px + dx, py + dy), str(text), font=font, fill=(0, 0, 0))
            painter.text((px, py), str(text), font=font, fill=rgb)
            cursor += int(size) + HUD_LINE_GAP
        image[by0:by1, bx0:bx1] = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
        return y0 + block_h
    cursor = y0
    for text, size, color in lines:
        safe = str(text).encode("ascii", "replace").decode("ascii")
        scale = max(0.4, int(size) / 30.0)
        if not panel:
            cv2.putText(
                image,
                safe,
                (x0, cursor + int(size)),
                cv2.FONT_HERSHEY_SIMPLEX,
                scale,
                COLOR_OUTLINE,
                3,
                cv2.LINE_AA,
            )
        cv2.putText(
            image,
            safe,
            (x0, cursor + int(size)),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            1,
            cv2.LINE_AA,
        )
        cursor += int(size) + HUD_LINE_GAP
    return y0 + block_h


def _draw_recognition_compact(
    base: np.ndarray, result: Optional[dict], show_summary: bool = True
) -> np.ndarray:
    """Only observed extended fingertip markers and the final finger result."""
    if result is None:
        if show_summary:
            draw_text_lines(base, [("未检测到手", 26, (70, 100, 255))], (16, 14), panel=True)
        return base
    match = result.get("finger_match") or {}
    if match.get("anchor") == "local_finger_axis":
        for feature in match.get("features", []):
            if not feature.get("extended"):
                continue
            slot = int(feature.get("slot", -1))
            if not 0 <= slot < 5:
                continue
            point = tuple((int(round(v)) for v in feature["point"]))
            cv2.circle(base, point, 6, COLOR_FINGER_ON, -1, cv2.LINE_AA)
            cv2.circle(base, point, 8, (255, 255, 255), 1, cv2.LINE_AA)
            _put_text_outlined(
                base, FINGER_BOX_TAGS[slot], (point[0] + 10, point[1] - 9), 0.65, COLOR_FINGER_ON, 2
            )
    else:
        for tip in result.get("tips", []):
            point = tuple((int(round(v)) for v in tip["point"]))
            cv2.circle(base, point, 6, COLOR_FINGER_ON, -1, cv2.LINE_AA)
    if show_summary:
        lines = [("手指：" + result.get("finger_text", "?"), 26, (0, 230, 255))]
        draw_text_lines(base, lines, (16, 14), panel=True)
    return base


def _draw_recognition(
    base: np.ndarray, result: Optional[dict], show_summary: bool = True, detailed=False
) -> np.ndarray:
    """Draw compact results or detailed geometry without changing any decision."""
    if not detailed:
        return _draw_recognition_compact(base, result, show_summary)
    height, width = base.shape[:2]
    if result is None:
        if show_summary:
            _put_text_outlined(base, "no hand", (20, 40), 0.9, (0, 0, 255), 2)
        return base
    palm_center = result["palm_center"]
    radius = result["palm_radius"]
    up = result["up"]
    wrist = result["wrist"]
    center_px = (int(round(palm_center[0])), int(round(palm_center[1])))
    hull = result.get("hull")
    if hull is not None and len(hull) >= 3:
        cv2.drawContours(base, [hull], -1, (255, 255, 0), 1, cv2.LINE_AA)
    ellipse = result.get("_display_palm_ellipse")
    if ellipse is None:
        cv2.circle(base, center_px, int(round(radius)), COLOR_PALM, 2)
    else:
        axes, angle = ellipse
        cv2.ellipse(base, center_px, axes, angle, 0, 360, COLOR_PALM, 2)
    cv2.drawMarker(base, center_px, COLOR_PALM, cv2.MARKER_CROSS, 40, 3)
    axis_length = radius * 1.4
    tip_up = (
        int(round(palm_center[0] + up[0] * axis_length)),
        int(round(palm_center[1] + up[1] * axis_length)),
    )
    cv2.arrowedLine(base, center_px, tip_up, COLOR_WRIST, 2, tipLength=0.15)
    right = np.asarray(result.get("_display_right", [-up[1], up[0]]))
    tip_right = (
        int(round(palm_center[0] + right[0] * radius * 0.7)),
        int(round(palm_center[1] + right[1] * radius * 0.7)),
    )
    cv2.arrowedLine(base, center_px, tip_right, COLOR_WRIST, 2, tipLength=0.15)
    if wrist is not None:
        wrist_px = (int(round(wrist[0])), int(round(wrist[1])))
        cv2.circle(base, wrist_px, 6, COLOR_WRIST, -1)
        cv2.line(base, wrist_px, center_px, COLOR_WRIST, 1)
    truncate_point = palm_center - up * (float(RECOG_TRUNCATE_FACTOR) * radius)
    half = float(RECOG_TRUNCATE_LINE_HALF) * radius
    raw1 = truncate_point + right * half
    raw2 = truncate_point - right * half
    p1 = (int(np.clip(round(raw1[0]), 0, width - 1)), int(np.clip(round(raw1[1]), 0, height - 1)))
    p2 = (int(np.clip(round(raw2[0]), 0, width - 1)), int(np.clip(round(raw2[1]), 0, height - 1)))
    cv2.line(base, p1, p2, (128, 128, 128), 2)
    for gap in result["gaps"]:
        point = (int(round(gap["point"][0])), int(round(gap["point"][1])))
        cv2.circle(base, point, 10, COLOR_GAP, 2)
    geometry_match = result.get("finger_match") or {}
    using_geometry = geometry_match.get("anchor") == "local_finger_axis"
    if using_geometry:
        for feature in geometry_match.get("features", []):
            slot = feature["slot"]
            tip_px = tuple((int(round(v)) for v in feature["point"]))
            root = feature.get("root_point") or feature["base_point"]
            root_px = tuple((int(round(v)) for v in root))
            col = COLOR_FINGER_ON if feature["extended"] else COLOR_FINGER_OFF
            cv2.line(base, root_px, tip_px, col, 3, cv2.LINE_AA)
            cv2.circle(base, root_px, 7, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.circle(base, tip_px, 7, col, -1, cv2.LINE_AA)
            text = "%s %s" % (FINGER_BOX_TAGS[slot], "ON" if feature["extended"] else "OFF")
            _put_text_outlined(base, text, (tip_px[0] + 12, tip_px[1] - 12), 0.7, col, 2)
    else:
        up_v = np.asarray(up, dtype=np.float64).reshape(2)
        right_v = np.asarray(result.get("_display_right", [-up_v[1], up_v[0]]))
        ray_c = float(radius)
        ray_len = float(FINGER_DEBUG_RAY_LEN_R) * ray_c
        angles = result.get("finger_ray_angles") or []
        exts = result.get("finger_extensions") or []
        exps = result.get("finger_expect_len") or []
        sts = result.get("finger_states") or []
        for i in range(min(5, len(angles))):
            a = float(angles[i])
            d = np.array([np.sin(np.radians(a)), -np.cos(np.radians(a))], dtype=np.float64)
            dirv = right_v * d[0] + up_v * -d[1]
            n = float(np.linalg.norm(dirv))
            if n < 1e-09:
                continue
            if "_display_right" not in result:
                dirv = dirv / n
            col = COLOR_FINGER_ON if i < len(sts) and sts[i] else COLOR_FINGER_OFF
            p_ray = (
                int(round(palm_center[0] + dirv[0] * ray_len)),
                int(round(palm_center[1] + dirv[1] * ray_len)),
            )
            cv2.line(base, center_px, p_ray, col, 1, cv2.LINE_AA)
            if i < len(exps):
                e = float(exps[i])
                pe = (
                    int(round(palm_center[0] + dirv[0] * e * ray_c)),
                    int(round(palm_center[1] + dirv[1] * e * ray_c)),
                )
                cv2.circle(base, pe, 13, (0, 255, 255), 2, cv2.LINE_AA)
            if i < len(exts):
                m = float(exts[i])
                pm = (
                    int(round(palm_center[0] + dirv[0] * m * ray_c)),
                    int(round(palm_center[1] + dirv[1] * m * ray_c)),
                )
                cv2.circle(base, pm, 7, col, -1, cv2.LINE_AA)
            if i < len(exps) and i < len(exts):
                pe = (
                    int(round(palm_center[0] + dirv[0] * float(exps[i]) * ray_c)),
                    int(round(palm_center[1] + dirv[1] * float(exps[i]) * ray_c)),
                )
                pm = (
                    int(round(palm_center[0] + dirv[0] * float(exts[i]) * ray_c)),
                    int(round(palm_center[1] + dirv[1] * float(exts[i]) * ray_c)),
                )
                cv2.line(base, pe, pm, (0, 255, 255), 1, cv2.LINE_AA)
            lab_at = (
                int(round(palm_center[0] + dirv[0] * (ray_len + 26))),
                int(round(palm_center[1] + dirv[1] * (ray_len + 26))),
            )
            txt = "%s" % FINGER_BOX_TAGS[i]
            if i < len(exts) and i < len(exps):
                txt += " %.1f/%.1f" % (float(exts[i]), float(exps[i]))
            _put_text_outlined(base, txt, lab_at, 0.7, col, 2)
    for finger in [] if using_geometry else result["fingers"]:
        tip_px = (int(round(finger["tip"][0])), int(round(finger["tip"][1])))
        base_px = (int(round(finger["base"][0])), int(round(finger["base"][1])))
        cv2.line(base, base_px, tip_px, COLOR_TIP, 3)
        cv2.circle(base, base_px, 7, COLOR_BASE, -1)
        cv2.circle(base, tip_px, 9, COLOR_TIP, -1)
        cv2.circle(base, tip_px, 11, (255, 255, 255), 2)
        _put_text_outlined(
            base, str(finger["index"]), (tip_px[0] + 14, tip_px[1] - 10), 0.8, COLOR_TIP, 2
        )
        mid = ((tip_px[0] + base_px[0]) // 2, (tip_px[1] + base_px[1]) // 2)
        _put_text_outlined(
            base, "%.2fR" % finger["length_ratio"], (mid[0] + 12, mid[1]), 0.55, COLOR_LABEL, 1
        )
    states = list(result.get("finger_states") or [])
    extensions = list(result.get("finger_extensions") or [])
    while len(states) < 5:
        states.append(False)
    while len(extensions) < 5:
        extensions.append(0.0)
    box = int(FINGER_BOX_SIZE)
    gap = int(FINGER_BOX_GAP)
    bx0 = int(HUD_MARGIN)
    by0 = int(HUD_MARGIN) if show_summary else max(180, int(HUD_MARGIN))
    for pos in range(5):
        i = 4 - pos
        x = bx0 + pos * (box + gap)
        if states[i]:
            cv2.rectangle(base, (x, by0), (x + box, by0 + box), COLOR_FINGER_ON, -1)
        else:
            cv2.rectangle(
                base, (x, by0), (x + box, by0 + box), COLOR_FINGER_OFF, int(FINGER_BOX_THICKNESS)
            )
        cv2.putText(
            base,
            FINGER_BOX_TAGS[i],
            (x + int(box * 0.32), by0 + box + 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            COLOR_LABEL,
            2,
            cv2.LINE_AA,
        )
    squares_bottom = by0 + box + 36
    palm_rel = result.get("palm_rel", (0.0, 0.0))
    hud_lines = []
    if show_summary:
        hud_lines.append(("手指: %s" % result.get("finger_text", "?"), 24, (0, 230, 255)))
    hud_lines.append(
        (
            ("局部长度 " if using_geometry else "延伸 ")
            + "  ".join(("%.1f" % e for e in extensions)),
            16,
            (185, 185, 185),
        )
    )
    hud_lines.append(
        (
            (
                "指尖与弯曲指节共同定位身份"
                if geometry_match.get("topology_used")
                else (
                    "局部指轴定位身份，宽度与细长程度判断伸缩"
                    if using_geometry
                    else "状态由模板匹配决定；延伸长度为辅助量"
                )
            ),
            15,
            (150, 150, 150),
        )
    )
    hud_lines.append((" ", 8, (0, 0, 0)))
    hud_lines.append(
        (
            "fingers %d    gaps %d"
            % (sum((bool(on) for on in result["finger_states"])), len(result["gaps"])),
            18,
            COLOR_LABEL,
        )
    )
    hud_lines.append(
        (
            "solidity %.2f    span %.0f deg"
            % (result.get("solidity", 0.0), result.get("angle_span_deg", 0.0)),
            18,
            COLOR_LABEL,
        )
    )
    hud_lines.append(
        (
            "R %.0f px    mean len %.2fR" % (radius, result.get("mean_length_ratio", 0.0)),
            18,
            COLOR_LABEL,
        )
    )
    hud_lines.append(("palm rel (%.2f, %.2f)" % palm_rel, 18, COLOR_LABEL))
    if wrist is not None:
        hud_lines.append(("wrist (%d, %d)" % (int(wrist[0]), int(wrist[1])), 18, COLOR_LABEL))
    else:
        hud_lines.append(("wrist not found -> up = image up", 18, (180, 180, 180)))
    draw_text_lines(base, hud_lines, (int(HUD_MARGIN), int(squares_bottom)), panel=True)
    return base


def _transform_display_points(points, matrix):
    """Map drawing coordinates without changing the recognizer's arrays."""
    points = np.asarray(points, dtype=np.float64)
    return points @ matrix[:2, :2].T + matrix[:2, 2]


def _transform_recognition_for_display(result, matrix):
    """Copy only drawing geometry; finger decisions and processing masks stay intact."""
    if result is None:
        return None
    out = dict(result)
    point = lambda value: tuple(_transform_display_points(value, matrix))
    linear = matrix[:2, :2]
    scale = float(np.sqrt(abs(np.linalg.det(linear))))
    out["palm_center"] = _transform_display_points(result["palm_center"], matrix)
    out["wrist"] = None if result["wrist"] is None else point(result["wrist"])
    up = np.asarray(result["up"], np.float64)
    out["up"] = linear @ up / scale
    out["_display_right"] = linear @ np.array([-up[1], up[0]]) / scale
    out["palm_radius"] = result["palm_radius"] * scale
    u, singular, _ = np.linalg.svd(linear)
    if not np.isclose(singular[0], singular[1]):
        axes = tuple((max(1, int(round(result["palm_radius"] * s))) for s in singular))
        angle = float(np.degrees(np.arctan2(u[1, 0], u[0, 0])))
        out["_display_palm_ellipse"] = (axes, angle)
    for key in ("hull", "contour"):
        if result.get(key) is not None:
            out[key] = np.rint(_transform_display_points(result[key], matrix)).astype(np.int32)
    for key, fields in (("tips", ("point",)), ("gaps", ("point",)), ("fingers", ("tip", "base"))):
        out[key] = [
            dict(item, **{f: point(item[f]) for f in fields if item.get(f) is not None})
            for item in result.get(key, [])
        ]
    match = result.get("finger_match") or {}
    if "features" in match:
        fields = ("point", "base_point", "root_point", "axis_origin")
        out["finger_match"] = dict(
            match,
            features=[
                dict(item, **{f: point(item[f]) for f in fields if item.get(f) is not None})
                for item in match["features"]
            ],
        )
    return out


def _rotation_edge_contact(blob):
    """Return the strongest foreground/image-edge contact and its validity."""
    height, width = blob.shape
    lines = (blob[:, 0], blob[:, -1], blob[0], blob[-1])
    counts = [int(np.count_nonzero(line)) for line in lines]
    edge = max(range(4), key=lambda index: counts[index])
    count = counts[edge]
    valid = (
        int(AUTO_ROTATE_EDGE_MIN_PIXELS)
        <= count
        <= float(AUTO_ROTATE_EDGE_MAX_FRAC) * max(height, width)
    )
    return (edge, count, valid)


def _rotation_edge_arm(blob, edge):
    """Centroid-to-contact direction; unchanged evidence from the old path."""
    height, width = blob.shape
    lines = (blob[:, 0], blob[:, -1], blob[0], blob[-1])
    contact = np.flatnonzero(lines[edge])
    ys, xs = np.nonzero(blob)
    if not len(contact) or not len(xs):
        return None
    position = float(contact.mean())
    target = (
        (0.0, position),
        (float(width - 1), position),
        (position, 0.0),
        (position, float(height - 1)),
    )[edge]
    return np.asarray(target) - np.array([float(xs.mean()), float(ys.mean())])


def _rotation_arm_angle(direction):
    if direction is None or not np.isfinite(direction).all():
        return 0.0
    if float(np.linalg.norm(direction)) < 1e-06:
        return 0.0
    return -float(np.degrees(np.arctan2(direction[0], direction[1])))


def _rotation_internal_arm(small, blob, *, known_board):
    """Find a repaired board exit or a broad skin-to-dark-clothing boundary.

    This only runs when the original contact test is inconclusive. Geometry
    proposes a terminal boundary, but cannot by itself identify it as a cuff.
    Ambiguous or absent clothing evidence returns None without inventing an
    arm direction. Operations are confined to the existing small image.
    """
    contours, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    rough = np.zeros_like(blob)
    cv2.drawContours(rough, contours, -1, 255, -1)
    if not np.any(rough == 0):
        return None
    _, rough_radius, _, _ = cv2.minMaxLoc(cv2.distanceTransform(rough, cv2.DIST_L2, 5))
    if not np.isfinite(rough_radius) or rough_radius < 2.0:
        return None
    kernel_size = 2 * max(1, round(0.08 * rough_radius)) + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    clean = cv2.morphologyEx(blob, cv2.MORPH_OPEN, kernel)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(clean, 8)
    if count <= 1:
        return None
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    clean = np.uint8(labels == biggest) * 255
    edge, _, valid = _rotation_edge_contact(clean)
    if known_board and valid:
        return _rotation_edge_arm(clean, edge)
    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    filled = np.zeros_like(clean)
    cv2.drawContours(filled, [contour], -1, 255, -1)
    if not np.any(filled == 0):
        return None
    distance = cv2.distanceTransform(filled, cv2.DIST_L2, 5)
    _, radius, _, location = cv2.minMaxLoc(distance)
    if not np.isfinite(radius) or radius < 2.0:
        return None
    max_y, max_x = np.nonzero(distance == radius)
    center = np.array([float(max_x.mean()), float(max_y.mean())])
    ys, xs = np.nonzero(filled)
    points = np.column_stack((xs, ys)).astype(np.float64)
    centroid = points.mean(axis=0)
    eigenvalues, eigenvectors = np.linalg.eigh(np.cov(points.T))
    if (
        not np.isfinite(eigenvalues).all()
        or eigenvalues[-1] <= 0.0
        or np.isclose(eigenvalues[0], eigenvalues[-1], rtol=1e-06, atol=1e-09)
    ):
        return None
    major = eigenvectors[:, -1]
    polygon = cv2.approxPolyDP(contour, max(1.0, 0.05 * radius), True).reshape(-1, 2)
    polygon = polygon.astype(np.float64)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    segments = []
    for first, second in zip(polygon, np.roll(polygon, -1, axis=0)):
        edge_vector = second - first
        length = float(np.linalg.norm(edge_vector))
        if length < 1.0:
            continue
        midpoint = (first + second) * 0.5
        normal = np.array([-edge_vector[1], edge_vector[0]]) / length
        if float(normal @ (midpoint - centroid)) < 0.0:
            normal = -normal
        if abs(float(normal @ major)) < 0.5:
            continue
        offset = midpoint - center
        reach = float(np.linalg.norm(offset))
        if reach < radius or float(normal @ offset) / max(reach, 1e-09) < 0.5:
            continue
        samples = first + np.linspace(0.15, 0.85, 9)[:, None] * edge_vector
        inside = np.rint(samples - 0.2 * radius * normal).astype(np.int32)
        outside = np.rint(samples + 0.2 * radius * normal).astype(np.int32)
        if any(
            (
                np.any(sample < 0)
                or np.any(sample[:, 0] >= width)
                or np.any(sample[:, 1] >= height)
                for sample in (inside, outside)
            )
        ):
            continue
        if float(np.mean(filled[inside[:, 1], inside[:, 0]] > 0)) < 0.75:
            continue
        inner_values = gray[inside[:, 1], inside[:, 0]].astype(np.float64)
        outer_values = gray[outside[:, 1], outside[:, 0]].astype(np.float64)
        inner_level = float(np.median(inner_values))
        outer_level = float(np.median(outer_values))
        contrast = float(np.median(inner_values - outer_values))
        if inner_level <= 0.0 or outer_level >= 0.5 * inner_level or contrast <= 0.0:
            continue
        pole = 1 if float(normal @ major) >= 0.0 else -1
        segments.append((pole, midpoint, length, contrast))
    targets = []
    for pole in (-1, 1):
        supported = [segment for segment in segments if segment[0] == pole]
        support = sum((segment[2] for segment in supported))
        if support < 0.65 * radius:
            continue
        weight = sum((length * contrast for _, _, length, contrast in supported))
        if weight <= 0.0:
            continue
        target = (
            sum((midpoint * length * contrast for _, midpoint, length, contrast in supported))
            / weight
        )
        targets.append(target)
    if len(targets) != 1:
        return None
    return targets[0] - center


def detect_hand_rotation(frame, *, color_state=None, calibrate_background=False):
    """Estimate rotation from edge contact, or observed internal cuff evidence."""
    height, width = frame.shape[:2]
    if height < 32 or width < 32:
        return 0.0
    scale = min(1.0, float(AUTO_ROTATE_DOWNSCALE) / max(height, width))
    small = (
        cv2.resize(
            frame,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
        if scale < 1.0
        else frame
    )
    ycrcb = cv2.cvtColor(small, cv2.COLOR_BGR2YCrCb)
    if color_state is None:
        color_state = SimpleNamespace(flags={"manual_roi": False})
    calibration = (
        cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        if calibrate_background and small is not frame
        else ycrcb
    )
    lower, upper = _skin_color_bounds(calibration, color_state, calibrate=calibrate_background)
    slack = float(AUTO_ROTATE_COLOR_SLACK)
    lower = (lower[0], max(0, round(lower[1] - slack)), max(0, round(lower[2] - slack)))
    upper = (upper[0], min(255, round(upper[1] + slack)), min(255, round(upper[2] + slack)))
    mask = cv2.inRange(ycrcb, lower, upper)
    if int(np.count_nonzero(mask)) < 64:
        return 0.0
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if count <= 1:
        return 0.0
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if int(stats[biggest, cv2.CC_STAT_AREA]) < float(AUTO_ROTATE_MIN_AREA) * mask.size:
        return 0.0
    blob = np.uint8(labels == biggest) * 255
    edge, contact_count, valid = _rotation_edge_contact(blob)
    if valid:
        return _rotation_arm_angle(_rotation_edge_arm(blob, edge))
    if not calibrate_background and contact_count >= int(AUTO_ROTATE_EDGE_MIN_PIXELS):
        return 0.0
    direction = _rotation_internal_arm(small, blob, known_board=calibrate_background)
    return _rotation_arm_angle(direction)


def _rotation_matrix_and_size(width: int, height: int, angle: float):
    """One pixel-center transform for both the colour frame and validity mask."""
    center = ((width - 1) * 0.5, (height - 1) * 0.5)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos_a, sin_a = (abs(matrix[0, 0]), abs(matrix[0, 1]))
    new_width = int(height * sin_a + width * cos_a)
    new_height = int(height * cos_a + width * sin_a)
    matrix[0, 2] += (new_width - 1) * 0.5 - center[0]
    matrix[1, 2] += (new_height - 1) * 0.5 - center[1]
    return (matrix, (new_width, new_height))


def auto_rotate_frame(frame, *, color_state=None, calibrate_background=False):
    """Estimate and apply orientation independently for this frame."""
    angle = detect_hand_rotation(
        frame, color_state=color_state, calibrate_background=calibrate_background
    )
    if abs(angle) < 1.0:
        return (frame, angle)
    height, width = frame.shape[:2]
    matrix, size = _rotation_matrix_and_size(width, height, angle)
    return (
        cv2.warpAffine(
            frame, matrix, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
        ),
        angle,
    )


def prepare_hand_frame(frame: np.ndarray):
    """Locate the board, crop actual pixels, then normalize hand direction.

    This entry point receives pixels only: no filename, label or data-group
    switches. Uncertain board detection leaves the image intact.
    """
    height, width = frame.shape[:2]
    enabled = True
    board = detect_board(frame)
    if board["detected"]:
        x0, y0, x1, y1 = (int(v) for v in board["rect"])
        if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
            raise ValueError("Board detector returned an invalid crop")
        working = np.ascontiguousarray(frame[y0:y1, x0:x1])
    else:
        working = frame
        x0, y0 = (0, 0)
    color_state = SimpleNamespace(flags={"calibrating_board": True})
    rotated, angle = auto_rotate_frame(
        working, color_state=color_state, calibrate_background=bool(board["detected"])
    )
    input_to_prepared = np.array([[1.0, 0.0, -x0], [0.0, 1.0, -y0], [0.0, 0.0, 1.0]])
    if abs(angle) >= 1.0:
        rotation, _ = _rotation_matrix_and_size(working.shape[1], working.shape[0], angle)
        rotation_h = np.eye(3)
        rotation_h[:2] = rotation
        input_to_prepared = rotation_h @ input_to_prepared
    board_valid = None
    if board["detected"] and len(board.get("polygon", [])) >= 3:
        board_valid = np.zeros(working.shape[:2], np.uint8)
        polygon = np.rint(np.asarray(board["polygon"]) - [x0, y0]).astype(np.int32)
        cv2.fillConvexPoly(board_valid, polygon, 255)
        if abs(angle) >= 1.0:
            matrix, size = _rotation_matrix_and_size(working.shape[1], working.shape[0], angle)
            board_valid = cv2.warpAffine(
                board_valid,
                matrix,
                size,
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
    info = {
        "board": board,
        "rotation_deg": float(angle),
        "input_to_prepared": input_to_prepared,
        "input_size": (width, height),
        "cropped_size": (working.shape[1], working.shape[0]),
        "board_valid_mask": board_valid,
    }
    if board["detected"] and "scene_shift" in color_state.flags:
        info["scene_calibration"] = {
            "shift": color_state.flags["scene_shift"],
            "background": color_state.flags.get("scene_bg_estimate", {}),
        }
    return (rotated, info)


def _fit_edge(points, extent):
    """Deterministic consensus line: dependent coordinate = a * axis + b."""
    points = np.asarray(points, np.float64)
    if len(points) < 12:
        return None
    ids = np.unique(np.linspace(0, len(points) - 1, min(48, len(points))).astype(int))
    selected = points[ids]
    first, second = np.triu_indices(len(selected), 1)
    delta = selected[second] - selected[first]
    usable = np.abs(delta[:, 0]) >= 0.2 * np.ptp(points[:, 0])
    delta, first = (delta[usable], first[usable])
    slopes = delta[:, 1] / delta[:, 0]
    intercepts = selected[first, 1] - slopes * selected[first, 0]
    usable = np.abs(slopes) <= np.tan(np.deg2rad(20.0))
    slopes, intercepts = (slopes[usable], intercepts[usable])
    if not len(slopes):
        return None
    tolerance = max(1.5, 0.005 * extent)
    best = None
    for start in range(0, len(slopes), 128):
        aa, bb = (slopes[start : start + 128], intercepts[start : start + 128])
        error = np.abs(points[None, :, 1] - aa[:, None] * points[None, :, 0] - bb[:, None])
        inside = error <= tolerance
        votes = inside.sum(axis=1)
        score = votes - np.minimum(error, tolerance).mean(axis=1) / (2 * tolerance)
        winner = int(np.argmax(score))
        if best is None or score[winner] > best[0]:
            best = (float(score[winner]), float(aa[winner]), float(bb[winner]))
    _, slope, intercept = best
    for _ in range(3):
        keep = np.abs(points[:, 1] - slope * points[:, 0] - intercept) <= tolerance
        if np.count_nonzero(keep) < 10:
            return None
        slope, intercept = np.polyfit(points[keep, 0], points[keep, 1], 1)
    residual = np.abs(points[:, 1] - slope * points[:, 0] - intercept)
    keep = residual <= tolerance
    span = np.ptp(points[keep, 0]) / max(1, np.ptp(points[:, 0]))
    return {
        "a": float(slope),
        "b": float(intercept),
        "support": float(keep.mean()),
        "span": float(span),
        "error": float(np.median(residual[keep])),
        "tolerance": tolerance,
    }


def _intersection(vertical, horizontal):
    av, bv = (vertical["a"], vertical["b"])
    ah, bh = (horizontal["a"], horizontal["b"])
    x = (av * bh + bv) / (1 - av * ah)
    return [x, ah * x + bh]


def _candidate(mask, height, width):
    result = {
        "rect": (0, 0, width, height),
        "detected": False,
        "confidence": 0.0,
        "reason": "no_pale_region",
        "lines": [],
        "polygon": [],
    }
    edges = []
    for axis in (0, 1):
        oriented = mask if axis == 0 else mask.T
        rows = np.flatnonzero(np.any(oriented, axis=1))
        low = np.argmax(oriented, axis=1)
        high = oriented.shape[1] - 1 - np.argmax(oriented[:, ::-1], axis=1)
        for name, envelope in zip(
            ("left", "right") if axis == 0 else ("top", "bottom"), (low, high)
        ):
            edge = _fit_edge(np.column_stack((rows, envelope[rows])), oriented.shape[1])
            if edge is None or edge["support"] < 0.28 or edge["span"] < 0.28:
                result["reason"] = "insufficient_straight_edges"
                return result
            edge["name"] = name
            edges.append(edge)
    left, right, top, bottom = edges
    if sum((edge["support"] >= 0.6 and edge["span"] >= 0.7 for edge in edges)) < 2:
        result["reason"] = "insufficient_unoccluded_edges"
        return result
    if abs(left["a"] - right["a"]) > 0.15 or abs(top["a"] - bottom["a"]) > 0.15:
        result["reason"] = "perspective_or_conflicting_edges"
        return result
    polygon = np.asarray(
        [
            _intersection(left, top),
            _intersection(right, top),
            _intersection(right, bottom),
            _intersection(left, bottom),
        ],
        np.float32,
    )
    sh, sw = mask.shape
    polygon[:, 0] = np.clip(polygon[:, 0], 0, sw - 1)
    polygon[:, 1] = np.clip(polygon[:, 1], 0, sh - 1)
    area = abs(float(cv2.contourArea(polygon)))
    if area < 0.16 * mask.size or not cv2.isContourConvex(polygon):
        result["reason"] = "invalid_board_geometry"
        return result
    polygon_mask = np.zeros(mask.shape, np.uint8)
    cv2.fillConvexPoly(polygon_mask, np.rint(polygon).astype(np.int32), 1)
    inside = polygon_mask != 0
    coverage = float(np.mean(mask[inside]))
    if coverage < 0.45:
        result["reason"] = "insufficient_board_surface"
        return result
    visible = 0
    for edge, extent, along in zip(edges, (sw, sw, sh, sh), (sh, sh, sw, sw)):
        values = edge["a"] * np.array([0.0, along - 1.0]) + edge["b"]
        clipped = np.all(values <= 2) or np.all(values >= extent - 3)
        edge["image_boundary"] = bool(clipped)
        visible += not clipped
    if visible < 2:
        result["reason"] = "board_edges_out_of_view"
        return result
    sx, sy = (width / sw, height / sh)
    original = polygon * np.array([sx, sy], np.float32)
    x0, y0 = np.floor(original.min(axis=0)).astype(int)
    x1, y1 = np.ceil((polygon.max(axis=0) + 1) * [sx, sy]).astype(int)
    rect = (max(0, int(x0)), max(0, int(y0)), min(width, int(x1)), min(height, int(y1)))
    if rect[2] - rect[0] < width * 0.2 or rect[3] - rect[1] < height * 0.2:
        result["reason"] = "board_too_narrow"
        return result
    confidence = float(np.clip(0.6 * min((e["support"] for e in edges)) + 0.4 * coverage, 0, 1))
    lines = []
    for index, edge in enumerate(edges):
        ends = {0: (0, 3), 1: (1, 2), 2: (0, 1), 3: (3, 2)}[index]
        lines.append(
            {
                "name": edge["name"],
                "points": original[list(ends)].tolist(),
                "support": edge["support"],
                "image_boundary": edge["image_boundary"],
            }
        )
    result.update(
        rect=rect,
        detected=True,
        confidence=confidence,
        reason="straight_pale_board",
        lines=lines,
        polygon=original.tolist(),
        surface_coverage=coverage,
    )
    return result


def _regions(pale, height, width, restore=None):
    count, labels, stats, _ = cv2.connectedComponentsWithStats(pale, connectivity=8)
    candidates = []
    for label in (1 + np.argsort(stats[1:, cv2.CC_STAT_AREA])[::-1])[:4]:
        if stats[label, cv2.CC_STAT_AREA] < 0.12 * pale.size:
            break
        mask = labels == label
        if restore is not None:
            mask = (cv2.dilate(mask.astype(np.uint8), np.ones((5, 5), np.uint8)) != 0) & (
                restore != 0
            )
        candidate = _candidate(mask, height, width)
        candidate["region_area"] = float(stats[label, cv2.CC_STAT_AREA] / pale.size)
        candidates.append(candidate)
    return candidates


def _select(candidates, fallback):
    valid = sorted(
        (r for r in candidates if r["detected"]), key=lambda r: r["confidence"], reverse=True
    )
    if not valid:
        return candidates[0] if candidates else fallback
    if len(valid) > 1 and valid[1]["region_area"] >= 0.8 * valid[0]["region_area"]:
        return dict(fallback, reason="ambiguous_pale_regions")
    return valid[0]


def detect_board(frame):
    """Return rect=(x0,y0,x1,y1), detected, confidence, reason, lines.

    Coordinates refer to the original image; upper bounds are exclusive.
    Confidence is a geometric support score, not a calibrated probability.
    ``rect`` is the whole image when ``detected`` is false.  ``lines`` and
    ``polygon`` describe detected straight board boundaries for diagnostics.
    At least two actual board edges must be visible.  Strong perspective,
    textured cloth and ambiguous borders are deliberately rejected.
    """
    if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("detect_board expects a BGR image")
    height, width = frame.shape[:2]
    fallback = {
        "rect": (0, 0, width, height),
        "detected": False,
        "confidence": 0.0,
        "reason": "no_pale_region",
        "lines": [],
        "polygon": [],
    }
    if min(height, width) < 32:
        return dict(fallback, reason="image_too_small")
    scale = min(1.0, 640.0 / max(height, width))
    small = (
        cv2.resize(
            frame, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA
        )
        if scale < 1
        else frame
    )
    smooth = cv2.GaussianBlur(small, (5, 5), 0)
    gray = cv2.cvtColor(smooth, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(smooth, cv2.COLOR_BGR2HSV)
    threshold, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    pale = ((hsv[:, :, 1] <= 45) & (gray >= max(45, threshold))).astype(np.uint8)
    pale = cv2.morphologyEx(
        pale, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    )
    result = _select(_regions(pale, height, width), fallback)
    if result["detected"] or result["reason"] == "ambiguous_pale_regions":
        return result
    dx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.clip(cv2.magnitude(dx, dy), 0, 255).astype(np.uint8)
    high, _ = cv2.threshold(magnitude, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    high = max(20.0, high)
    image_edges = cv2.Canny(gray, 0.4 * high, high, L2gradient=True)
    barrier = cv2.dilate(image_edges, np.ones((3, 3), np.uint8))
    separated = pale & (barrier == 0).astype(np.uint8)
    refined = _select(_regions(separated, height, width, restore=pale), fallback)
    if refined["detected"]:
        refined["candidate_source"] = "image_edges"
        return refined
    if refined["reason"] == "ambiguous_pale_regions":
        return refined
    return result


@dataclass
class RecognitionResult:
    """One processed frame; absent recognition is distinct from a closed fist."""

    image: np.ndarray
    recognition: Optional[dict]
    timings_ms: dict[str, float]
    preprocess: dict
    flags: dict

    @property
    def finger_states(self):
        if self.recognition is None:
            return None
        return tuple(bool(value) for value in self.recognition["finger_states"])


def recognize(frame: np.ndarray) -> RecognitionResult:
    """Process one BGR uint8 frame, independent of every earlier frame."""
    if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8:
        raise ValueError("Expected a BGR uint8 image")
    if frame.ndim != 3 or frame.shape[2] != 3 or not frame.size:
        raise ValueError("Expected a nonempty three-channel BGR image")
    started = perf_counter()
    height, width = frame.shape[:2]
    scale = min(1.0, VIDEO_MAX_LONG_SIDE / float(max(height, width)))
    if scale < 1.0:
        image = cv2.resize(
            frame,
            (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    else:
        image = frame.copy()
    resized = perf_counter()
    prepared, preprocessing = prepare_hand_frame(image)
    oriented = perf_counter()
    state = SimpleNamespace(
        flags={"preprocess": preprocessing, "input_frame": image, "still_image": False},
        raw_frame=prepared,
    )
    segmented = segment_skin(prepared, state)
    thresholded = perf_counter()
    cleaned = clean_mask(segmented, state)
    cleaned_at = perf_counter()
    reconstructed = reconstruct_palm(cleaned, state)
    rebuilt = perf_counter()
    result = _recognize(reconstructed[:, :, 0].copy(), state)
    finished = perf_counter()
    timings = {
        "resize": (resized - started) * 1000.0,
        "board_and_orientation": (oriented - resized) * 1000.0,
        "segmentation": (thresholded - oriented) * 1000.0,
        "cleanup": (cleaned_at - thresholded) * 1000.0,
        "palm_reconstruction": (rebuilt - cleaned_at) * 1000.0,
        "finger_recognition": (finished - rebuilt) * 1000.0,
        "total": (finished - started) * 1000.0,
    }
    state.flags["recognize"] = result
    state.flags["roi_enabled"] = roi_enabled_now(state)
    return RecognitionResult(image, result, timings, preprocessing, state.flags)


def render_result(result: RecognitionResult, detailed: bool = False) -> np.ndarray:
    """Draw cached measurements in original image coordinates; never infer again."""
    matrix = np.linalg.inv(result.preprocess["input_to_prepared"])
    drawing = _transform_recognition_for_display(result.recognition, matrix)
    canvas = _draw_recognition(result.image.copy(), drawing, detailed=detailed)
    if result.flags["roi_enabled"]:
        source = result.flags["segment_source_bgr"]
        x0, y0, x1, y1 = roi_rect_pixels(source.shape[:2])
        corners = np.array([[x0, y0], [x1 - 1, y0], [x1 - 1, y1 - 1], [x0, y1 - 1]])
        polygon = np.rint(_transform_display_points(corners, matrix)).astype(np.int32)
        cv2.polylines(canvas, [polygon], True, COLOR_BOUNDARY, 2, cv2.LINE_AA)
    if detailed:
        draw_text_lines(
            canvas,
            [("整帧显示：裁板和转正仅用于内部识别", 18, COLOR_TEXT)],
            (12, max(24, canvas.shape[0] - 60)),
            panel=True,
        )
    return canvas
