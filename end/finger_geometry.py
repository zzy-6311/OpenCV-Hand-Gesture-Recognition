"""Finger identity from local contour axes and transverse palm position.

No image names, truth labels, learned models, or previous frames are consumed.
Identity and extension are separate: a broad, short candidate can occupy a
finger slot while its extension bit stays false. The final ordered assignment
uses Q10 integers and at most 16 subsets, suitable for a small fixed-size port.

Feature extraction is a NumPy reference implementation, not an FPGA design.
The upstream mask, contour, palm center, axis and radius must already exist.
"""
from dataclasses import asdict, dataclass
from itertools import combinations

import numpy as np


@dataclass(frozen=True)
class GeometryConfig:
    root_y_r: float = -0.7
    scan_rows_r: tuple = (-0.8, -0.7, -0.6)
    scan_limit_r: float = 3.5
    scan_step_r: float = 0.025
    contour_limit_r: float = 3.0
    contour_step_r: float = 0.05
    width_start_r: float = 0.35
    width_end_r: float = 0.65
    base_width_factor: float = 1.6
    curled_width_r: float = 0.50
    curled_aspect_max: float = 3.0
    thumb_slope: float = 0.84
    five_candidate_thumb_slope: float = 0.50
    # I, M, R, P; transverse palm position from the little-finger side.
    slot_positions_q10: tuple = (717, 461, 205, 51)


DEFAULT_CONFIG = GeometryConfig()
Q10 = 1024
ROOT_MIN_Q10 = -1024
ROOT_MAX_Q10 = 2048


def config_dict(config=DEFAULT_CONFIG):
    return asdict(config)


def assign_slots_q10(roots, positions=DEFAULT_CONFIG.slot_positions_q10):
    """Sorted descending Q10 observations -> distinct ordered I/M/R/P slots.

    Cost is a sum of squared integer differences (Q20). With root inputs in
    [-1024, 2048], the default reference positions fit safely in signed int32.
    No angle search, scale search, or preference for common gestures is used.
    """
    roots = [int(x) for x in roots]
    positions = [int(x) for x in positions]
    if len(roots) > 4 or len(positions) != 4:
        raise ValueError("Expected up to four observations and four reference positions")
    if any(x < ROOT_MIN_Q10 or x > ROOT_MAX_Q10 for x in roots + positions):
        raise ValueError("Q10 position outside bounded arithmetic range")
    if roots != sorted(roots, reverse=True):
        raise ValueError("Root positions must be ordered from index to little finger")
    ranked = []
    for slots in combinations(range(1, 5), len(roots)):
        cost = sum((root - positions[slot - 1]) ** 2 for root, slot in zip(roots, slots))
        ranked.append({"slots": list(slots), "cost_q20": cost})
    ranked.sort(key=lambda item: item["cost_q20"])
    margin = ranked[1]["cost_q20"] - ranked[0]["cost_q20"] if len(ranked) > 1 else None
    return ranked[0]["slots"], ranked, margin


def _palm_bounds(mask, center, up, radius, config):
    right = np.array([-up[1], up[0]])
    xs = np.arange(-config.scan_limit_r, config.scan_limit_r + 1e-9, config.scan_step_r)
    bounds = []
    for y in config.scan_rows_r:
        xy = np.rint(center + radius * (xs[:, None] * right - y * up)).astype(np.int32)
        inside = ((xy[:, 0] >= 0) & (xy[:, 0] < mask.shape[1]) &
                  (xy[:, 1] >= 0) & (xy[:, 1] < mask.shape[0]))
        hits = inside & (mask[np.clip(xy[:, 1], 0, mask.shape[0]-1),
                              np.clip(xy[:, 0], 0, mask.shape[1]-1)] > 0)
        if hits.any():
            bounds.append([float(xs[hits].min()), float(xs[hits].max())])
    if len(bounds) < 2:
        return None
    lo, hi = np.median(bounds, axis=0)
    return (float(lo), float(hi)) if hi - lo > 0.5 else None


def _measure_tip(contour, tip, center, up, radius, config):
    pts = contour.reshape(-1, 2).astype(np.float64)
    start = int(np.argmin(np.sum((pts - tip) ** 2, axis=1)))
    steps = np.arange(0.0, config.contour_limit_r + 1e-9, config.contour_step_r)
    sides = []
    for sign in (-1, 1):
        chain = pts[(start + sign * np.arange(len(pts))) % len(pts)]
        arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(chain, axis=0), axis=1))]
        sides.append(np.column_stack([
            np.interp(steps * radius, arc, chain[:, k]) for k in (0, 1)]))
    mid = (sides[0] + sides[1]) * 0.5
    widths = np.linalg.norm(sides[0] - sides[1], axis=1) / radius
    width = float(np.median(widths[(steps >= config.width_start_r) & (steps <= config.width_end_r)]))
    base_candidates = np.flatnonzero((steps >= config.width_end_r) &
                                    (widths > config.base_width_factor * width))
    base_index = int(base_candidates[0]) if len(base_candidates) else len(steps)-1
    base = mid[base_index]
    length = float(np.linalg.norm(tip - base) / radius)
    axis = mid[4] - mid[min(base_index, 14)]
    axis /= max(1e-9, float(np.linalg.norm(axis)))
    right = np.array([-up[1], up[0]])
    tip_local = np.array([(tip-center) @ right, -(tip-center) @ up]) / radius
    axis_local = np.array([axis @ right, -axis @ up])
    root_x = (float(tip_local[0] + (config.root_y_r-tip_local[1]) *
                    axis_local[0] / axis_local[1]) if abs(axis_local[1]) > .15 else None)
    aspect = length / max(.01, width)
    return {"point": tip.tolist(), "base_point": base.tolist(),
            "tip_local": tip_local.tolist(), "axis_local": axis_local.tolist(),
            "root_x_r": root_x, "width_r": width, "length_r": length,
            "aspect": aspect,
            "extended": not (width > config.curled_width_r and aspect < config.curled_aspect_max)}


def recognize_fingers(mask, contour, palm_center, up, radius, tips, config=DEFAULT_CONFIG):
    """Return (measurement, failure_reason); failure leaves fallback to caller."""
    if contour is None or len(contour) < 8 or not np.isfinite(radius) or radius <= 0:
        return None, "invalid_geometry"
    center = np.asarray(palm_center, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)
    norm = float(np.linalg.norm(up))
    if not np.isfinite(norm) or norm < 1e-9:
        return None, "invalid_axis"
    up = up / norm
    features = [_measure_tip(contour, np.asarray(t["point"], dtype=float), center, up, radius, config)
                for t in tips]
    for index, feature in enumerate(features):
        feature["tip_index"] = index
    bounds = _palm_bounds(mask, center, up, radius, config)
    if bounds is None:
        return None, "palm_scan_unavailable"
    lo, hi = bounds
    # Select the most thumbward tip without an inverse trigonometric function.
    first = max(features, key=lambda f: f["tip_local"][0] / max(1e-6, -f["tip_local"][1]), default=None)
    thumb = None
    if first is not None:
        x, y = first["tip_local"]
        if y < 0 and (x > config.thumb_slope * -y or
                      (len(features) == 5 and x > config.five_candidate_thumb_slope * -y)):
            thumb = first
    others = [f for f in features if f is not thumb]
    if len(others) > 4:
        return None, "too_many_candidates"
    for f in features:
        root = f["root_x_r"]
        if root is None:
            if f is not thumb:
                return None, "horizontal_finger_axis"
            f["root_q10"] = None
            f["root_point"] = None
            continue
        q = int(np.floor((root-lo) / (hi-lo) * Q10 + 0.5))
        if not ROOT_MIN_Q10 <= q <= ROOT_MAX_Q10 and f is not thumb:
            return None, "root_out_of_range"
        f["root_q10"] = q
        f["root_point"] = (center + radius * (
            root * np.array([-up[1], up[0]]) - config.root_y_r * up)).tolist()
    others.sort(key=lambda f: -f["root_q10"])
    slots, ranked, margin = assign_slots_q10([f["root_q10"] for f in others], config.slot_positions_q10)
    states = [False] * 5
    for slot, feature in zip(slots, others):
        feature["slot"] = slot
        states[slot] = feature["extended"]
    if thumb is not None:
        thumb["slot"] = 0
        thumb["extended"] = True
        states[0] = True
    return {"states": states, "features": features, "palm_bounds_r": [lo, hi],
            "candidates": ranked, "score_margin_q20": margin,
            "total_q20": ranked[0]["cost_q20"], "method": "root_geometry_q10"}, None
