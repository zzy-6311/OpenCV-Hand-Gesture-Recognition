"""Finger identity from local contour axes and transverse palm position.

No image names, truth labels, learned models, or previous frames are consumed.
Identity and extension are separate: observed curled knuckles can occupy a
finger slot while their extension bits stay false. Complete independent caps
use silhouette order; incomplete evidence falls back to transverse position
with Q10 integers and at most 16 subsets. The cap collector also uses contour
curvature; its floating-point reference needs a separate hardware design.

Feature extraction is a NumPy reference implementation, not an FPGA design.
The upstream mask, contour, palm center, axis and radius must already exist.
"""
from dataclasses import asdict, dataclass
from itertools import combinations

import numpy as np


@dataclass(frozen=True)
class GeometryConfig:
    root_y_r: float = -0.7
    # Measure width across the palm body, behind its center. Distal rows can
    # include splayed finger branches and shift every normalized finger root.
    # The four-finger projection line remains root_y_r; it serves a different
    # purpose from these three transverse width measurements.
    scan_rows_r: tuple = (0.0, 0.1, 0.2)
    scan_limit_r: float = 3.5
    scan_step_r: float = 0.025
    contour_limit_r: float = 3.0
    contour_step_r: float = 0.05
    # Arc-length window below the fingertip, in palm-radius units. The cap
    # and the widening into the palm are excluded from the shaft direction.
    axis_start_r: float = 0.20
    axis_end_r: float = 0.70
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
    # The fixed local shaft window also distinguishes a rounded slender
    # finger from a progressively widening needle/shadow. Do not truncate
    # this window at an early base event: that would hide the widening.
    shaft_widths = widths[(steps >= config.axis_start_r - 1e-9) &
                         (steps <= config.axis_end_r + 1e-9)]
    shaft_width_ratio = (float(shaft_widths.max() / shaft_widths.min())
                         if len(shaft_widths) >= 2 and np.isfinite(shaft_widths).all()
                         and shaft_widths.min() > 0 else float("inf"))
    base_candidates = np.flatnonzero((steps >= config.width_end_r) &
                                    (widths > config.base_width_factor * width))
    base_index = int(base_candidates[0]) if len(base_candidates) else len(steps)-1
    base = mid[base_index]
    length = float(np.linalg.norm(tip - base) / radius)
    # Estimate direction from every centerline point in the local straight
    # shaft, instead of subtracting two contour samples. Pixel-scale changes
    # to either endpoint otherwise rotate the extrapolated root appreciably.
    # Regress both coordinates against arc length; this is a pair of bounded
    # sums and needs no eigenvector, inverse matrix, or image-specific model.
    axis_samples = ((steps >= config.axis_start_r - 1e-9) &
                    (steps <= min(steps[base_index], config.axis_end_r) + 1e-9))
    sample_steps = steps[axis_samples]
    sample_mid = mid[axis_samples]
    if len(sample_steps) >= 2:
        centered_steps = sample_steps - sample_steps.mean()
        denominator = float(centered_steps @ centered_steps)
        slope = (sample_mid * centered_steps[:, None]).sum(axis=0) / max(1e-9, denominator)
        fitted = sample_mid.mean(axis=0) + centered_steps[:, None] * slope
        residual_r = float(np.sqrt(np.mean(np.sum((sample_mid-fitted)**2, axis=1))) / radius)
        axis = -slope
        # A least-squares line passes through the sample centroid. The cap's
        # extreme contour point need not lie on that line, especially with a
        # rounded or asymmetric fingertip; using it would translate the fit.
        axis_origin = sample_mid.mean(axis=0)
    else:
        # A degenerate configuration is diagnosed through the existing
        # horizontal/invalid-axis fallback, not fabricated finger states.
        axis = np.zeros(2, dtype=np.float64)
        residual_r = None
        axis_origin = np.asarray(tip, dtype=np.float64)
    axis /= max(1e-9, float(np.linalg.norm(axis)))
    right = np.array([-up[1], up[0]])
    tip_local = np.array([(tip-center) @ right, -(tip-center) @ up]) / radius
    origin_local = np.array([(axis_origin-center) @ right,
                             -(axis_origin-center) @ up]) / radius
    axis_local = np.array([axis @ right, -axis @ up])
    root_x = (float(origin_local[0] + (config.root_y_r-origin_local[1]) *
                    axis_local[0] / axis_local[1]) if abs(axis_local[1]) > .15 else None)
    aspect = length / max(.01, width)
    # An early contour-width event can be caused by a rough cap, rather than
    # the real finger base. Record a suspiciously short extent, but do not use
    # this same, potentially biased measurement as a hard extension veto.
    # Keep the established broad-knuckle criterion until independent shaft
    # evidence can distinguish a short bump from a rough-tipped long finger.
    has_branch_extent = bool(length > width)
    broad_curled = bool(width > config.curled_width_r and aspect < config.curled_aspect_max)
    # The residual describes local line fit only. It is not a calibrated
    # probability that the assigned finger identity is correct.
    return {"point": tip.tolist(), "base_point": base.tolist(),
            "tip_local": tip_local.tolist(), "axis_local": axis_local.tolist(),
            "axis_origin": axis_origin.tolist(), "axis_origin_local": origin_local.tolist(),
            "root_x_r": root_x, "width_r": width, "length_r": length,
            "shaft_width_ratio": shaft_width_ratio,
            "axis_fit_samples": int(len(sample_steps)), "axis_fit_residual_r": residual_r,
            "axis_fit_span_r": float(np.ptp(sample_steps)) if len(sample_steps) else 0.0,
            "base_detected": bool(len(base_candidates)),
            "aspect": aspect,
            "has_branch_extent": has_branch_extent,
            "extension_warning": None if has_branch_extent else "short_contour_extent",
            "extension_rejection_reason": "broad_curled" if broad_curled else None,
            "extended": not broad_curled}


def shares_extended_branch(cap, observed):
    """Reject a near-palm cap already covered by an observed finger shaft.

    Use the measured axis line and width; a line from fingertip to palm center
    is generally different when fingers spread. Missing axis measurements use
    the conservative radial fallback. This veto marks ambiguous overlap, not
    a proof that two observed points belong to the same anatomical finger.
    """
    cap_point = np.asarray(cap['tip_local'], dtype=float)
    for finger in observed:
        if not finger.get('extended') or finger.get('slot') == 0:
            continue
        tip = np.asarray(finger['tip_local'], dtype=float)
        axis = np.asarray(finger.get('axis_local', []), dtype=float)
        origin = np.asarray(finger.get('axis_origin_local', []), dtype=float)
        reliable = axis.shape == (2,) and origin.shape == (2,) and np.isfinite(axis).all() and np.isfinite(origin).all() and (np.linalg.norm(axis) > 1e-09) and (finger.get('axis_fit_samples', 0) >= 2) and (finger.get('axis_fit_span_r', 0) > 0)
        if reliable:
            direction = axis / np.linalg.norm(axis)
            reach = float(tip @ direction)
            if reach <= 0:
                reliable = False
        if not reliable:
            reach = float(np.linalg.norm(tip))
            if reach <= 0:
                continue
            direction = tip / reach
            origin = np.zeros(2, dtype=float)
        along = float(cap_point @ direction)
        if not 0 < along < reach:
            continue
        delta = cap_point - origin
        across = abs(float(direction[0] * delta[1] - direction[1] * delta[0]))
        if across <= 0.5 * float(finger['width_r']):
            return True
    return False


def curvature_caps(contour, center, up, radius, config=DEFAULT_CONFIG, minimum_turn_ratio=0.1):
    """Measure one closed-cap candidate per signed convex contour arc.

    Canonical hand-local traversal fixes the sampling phase. Arc samples are
    smoothed at half the existing short-shaft support, then each continuous
    positive-curvature run contributes its cumulative-turn midpoint. Its own
    inflection shoulders define the chord: a fixed-width window can cross an
    adjacent finger and cancel the real cap. This is a contour pass, with no
    image-size distance transform or per-photo inputs.
    """
    if contour is None or not np.isfinite(radius) or radius <= 0:
        return []
    pts = np.asarray(contour).reshape(-1, 2).astype(float)
    if len(pts) < 8 or not np.isfinite(pts).all():
        return []
    if not np.isfinite(up).all() or np.linalg.norm(up) < 1e-09:
        return []
    center = np.asarray(center, float)
    up = np.asarray(up, float)
    up = up / np.linalg.norm(up)
    right = np.array([-up[1], up[0]])
    yy = -(pts - center) @ up
    xx = (pts - center) @ right
    top = np.flatnonzero(yy <= yy.min() + 1e-07 * radius)
    start = int(top[np.argmin(xx[top])])
    pts = np.roll(pts, -start, axis=0)
    signed_area = np.sum(pts[:, 0] * np.roll(pts[:, 1], -1) - pts[:, 1] * np.roll(pts[:, 0], -1))
    if signed_area > 0:
        pts = np.r_[pts[:1], pts[:0:-1]]
    closed = np.vstack((pts, pts[0]))
    arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(closed, axis=0), axis=1))]
    if arc[-1] <= 0:
        return []
    n = max(8, round(arc[-1] / (config.contour_step_r * radius)))
    at = np.arange(n) * arc[-1] / n
    pp = np.column_stack([np.interp(at, arc, closed[:, j]) for j in (0, 1)])
    sigma = config.axis_start_r / (2 * config.contour_step_r)
    offsets = np.arange(-int(np.ceil(3 * sigma)), int(np.ceil(3 * sigma)) + 1)
    weights = np.exp(-0.5 * (offsets / sigma) ** 2)
    weights /= weights.sum()
    smoothed = sum((w * np.roll(pp, int(j), axis=0) for w, j in zip(weights, offsets)))
    k = max(1, round(config.axis_start_r / config.contour_step_r))
    a = smoothed - np.roll(smoothed, k, axis=0)
    b = np.roll(smoothed, -k, axis=0) - smoothed
    area = float(np.sum(pts[:, 0] * np.roll(pts[:, 1], -1) - pts[:, 1] * np.roll(pts[:, 0], -1)))
    winding = 1 if area >= 0 else -1
    turn = winding * np.arctan2(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0], (a * b).sum(1))
    positive = turn > 1e-09
    keep = np.zeros(n, dtype=bool)
    starts = np.flatnonzero(positive & ~np.roll(positive, 1))
    if positive.all():
        starts = np.array([0])
    cap_shoulders = {}
    for start in starts:
        ids = []
        j = int(start)
        while positive[j] and len(ids) < n:
            ids.append(j)
            j = (j + 1) % n
        bend = np.cumsum(turn[ids])
        cap_index = ids[int(np.searchsorted(bend, 0.5 * bend[-1]))]
        keep[cap_index] = True
        cap_shoulders[cap_index] = ((ids[0] - 1) % n, (ids[-1] + 1) % n)
    center = np.asarray(center, float)
    up = np.asarray(up, float)
    up = up / np.linalg.norm(up)
    right = np.array([-up[1], up[0]])
    local = np.column_stack(((smoothed - center) @ right, -(smoothed - center) @ up)) / radius
    keep &= (local[:, 1] < 0) & (np.linalg.norm(local, axis=1) > 1)
    caps = []
    for i in np.flatnonzero(keep):
        left, right_index = cap_shoulders[int(i)]
        aa = smoothed[left]
        bb = smoothed[right_index]
        chord = bb - aa
        offset = smoothed[i] - aa
        length = np.linalg.norm(chord)
        depth = -winding * (chord[0] * offset[1] - chord[1] * offset[0]) / max(1e-09, length)
        if depth < minimum_turn_ratio * length:
            continue
        point = pts[np.argmin(np.sum((pts - smoothed[i]) ** 2, axis=1))]
        f = _measure_tip(contour, point, center, up, radius, config)
        f['curvature_turn'] = float(turn[i])
        f['cap_support_depth_r'] = float(depth / radius)
        f['cap_support_chord_r'] = float(length / radius)
        caps.append(f)
    return caps


def _curvature_supports(measured, caps, radius, config, *, independent=False):
    """Collect closed positions without modifying the shared raw cap measurements."""
    others = [feature for feature in measured["features"] if feature["slot"] != 0]
    index = next((feature for feature in others if feature["slot"] == 1), None)
    min_width = min(feature["width_r"] for feature in others)
    extended = [feature for feature in others if feature["extended"]]
    bracket = None
    if not independent and len(extended) >= 2:
        angles = [np.arctan2(f["tip_local"][0], -f["tip_local"][1]) for f in extended]
        bracket = min(angles), max(angles)
    supports = []
    for cap in caps:
        feature = dict(cap)
        x, y = feature["tip_local"]
        if y >= 0 or x > config.thumb_slope * -y:
            continue
        if bracket is not None:
            angle = np.arctan2(x, -y)
            if not bracket[0] <= angle <= bracket[1]:
                # Outside the observed fingers, a tiny convex ripple on the
                # palm/shaft edge is not an independent folded digit. Require
                # the same shoulder support used by the independent pass.
                if (feature.get("cap_support_chord_r", 0) < 0.5 * feature["width_r"] or
                        (x > 0 and y >= -1.0)):
                    continue
        broad = (feature["width_r"] > config.curled_width_r and
                 feature["aspect"] < config.curled_aspect_max)
        compact = (feature["aspect"] < config.curled_aspect_max and
                   feature["width_r"] >= min_width)
        if not (broad or compact):
            continue
        if independent:
            if (feature.get("cap_support_chord_r") is None or
                    feature.get("cap_support_depth_r") is None):
                continue
            # A very short convex arc can sit on a much wider palm/shaft edge.
            # Require shoulders to span at least half the measured local width.
            if feature["cap_support_chord_r"] < 0.5 * feature["width_r"]:
                continue
            # On the thumb side, a folded digit must project past the distal
            # tangent to the inscribed palm circle. A lateral palm cap cannot
            # supply the missing index position merely because its x is large.
            if x > 0 and y >= -1.0:
                continue
            # The independent short-branch test belongs to thumb recovery,
            # not to closed-finger occupancy.
            if (x > 0 and feature["length_r"] > feature["width_r"] and
                    feature["cap_support_depth_r"] > feature["width_r"]):
                continue
        elif (index is not None and x > 0 and x > index["tip_local"][0] and
              (feature["root_x_r"] is None or feature["root_x_r"] > index["root_x_r"])):
            continue
        if shares_extended_branch(feature, others):
            continue
        if any(np.linalg.norm(np.asarray(feature["point"]) - other["point"]) <
               0.5 * (feature["width_r"] + other["width_r"]) * radius
               for other in measured["features"]):
            continue
        supports.append(feature)
    chosen = []
    for feature in sorted(supports, key=lambda item: -item["curvature_turn"]):
        if any(np.linalg.norm(np.asarray(feature["point"]) - other["point"]) <
               0.5 * max(feature["width_r"], other["width_r"]) * radius
               for other in chosen):
            continue
        chosen.append(feature)
    return chosen


def _refine_with_curvature_caps_base(measured, contour, center, up, radius,
                               config=DEFAULT_CONFIG, *, caps=None):
    """Complete four contour positions, then fall back to roots if ambiguous.

    An already complete position set is preserved. If ordinary caps cannot
    complete four positions, require independent shoulder support and the
    distal palm geometry instead of trusting the preliminary index identity.
    Competing or insufficient positions still leave root assignment intact.
    Both passes reuse the same contour measurements; no image labels enter.
    """
    if measured is None or measured.get("topology_used"):
        return measured
    others = [feature for feature in measured["features"] if feature["slot"] != 0]
    if not 0 < len(others) < 4:
        return measured
    if caps is None:
        caps = curvature_caps(contour, center, up, radius, config)
    chosen = _curvature_supports(measured, caps, radius, config)
    measured["curvature_support_candidates"] = chosen
    method = "complete_digit_topology_curvature"
    if len(others) + len(chosen) != 4:
        independent = _curvature_supports(measured, caps, radius, config, independent=True)
        measured["independent_support_candidates"] = independent
        if len(others) + len(independent) != 4:
            return measured
        chosen = independent
        method = "complete_digit_topology_supported_caps"
        measured["topology_support_reason"] = "ordinary_positions_incomplete_or_competing"
    for feature in chosen:
        feature.update(extended=False, extension_rejection_reason="curvature_closed_support",
                       tip_index=None, root_q10=None, root_point=None, closed_support=True)
    return _complete_digit_topology(measured, chosen, method)


def _complete_digit_topology(measured, supports, method="complete_digit_topology"):
    """Use silhouette order only when all four non-thumb positions are observed.

    Missing or competing caps leave the reference-line assignment unchanged.
    The extension bits of existing candidates are never modified here.
    """
    others = [f for f in measured["features"] if f["slot"] != 0]
    thumb = [f for f in measured["features"] if f["slot"] == 0]
    if len(others) + len(supports) != 4:
        measured["topology_used"] = False
        return measured
    ordered = sorted(others + supports,
                     key=lambda f: np.arctan2(f["tip_local"][0], -f["tip_local"][1]),
                     reverse=True)
    for slot, feature in enumerate(ordered, 1):
        feature["slot"] = slot
    # Q20 costs belong to the fallback root assignment, not this ordering.
    measured["root_assignment"] = {key: measured[key] for key in
                                  ("candidates", "score_margin_q20", "total_q20")}
    measured.update(states=[bool(thumb)] + [bool(f["extended"]) for f in ordered],
                    features=thumb + ordered, candidates=[], score_margin_q20=None,
                    total_q20=None, method=method, topology_used=True)
    return measured


def refine_with_closed_caps(measured, contour, palm_center, up, radius,
                            caps, config=DEFAULT_CONFIG):
    """Retain short knuckles as closed positions, not additional fingertips.

    Width is compared with fingers measured in this frame. No absolute pixel
    dimensions or per-image values enter the topology gate.
    """
    if measured is None:
        return measured
    others = [f for f in measured["features"] if f["slot"] != 0]
    thumb = [f for f in measured["features"] if f["slot"] == 0]
    index = next((f for f in others if f["slot"] == 1), None)
    min_width = min((f["width_r"] for f in others), default=0.0)
    supports = []
    for cap in caps:
        if not cap.get("closed_support"):
            continue
        feature = _measure_tip(contour, np.asarray(cap["point"]), palm_center,
                               up, radius, config)
        # This gate admits new short caps. It does not veto an existing finger:
        # a rough fingertip can bias its base measurement towards the cap.
        if feature["length_r"] >= feature["width_r"] or feature["width_r"] < min_width:
            continue
        x, y = feature["tip_local"]
        if y >= 0 or x > config.thumb_slope * -y:
            continue
        if (index is not None and x > index["tip_local"][0] and
                (feature["root_x_r"] is None or feature["root_x_r"] > index["root_x_r"])):
            continue
        if thumb and abs(x - thumb[0]["tip_local"][0]) < feature["width_r"]:
            continue
        if shares_extended_branch(feature, others):
            continue
        feature.update(extended=False, extension_rejection_reason="closed_contour_support",
                       tip_index=None, root_q10=None, root_point=None, closed_support=True)
        supports.append(feature)
    measured["closed_support_candidates"] = supports
    return _complete_digit_topology(measured, supports)


def _is_lateral_thumb_branch(feature, center, up, radius, config):
    """Recognize an upright thumb from its observed low, lateral attachment.

    Angle alone misses a thumb held close to the index direction. Require a
    real extended branch with a detected base below the four-finger root line
    and on the lateral palm. A tip beyond one palm diameter is excluded: long
    fingers can also produce a low apparent base. All distances use the palm
    radius, and the measurements already exist; no extra image scan is needed.
    """
    if (not feature.get("extended") or not feature.get("has_branch_extent") or
            not feature.get("base_detected")):
        return False
    x, y = feature["tip_local"]
    if (not np.isfinite([x, y]).all() or not -2.0 < y < 0 or
            x <= config.five_candidate_thumb_slope * -y):
        return False
    right = np.array([-up[1], up[0]])
    delta = np.asarray(feature["base_point"], dtype=float) - center
    base_x, base_y = float(delta @ right / radius), float(-delta @ up / radius)
    return bool(np.isfinite([base_x, base_y]).all() and
                base_y > config.root_y_r and base_x > abs(base_y))


def recognize_fingers(mask, contour, palm_center, up, radius, tips, config=DEFAULT_CONFIG,
                      *, closed_candidates=None):
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
    # The five-finger relaxation needs four assignable non-thumb candidates.
    # A short invalid-axis bump that will be discarded must not manufacture
    # that evidence. Keep `first` eligible as the prospective thumb: a real
    # lateral thumb need not intersect the four-finger reference line.
    assignable_count = int(first is not None)
    for f in features:
        if f is first or f["root_x_r"] is None:
            continue
        q = int(np.floor((f["root_x_r"]-lo) / (hi-lo) * Q10 + 0.5))
        if ROOT_MIN_Q10 <= q <= ROOT_MAX_Q10:
            assignable_count += 1
    thumb = None
    if first is not None:
        x, y = first["tip_local"]
        if y < 0 and (x > config.thumb_slope * -y or
                      (assignable_count == 5 and x > config.five_candidate_thumb_slope * -y) or
                      _is_lateral_thumb_branch(first, center, up, radius, config)):
            thumb = first
    retained = []
    discarded = []
    for f in features:
        root = f["root_x_r"]
        if root is None:
            if f is not thumb:
                if not f["extended"]:
                    f["discard_reason"] = "horizontal_finger_axis"
                    discarded.append(f)
                    continue
                return None, "horizontal_finger_axis"
            f["root_q10"] = None
            f["root_point"] = None
            retained.append(f)
            continue
        q = int(np.floor((root-lo) / (hi-lo) * Q10 + 0.5))
        if not ROOT_MIN_Q10 <= q <= ROOT_MAX_Q10 and f is not thumb:
            if not f["extended"]:
                f["discard_reason"] = "root_out_of_range"
                discarded.append(f)
                continue
            return None, "root_out_of_range"
        f["root_q10"] = q
        f["root_point"] = (center + radius * (
            root * np.array([-up[1], up[0]]) - config.root_y_r * up)).tolist()
        retained.append(f)
    features = retained
    others = [f for f in features if f is not thumb]
    if len(others) > 4:
        return None, "too_many_candidates"
    others.sort(key=lambda f: -f["root_q10"])
    slots, ranked, margin = assign_slots_q10([f["root_q10"] for f in others], config.slot_positions_q10)
    states = [False] * 5
    for slot, feature in zip(slots, others):
        feature["slot"] = slot
        states[slot] = feature["extended"]
    if thumb is not None:
        thumb["slot"] = 0
        # Thumb proportions differ from the four fingers. A suspicious short
        # contour extent remains diagnostic; it cannot reliably veto a thumb
        # when a rough cap has displaced the measured contour base.
        thumb["extended"] = True
        thumb["extension_rejection_reason"] = None
        states[0] = True
    measured = {"states": states, "features": features, "palm_bounds_r": [lo, hi],
            "discarded_features": discarded,
            "candidates": ranked, "score_margin_q20": margin,
            "total_q20": ranked[0]["cost_q20"], "method": "root_geometry_q10"}
    if closed_candidates is not None:
        measured = refine_with_closed_caps(measured, contour, center, up, radius,
                                          closed_candidates, config)
        non_thumb_count = sum(feature["slot"] != 0 for feature in measured["features"])
        need_caps = ((0 < non_thumb_count < 4 and not measured.get("topology_used")) or
                     (non_thumb_count == 4 and not measured["states"][0]))
        caps = curvature_caps(contour, center, up, radius, config) if need_caps else []
        measured = refine_with_curvature_caps(measured, contour, center, up, radius, config,
                                             caps=caps)
        measured = refine_with_outer_thumb(measured, contour, center, up, radius, config,
                                          caps=caps)
        measured = refine_thumb_from_five_positions(measured, config)
    return measured, None


def refine_with_outer_thumb(measured, contour, center, up, radius,
                            config=DEFAULT_CONFIG, *, caps=None):
    """Recover an independent short outer branch after four fingers are accounted for.

    Requiring all four non-thumb contour positions avoids using absent fingers
    as evidence for a thumb. The candidate's own convex-arc depth must exceed
    its shaft width; this rejects shallow palm roughness in the current tests.
    This contour-only reference remains a hypothesis for a visible branch,
    not recovery of an occluded anatomical thumb.
    """
    if measured is None or measured["states"][0]:
        return measured
    observed = measured["features"]
    if sorted(feature["slot"] for feature in observed) != [1, 2, 3, 4]:
        return measured
    index = next(feature for feature in observed if feature["slot"] == 1)
    candidates = []
    if caps is None:
        caps = curvature_caps(contour, center, up, radius, config)
    for candidate in caps:
        cap = dict(candidate)
        x, y = cap["tip_local"]
        if x <= max(0, index["tip_local"][0]):
            continue
        if cap.get("cap_support_depth_r") is None:
            continue
        if cap["length_r"] <= cap["width_r"] or cap["cap_support_depth_r"] <= cap["width_r"]:
            continue
        if (cap["root_x_r"] is None or index["root_x_r"] is None or
                cap["root_x_r"] <= index["root_x_r"]):
            continue
        if shares_extended_branch(cap, observed):
            continue
        if any(np.linalg.norm(np.asarray(cap["point"]) - feature["point"]) <
               0.5 * (cap["width_r"] + feature["width_r"]) * radius
               for feature in observed):
            continue
        candidates.append(cap)
    if len(candidates) != 1:
        return measured
    cap = candidates[0]
    cap.update(slot=0, extended=True, extension_rejection_reason=None,
               tip_index=None, root_q10=None, root_point=None, thumb_recovered=True)
    measured["states"][0] = True
    measured["features"].append(cap)
    measured["thumb_recovered"] = "four_slots_and_outer_branch"
    return measured


def refine_with_curvature_caps(measured, contour, center, up, radius, config=DEFAULT_CONFIG, *, caps=None):
    if measured is None:
        return measured
    if measured.get("topology_used"):
        return measured
    observed = [dict(feature) for feature in measured["features"] if feature["slot"] != 0]
    thumbs = [dict(feature) for feature in measured["features"] if feature["slot"] == 0]
    if not 0 < len(observed) < 4:
        return measured
    raw_caps = caps if caps is not None else curvature_caps(contour, center, up, radius, config)
    baseline = dict(measured)
    baseline["features"] = [dict(feature) for feature in measured["features"]]
    baseline = _refine_with_curvature_caps_base(baseline, contour, center, up, radius, config, caps=raw_caps)
    independent = _curvature_supports(measured, raw_caps, radius, config, independent=True)
    if baseline.get("topology_used"):
        # The old ordinary pass can count a tiny convex arc on the palm edge.
        # Replace it only if the existing stricter shoulder test independently
        # supplies a complete set; otherwise preserve the existing decision.
        ordinary = baseline.get("curvature_support_candidates", [])
        weak = any(cap.get("cap_support_chord_r", 0) < .5 * cap["width_r"] for cap in ordinary)
        if (baseline.get("method") == "complete_digit_topology_curvature" and weak and
                len(observed) + len(independent) == 4):
            supports = []
            for cap in independent:
                cap = dict(cap)
                cap.update(extended=False, extension_rejection_reason="independent_shoulder_support",
                           tip_index=None, root_q10=None, root_point=None, closed_support=True)
                supports.append(cap)
            measured["independent_support_candidates"] = supports
            measured["topology_support_reason"] = "ordinary_cap_lacks_independent_shoulders"
            return _complete_digit_topology(measured, supports,
                         method="complete_digit_topology_curvature")
        return baseline
    if not independent:
        return baseline
    lo, hi = measured["palm_bounds_r"]
    supported = []
    for feature in independent:
        root = feature["root_x_r"]
        # A closed-cap shaft must reach the observed palm cross-section.
        # Unlike fingertip polar order, this is comparable to extended roots.
        if root is None or not np.isfinite(root) or not lo <= root <= hi:
            continue
        feature = dict(feature)
        feature.update(extended=False, extension_rejection_reason="independent_closed_root_support",
                       tip_index=None, root_q10=int(np.floor((root-lo)/(hi-lo)*Q10 + .5)),
                       root_point=None, closed_support=True)
        supported.append(feature)
    measured["joint_closed_candidates"] = supported
    if len(observed) + len(supported) > 4:
        measured["joint_rejection"] = "competing_observed_positions"
        return baseline
    # Incomplete closed caps constrain identity only when they bracket the
    # observed fingers. One-sided caps alone cannot anchor both slot ends.
    angles = lambda fs: [float(np.arctan2(f["tip_local"][0], -f["tip_local"][1])) for f in fs]
    observed_angles, closed_angles = angles(observed), angles(supported)
    if (len(supported) < 2 or min(closed_angles) >= min(observed_angles) or
            max(closed_angles) <= max(observed_angles)):
        return baseline
    combined = sorted(observed + supported, key=lambda item: -item["root_q10"])
    slots, ranked, margin = assign_slots_q10([feature["root_q10"] for feature in combined],
                                                       config.slot_positions_q10)
    states = [bool(thumbs), False, False, False, False]
    for slot, feature in zip(slots, combined):
        feature["slot"] = slot
        states[slot] = bool(feature["extended"])
    measured["initial_root_assignment"] = {key: measured.get(key) for key in
                                           ("candidates", "score_margin_q20", "total_q20")}
    measured.update(features=thumbs + combined, states=states, candidates=ranked,
                    score_margin_q20=margin, total_q20=ranked[0]["cost_q20"],
                    method="root_geometry_q10", topology_used=False,
                    identity_refinement="bracketed_closed_root_support")
    return measured



def refine_thumb_from_five_positions(measured, config=DEFAULT_CONFIG):
    """Recover a steep outer thumb only with five independent contour positions.

    The independent caps must come from _curvature_supports(...,
    independent=True). This helper reuses those measurements; it performs no
    extra contour scan. Missing/competing evidence leaves measured unchanged.
    On success, four non-thumb positions are ordered by the existing topology
    completion. Five observed positions remain a silhouette-based hypothesis,
    not a guarantee against segmentation artifacts or anatomical overlap.
    """
    if measured is None or measured["states"][0] or measured.get("topology_used"):
        return measured
    observed = measured["features"]
    supports = measured.get("independent_support_candidates") or ()
    if not observed or not supports or len(observed) + len(supports) != 5:
        return measured

    first = max(observed, key=lambda f: f["tip_local"][0] /
                max(1e-6, -f["tip_local"][1]))
    x, y = first["tip_local"]
    if (y >= 0 or x <= config.five_candidate_thumb_slope * -y or
            not first["extended"] or not first["has_branch_extent"]):
        return measured
    for cap in supports:
        cap_x, cap_y = cap["tip_local"]
        # Lateral position and angular order must both place the thumb outside
        # every closed digit. Cross multiplication reuses the topology order.
        if (cap_y >= 0 or x <= cap_x or
                x * -cap_y <= cap_x * -y):
            return measured

    # All admission checks passed. Copy only the small cap dictionaries so
    # their raw diagnostic measurements remain intact; the output is updated
    # in place like the other refinement helpers.
    closed = [dict(cap) for cap in supports]
    for cap in closed:
        cap.update(extended=False, extension_rejection_reason="curvature_closed_support",
                   tip_index=None, root_q10=None, root_point=None, closed_support=True)
    first["slot"] = 0
    result = _complete_digit_topology(measured, closed, "five_observed_positions_thumb")
    result["thumb_recovered"] = "five_observed_positions"
    return result
