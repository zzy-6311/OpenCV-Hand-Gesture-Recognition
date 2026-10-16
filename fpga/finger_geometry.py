"""Finger identity from local contour axes and transverse palm position.

No image names, truth labels, learned models, or previous frames are consumed.
Identity and extension are separate: observed curled knuckles can occupy a
finger slot while their extension bits stay false. Complete independent caps
use silhouette order; incomplete evidence falls back to transverse position
with Q6 integers and at most 16 subsets. The cap collector also uses contour
convex arcs, with cross/dot thresholds and maximum outward chord height.

Feature extraction rounds coordinates to 1/1 pixels and normalized
features to 1/32. Local tip contour-length sums retain Q4 precision; the separate curve
resampling arc still uses floating point. Centerline
regression uses integer centered-coordinate sums. Interpolation, square
roots, division and angle evaluation still use floating point: this is a
tested numerical reference, not a complete fixed-point RTL implementation.
The upstream mask, contour, palm center, axis and radius must already exist.
"""
from dataclasses import asdict, dataclass
from copy import deepcopy
from functools import lru_cache
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
    slot_positions_fixed: tuple = (45, 29, 13, 3)


DEFAULT_CONFIG = GeometryConfig()
ROOT_FRACTION_BITS = 6
ROOT_SCALE = 1 << ROOT_FRACTION_BITS
ROOT_MIN_FIXED = -ROOT_SCALE
ROOT_MAX_FIXED = 2 * ROOT_SCALE

# Numerical precision only; gesture thresholds stay in GeometryConfig.
_PIXEL_SCALE = 1
_NORMAL_SCALE = 32
_NORMAL_PRODUCT_SCALE = _NORMAL_SCALE * _NORMAL_SCALE
_ARC_SCALE = 16


def _quantize(value, scale):
    """Round to the tested binary grid, nearest with ties to even.

    Preserve None/nonfinite values for the existing invalid-feature paths.
    Host int64 is an accumulator container, not a proposed RTL storage width;
    no saturation or wrapping is added to the recognition decisions.
    """
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float64)
    finite = np.isfinite(arr)
    out = arr.copy()
    codes_float = np.rint(arr[finite] * scale)
    if np.any(np.abs(codes_float) >= 2 ** 62):
        raise OverflowError("Geometry quantization exceeds the host integer range")
    codes = codes_float.astype(np.int64)
    out[finite] = codes.astype(np.float64) / scale
    return float(out) if arr.ndim == 0 else out


def _contour_arc_fixed(lengths):
    """Quantize local edge lengths before the integer contour-length sum."""
    codes = np.rint(np.asarray(lengths) * _ARC_SCALE).astype(np.int64)
    accumulated = np.cumsum(codes, dtype=np.int64)
    return np.r_[0., accumulated.astype(float) / _ARC_SCALE]


@lru_cache(maxsize=64)
def _axis_fit_constants(step_codes):
    """Configuration-derived coefficients, independent of images or labels.

    c = n*w - sum(w) has exactly zero sum. It gives the same integer
    accumulator as sum((n*pixel-sum(pixel))*w), without per-tip centering.
    Default sampling uses 10 or 11 coefficients; custom configurations
    generate their own table while retaining all rounding boundaries.
    """
    steps = np.asarray(step_codes, dtype=np.float64) / _NORMAL_SCALE
    centered = _quantize(steps - steps.mean(), _NORMAL_SCALE)
    denominator = _quantize(float(centered @ centered), _NORMAL_PRODUCT_SCALE)
    weights = np.rint(centered * _NORMAL_SCALE).astype(np.int64)
    coefficients = len(steps) * weights - weights.sum(dtype=np.int64)
    coefficients.setflags(write=False)
    centered.setflags(write=False)
    divisor = len(steps) * _PIXEL_SCALE * _NORMAL_SCALE * max(1e-9, denominator)
    return coefficients, divisor, centered


def _centered_axis_slope(mid, steps, centered=None, denominator=None):
    pixel = np.rint(mid * _PIXEL_SCALE).astype(np.int64)
    step_codes = tuple(np.rint(steps * _NORMAL_SCALE).astype(np.int64))
    coefficients, divisor, _ = _axis_fit_constants(step_codes)
    accumulated = (pixel * coefficients[:, None]).sum(axis=0, dtype=np.int64)
    return accumulated.astype(float) / divisor


def config_dict(config=DEFAULT_CONFIG):
    return asdict(config)


def assign_slots_fixed(roots, positions=DEFAULT_CONFIG.slot_positions_fixed):
    """Sorted descending fixed-point observations -> distinct ordered I/M/R/P slots.

    Cost is a sum of squared integer differences, at twice ROOT_FRACTION_BITS.
    Roots and reference positions are checked within [-ROOT_SCALE, 2*ROOT_SCALE].
    No angle search, scale search, or preference for common gestures is used.
    """
    roots = [int(x) for x in roots]
    positions = [int(x) for x in positions]
    if len(roots) > 4 or len(positions) != 4:
        raise ValueError("Expected up to four observations and four reference positions")
    if any(x < ROOT_MIN_FIXED or x > ROOT_MAX_FIXED for x in roots + positions):
        raise ValueError("Fixed-point position outside bounded arithmetic range")
    if roots != sorted(roots, reverse=True):
        raise ValueError("Root positions must be ordered from index to little finger")
    ranked = []
    for slots in combinations(range(1, 5), len(roots)):
        cost = sum((root - positions[slot - 1]) ** 2 for root, slot in zip(roots, slots))
        ranked.append({"slots": list(slots), "cost_fixed_sq": cost})
    ranked.sort(key=lambda item: item["cost_fixed_sq"])
    margin = ranked[1]["cost_fixed_sq"] - ranked[0]["cost_fixed_sq"] if len(ranked) > 1 else None
    return ranked[0]["slots"], ranked, margin


def _palm_bounds(mask, center, up, radius, config):
    center = _quantize(center, _PIXEL_SCALE)
    up = _quantize(up, _NORMAL_SCALE)
    radius = _quantize(radius, _PIXEL_SCALE)
    right = np.array([-up[1], up[0]])
    xs = np.arange(-config.scan_limit_r, config.scan_limit_r + 1e-9, config.scan_step_r)
    xs = _quantize(xs, _NORMAL_SCALE)
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
    lo, hi = _quantize(np.median(bounds, axis=0), _NORMAL_SCALE)
    return (float(lo), float(hi)) if hi - lo > 0.5 else None


@lru_cache(maxsize=8)
def _tip_sampling_template(config):
    """Configuration-only samples; keep the reference floating-point values."""
    steps = np.arange(0.0, config.contour_limit_r + 1e-9, config.contour_step_r)
    width_samples = (steps >= config.width_start_r) & (steps <= config.width_end_r)
    shaft_samples = ((steps >= config.axis_start_r - 1e-9) &
                     (steps <= config.axis_end_r + 1e-9))
    for values in (steps, width_samples, shaft_samples):
        values.setflags(write=False)
    return steps, width_samples, shaft_samples


def _dense_contour_edge_lengths(points):
    """Local arc step lookup for an original unit-grid CHAIN_APPROX_NONE contour.

    Interpolated, smoothed or sparse contours must use the general norm.
    Horizontal/vertical edges use _ARC_SCALE; diagonals round to 23.
    """
    delta = np.roll(points, -1, axis=0) - points
    absolute = np.abs(delta)
    if (np.equal(absolute, np.rint(absolute)).all() and
            np.all(absolute <= 1)):
        diagonal = (absolute[:, 0] == 1) & (absolute[:, 1] == 1)
        nonzero = np.any(absolute != 0, axis=1)
        codes = np.where(diagonal, 23, np.where(nonzero, _ARC_SCALE, 0))
        return codes.astype(np.float64) / _ARC_SCALE
    return np.linalg.norm(delta, axis=1)


class TipMeasurementContext:
    """Frame-local measurements for one immutable contour revision.

    Create a new context after changing the contour; never retain it across
    frames. The source contour must not be edited in place while this context
    is in use. A different contour object automatically uses the uncached
    path. Exact input keys deliberately distinguish even tiny axis changes.

    Only raw measurements are cached, never slot/extension decisions. Every
    caller owns its returned dictionary, including its nested point lists.
    The bounded cache is a reference for a shared measurement service, not a
    requirement to instantiate one arithmetic unit per cached candidate.
    """

    def __init__(self, contour, *, max_entries=32):
        self._source_contour = contour
        self._points = contour.reshape(-1, 2).astype(np.float64)
        self._points.setflags(write=False)
        self._offsets = np.arange(len(self._points))
        self._offsets.setflags(write=False)
        # Each contour edge has the same Euclidean length in either direction.
        # Reordering these lengths preserves the original cumsum order.
        self._edge_lengths = _dense_contour_edge_lengths(self._points)
        self._edge_lengths.setflags(write=False)
        self._max_entries = max(0, int(max_entries))
        self._measurements = {}
        self.computations = 0
        self.cache_hits = 0

    @staticmethod
    def _array_key(value):
        values = np.asarray(value)
        # Preserve dtype as well as bits: float32 subtraction can otherwise
        # differ from float64 subtraction of numerically identical inputs.
        return values.dtype.str, values.shape, values.tobytes()

    def measure(self, contour, tip, center, up, radius, config=DEFAULT_CONFIG):
        if contour is not self._source_contour:
            return _measure_tip_impl(contour, tip, center, up, radius, config)
        key = (self._array_key(tip), self._array_key(center),
               self._array_key(up), self._array_key(radius), config)
        if key in self._measurements:
            self.cache_hits += 1
            return deepcopy(self._measurements[key])
        result = _measure_tip_impl(contour, tip, center, up, radius, config,
                                   points=self._points, offsets=self._offsets,
                                   edge_lengths=self._edge_lengths)
        self.computations += 1
        if len(self._measurements) < self._max_entries:
            self._measurements[key] = deepcopy(result)
        return result


def _measure_tip(contour, tip, center, up, radius, config, *, measurement_context=None):
    if measurement_context is not None:
        return measurement_context.measure(contour, tip, center, up, radius, config)
    return _measure_tip_impl(contour, tip, center, up, radius, config)


def _measure_tip_impl(contour, tip, center, up, radius, config, *,
                      points=None, offsets=None, edge_lengths=None):
    # Keep the same rounding boundaries as the independently tested candidate.
    tip = _quantize(tip, _PIXEL_SCALE)
    center = _quantize(center, _PIXEL_SCALE)
    up = _quantize(up, _NORMAL_SCALE)
    radius = _quantize(radius, _PIXEL_SCALE)
    if radius <= 0:
        raise ValueError("Quantized palm radius is not positive")
    pts = _quantize(contour.reshape(-1, 2).astype(np.float64)
                    if points is None else points, _PIXEL_SCALE)
    start = int(np.argmin(np.sum((pts - tip) ** 2, axis=1)))
    steps, width_samples, shaft_samples = _tip_sampling_template(config)
    offsets = np.arange(len(pts)) if offsets is None else offsets
    sides = []
    for sign in (-1, 1):
        chain = pts[(start + sign * offsets) % len(pts)]
        if edge_lengths is None:
            lengths = np.linalg.norm(np.diff(chain, axis=0), axis=1)
        else:
            edge_ids = ((start - offsets[1:]) if sign < 0 else
                        (start + offsets[:-1])) % len(pts)
            lengths = edge_lengths[edge_ids]
        arc = _contour_arc_fixed(lengths)
        sides.append(_quantize(np.column_stack([
            np.interp(steps * radius, arc, chain[:, k]) for k in (0, 1)]),
            _PIXEL_SCALE))
    mid = _quantize((sides[0] + sides[1]) * 0.5, _PIXEL_SCALE)
    widths = _quantize(np.linalg.norm(sides[0] - sides[1], axis=1) / radius,
                       _NORMAL_SCALE)
    width = _quantize(float(np.median(widths[width_samples])), _NORMAL_SCALE)
    # The fixed local shaft window also distinguishes a rounded slender
    # finger from a progressively widening needle/shadow. Do not truncate
    # this window at an early base event: that would hide the widening.
    shaft_widths = widths[shaft_samples]
    shaft_width_ratio = _quantize(
        float(shaft_widths.max() / shaft_widths.min())
        if len(shaft_widths) >= 2 and np.isfinite(shaft_widths).all()
        and shaft_widths.min() > 0 else float("inf"), _NORMAL_SCALE)
    base_candidates = np.flatnonzero((steps >= config.width_end_r) &
                                    (widths > config.base_width_factor * width))
    base_index = int(base_candidates[0]) if len(base_candidates) else len(steps)-1
    base = _quantize(mid[base_index], _PIXEL_SCALE)
    length = _quantize(float(np.linalg.norm(tip - base) / radius), _NORMAL_SCALE)
    # Estimate direction from every centerline point in the local straight
    # shaft, instead of subtracting two contour samples. Pixel-scale changes
    # to either endpoint otherwise rotate the extrapolated root appreciably.
    # Regress both coordinates against arc length; this is a pair of bounded
    # sums and needs no eigenvector, inverse matrix, or image-specific model.
    axis_samples = ((steps >= config.axis_start_r - 1e-9) &
                    (steps <= min(steps[base_index], config.axis_end_r) + 1e-9))
    sample_steps = _quantize(steps[axis_samples], _NORMAL_SCALE)
    sample_mid = _quantize(mid[axis_samples], _PIXEL_SCALE)
    if len(sample_steps) >= 2:
        step_codes = tuple(np.rint(sample_steps * _NORMAL_SCALE).astype(np.int64))
        _, _, centered_steps = _axis_fit_constants(step_codes)
        slope = _quantize(_centered_axis_slope(sample_mid, sample_steps), _PIXEL_SCALE)
        # No identity/extension decision consumes the line-fit residual.
        # Keep its compatibility field but omit the unused image arithmetic.
        residual_r = None
        axis = _quantize(-slope, _PIXEL_SCALE)
        # A least-squares line passes through the sample centroid. The cap's
        # extreme contour point need not lie on that line, especially with a
        # rounded or asymmetric fingertip; using it would translate the fit.
        axis_origin = _quantize(sample_mid.mean(axis=0), _PIXEL_SCALE)
    else:
        # A degenerate configuration is diagnosed through the existing
        # horizontal/invalid-axis fallback, not fabricated finger states.
        axis = _quantize(np.zeros(2, dtype=np.float64), _PIXEL_SCALE)
        residual_r = None
        axis_origin = _quantize(np.asarray(tip, dtype=np.float64), _PIXEL_SCALE)
    axis /= max(1e-9, float(np.linalg.norm(axis)))
    axis = _quantize(axis, _NORMAL_SCALE)
    right = _quantize(np.array([-up[1], up[0]]), _NORMAL_SCALE)
    tip_local = _quantize(
        np.array([(tip-center) @ right, -(tip-center) @ up]) / radius, _NORMAL_SCALE)
    origin_local = _quantize(np.array([(axis_origin-center) @ right,
                             -(axis_origin-center) @ up]) / radius, _NORMAL_SCALE)
    axis_local = _quantize(np.array([axis @ right, -axis @ up]), _NORMAL_SCALE)
    root_x = _quantize(
        float(origin_local[0] + (config.root_y_r-origin_local[1]) *
              axis_local[0] / axis_local[1]) if abs(axis_local[1]) > .15 else None,
        _NORMAL_SCALE)
    aspect = _quantize(length / max(.01, width), _NORMAL_SCALE)
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


_CONVEX_TAN_FRAC_BITS = 8
_CONVEX_HALF_DEGREE_TAN_CODE = 2



@lru_cache(maxsize=16)
def _curvature_smoothing_template(config):
    """Configuration constants: calculate once, not for every hand contour."""
    sigma = config.axis_start_r / (2 * config.contour_step_r)
    offsets = np.arange(-int(np.ceil(3 * sigma)), int(np.ceil(3 * sigma)) + 1)
    weights = np.exp(-0.5 * (offsets / sigma) ** 2)
    weights /= weights.sum()
    offsets.setflags(write=False)
    weights.setflags(write=False)
    return offsets, weights, max(1, round(config.axis_start_r / config.contour_step_r))



def curvature_caps(contour, center, up, radius, config=DEFAULT_CONFIG, minimum_turn_ratio=0.1,
                   *, measurement_context=None, include_lateral=True):
    """Measure the highest shoulder-supported point of each convex arc.

    Canonical traversal and configuration-only smoothing preserve the input
    phase. Cross/dot comparisons approximate the half-degree convex-run dead zone.
    Each run's own inflection shoulders define one chord; its furthest outward
    point is chosen by signed cross product, without local angles or turn
    accumulation. Independent shoulders, local shaft measurements and closed
    digit occupancy checks remain downstream. This is a tested approximation
    to cumulative-turn midpoint selection, not an algebraic equivalence.
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
    offsets, weights, k = _curvature_smoothing_template(config)
    smoothed = sum((w * np.roll(pp, int(j), axis=0) for w, j in zip(weights, offsets)))
    a = smoothed - np.roll(smoothed, k, axis=0)
    b = np.roll(smoothed, -k, axis=0) - smoothed
    area = float(np.sum(pts[:, 0] * np.roll(pts[:, 1], -1) - pts[:, 1] * np.roll(pts[:, 0], -1)))
    winding = 1 if area >= 0 else -1
    cross = winding * (a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])
    dot = (a * b).sum(1)
    # Q8 approximates the half-degree dead zone (about 0.448 degrees).
    positive = ((~np.signbit(cross)) &
                ((dot < 0) | (cross * (1 << _CONVEX_TAN_FRAC_BITS) > dot * _CONVEX_HALF_DEGREE_TAN_CODE)))
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
        left = (ids[0] - 1) % n
        right_index = (ids[-1] + 1) % n
        chord = smoothed[right_index] - smoothed[left]
        offset = smoothed[ids] - smoothed[left]
        # One chord is shared by the arc, so its length cancels for ranking.
        outward_area = -winding * (chord[0] * offset[:, 1] - chord[1] * offset[:, 0])
        cap_index = ids[int(np.argmax(outward_area))]
        keep[cap_index] = True
        cap_shoulders[cap_index] = ((ids[0] - 1) % n, (ids[-1] + 1) % n)
    center = np.asarray(center, float)
    up = np.asarray(up, float)
    up = up / np.linalg.norm(up)
    right = np.array([-up[1], up[0]])
    local = np.column_stack(((smoothed - center) @ right, -(smoothed - center) @ up)) / radius
    keep &= np.sum(local * local, axis=1) > 1
    # Precompute the exact coordinate grids used by _measure_tip_impl. The
    # smoothed arc point is unsuitable for this proof: its nearest original
    # contour point can lie on the opposite side of a zero boundary.
    measure_center = _quantize(center, _PIXEL_SCALE)
    measure_up = _quantize(up, _NORMAL_SCALE)
    measure_right = _quantize(np.array([-measure_up[1], measure_up[0]]), _NORMAL_SCALE)
    measure_radius = _quantize(radius, _PIXEL_SCALE)
    unused_below = max(0.0, config.root_y_r)
    caps = []
    for i in np.flatnonzero(keep):
        left, right_index = cap_shoulders[int(i)]
        aa = smoothed[left]
        bb = smoothed[right_index]
        chord = bb - aa
        offset = smoothed[i] - aa
        length_sq = float(chord @ chord)
        outward_area = -winding * (chord[0] * offset[1] - chord[1] * offset[0])
        if length_sq >= 1e-18 and outward_area < minimum_turn_ratio * length_sq:
            continue
        length = np.sqrt(length_sq)
        depth = outward_area / max(1e-09, length)
        if length_sq < 1e-18 and depth < minimum_turn_ratio * length:
            continue
        point = pts[np.argmin(np.sum((pts - smoothed[i]) ** 2, axis=1))]
        if measure_radius > 0:
            delta = _quantize(point, _PIXEL_SCALE) - measure_center
            tip_local = _quantize(np.array([delta @ measure_right,
                                          -delta @ measure_up]) / measure_radius,
                                  _NORMAL_SCALE)
            # Closed-digit supports reject y >= root_y_r. The only consumer
            # of lower caps is outer-thumb recovery, which requires x > 0
            # and is unreachable once a thumb is already observed. No shape
            # threshold is relaxed, and the measured-coordinate rounding is
            # identical to the downstream consumer's values.
            if (tip_local[1] >= unused_below and
                    (not include_lateral or tip_local[0] <= 0)):
                continue
        f = _measure_tip(contour, point, center, up, radius, config,
                         measurement_context=measurement_context)
        # Preserve downstream strongest-first ordering with a dimensionless
        # shoulder depth score. This deliberately replaces local turn angle.
        f['curvature_turn'] = float(depth / max(1e-09, length))
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
        # A folded digit support must project beyond the same distal root
        # row used by finger identity; a near-palm side bulge is not a digit.
        if y >= config.root_y_r or x > config.thumb_slope * -y:
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
                       tip_index=None, root_fixed=None, root_point=None, closed_support=True)
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
    # Squared fixed-point costs belong to transverse root assignment, not this ordering.
    measured["root_assignment"] = {key: measured[key] for key in
                                  ("candidates", "score_margin_fixed_sq", "total_fixed_sq")}
    measured.update(states=[bool(thumb)] + [bool(f["extended"]) for f in ordered],
                    features=thumb + ordered, candidates=[], score_margin_fixed_sq=None,
                    total_fixed_sq=None, method=method, topology_used=True)
    return measured


def refine_with_closed_caps(measured, contour, palm_center, up, radius,
                            caps, config=DEFAULT_CONFIG, *, measurement_context=None):
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
                               up, radius, config, measurement_context=measurement_context)
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
                       tip_index=None, root_fixed=None, root_point=None, closed_support=True)
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
                      *, closed_candidates=None, measurement_context=None):
    """Return (measurement, failure_reason); failure leaves fallback to caller."""
    if contour is None or len(contour) < 8 or not np.isfinite(radius) or radius <= 0:
        return None, "invalid_geometry"
    center = np.asarray(palm_center, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)
    norm = float(np.linalg.norm(up))
    if not np.isfinite(norm) or norm < 1e-9:
        return None, "invalid_axis"
    up = up / norm
    if measurement_context is None:
        measurement_context = TipMeasurementContext(contour)
    features = [_measure_tip(contour, np.asarray(t["point"], dtype=float), center, up, radius, config,
                              measurement_context=measurement_context)
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
        q = int(np.floor((f["root_x_r"]-lo) / (hi-lo) * ROOT_SCALE + 0.5))
        if ROOT_MIN_FIXED <= q <= ROOT_MAX_FIXED:
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
            f["root_fixed"] = None
            f["root_point"] = None
            retained.append(f)
            continue
        q = int(np.floor((root-lo) / (hi-lo) * ROOT_SCALE + 0.5))
        if not ROOT_MIN_FIXED <= q <= ROOT_MAX_FIXED and f is not thumb:
            if not f["extended"]:
                f["discard_reason"] = "root_out_of_range"
                discarded.append(f)
                continue
            return None, "root_out_of_range"
        f["root_fixed"] = q
        f["root_point"] = (center + radius * (
            root * np.array([-up[1], up[0]]) - config.root_y_r * up)).tolist()
        retained.append(f)
    features = retained
    others = [f for f in features if f is not thumb]
    if len(others) > 4:
        return None, "too_many_candidates"
    others.sort(key=lambda f: -f["root_fixed"])
    slots, ranked, margin = assign_slots_fixed([f["root_fixed"] for f in others], config.slot_positions_fixed)
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
            "candidates": ranked, "score_margin_fixed_sq": margin,
            "total_fixed_sq": ranked[0]["cost_fixed_sq"], "method": "root_geometry_fixed"}
    if closed_candidates is not None:
        measured = refine_with_closed_caps(measured, contour, center, up, radius,
                                          closed_candidates, config,
                                          measurement_context=measurement_context)
        non_thumb_count = sum(feature["slot"] != 0 for feature in measured["features"])
        need_caps = ((0 < non_thumb_count < 4 and not measured.get("topology_used")) or
                     (non_thumb_count == 4 and not measured["states"][0]))
        caps = curvature_caps(contour, center, up, radius, config,
                              measurement_context=measurement_context,
                              include_lateral=not measured["states"][0]) if need_caps else []
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
               tip_index=None, root_fixed=None, root_point=None, thumb_recovered=True)
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
                           tip_index=None, root_fixed=None, root_point=None, closed_support=True)
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
                       tip_index=None, root_fixed=int(np.floor((root-lo)/(hi-lo)*ROOT_SCALE + .5)),
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
    combined = sorted(observed + supported, key=lambda item: -item["root_fixed"])
    slots, ranked, margin = assign_slots_fixed([feature["root_fixed"] for feature in combined],
                                                       config.slot_positions_fixed)
    states = [bool(thumbs), False, False, False, False]
    for slot, feature in zip(slots, combined):
        feature["slot"] = slot
        states[slot] = bool(feature["extended"])
    measured["initial_root_assignment"] = {key: measured.get(key) for key in
                                           ("candidates", "score_margin_fixed_sq", "total_fixed_sq")}
    measured.update(features=thumbs + combined, states=states, candidates=ranked,
                    score_margin_fixed_sq=margin, total_fixed_sq=ranked[0]["cost_fixed_sq"],
                    method="root_geometry_fixed", topology_used=False,
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
                   tip_index=None, root_fixed=None, root_point=None, closed_support=True)
    first["slot"] = 0
    result = _complete_digit_topology(measured, closed, "five_observed_positions_thumb")
    result["thumb_recovered"] = "five_observed_positions"
    return result

