"""Finger identity from local contour axes and transverse palm position.

No image names, truth labels, learned models, or previous frames are consumed.
Identity and extension are separate. Five-finger identity uses transverse
root position with Q6 integers and at most 16 subsets. Closed-knuckle identity
completion and its extra support list are removed. Weak fingertip promotion
still uses the same local shaft measurement upstream.

The deployed contour contract is integer CHAIN_APPROX_NONE: adjacent points
are eight-connected, with repeated points permitted. Candidates carry their
canonical first-occurrence contour index. There is no coordinate-index table,
nearest-point search fallback, or stored edge-code plane. A shared pair of
coordinate equality comparisons derives each Q4 step (0, 16 or 23) from the
existing points; a shared wide accumulator preserves the original precision.

Feature extraction rounds coordinates to 1/1 pixels and normalized
features to 1/32. Local tip contour-length sums retain Q4 precision. Centerline
regression uses integer centered-coordinate sums. Interpolation, square
roots, division and point projection still use floating point: this is a
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

# Engineering interface capacities, not a proof that every binary mask fits.
# An overflowing frame is unavailable; points/candidates are never truncated.
MAX_CONTOUR_POINTS = 2048
MAX_RAW_CANDIDATES = 16
MAX_TIP_SAMPLES = 61
MAX_IDENTIFICATION_TIPS = 6
MAX_MEASUREMENT_REQUESTS = 2 * MAX_RAW_CANDIDATES + 2 * MAX_IDENTIFICATION_TIPS
MAX_MEASUREMENT_COMPUTATIONS = MAX_RAW_CANDIDATES + MAX_IDENTIFICATION_TIPS


class GeometryCapacityError(RuntimeError):
    """Finite hardware contract exceeded; caller must emit unavailable."""

    def __init__(self, reason, actual, limit):
        self.reason = str(reason)
        self.actual = int(actual)
        self.limit = int(limit)
        super().__init__(f"{self.reason}: {self.actual} > {self.limit}")


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


def _validate_dense_contour(points):
    """PC contract check; the upstream CHAIN_APPROX_NONE trace guarantees this.

    The deployed interface accepts only integer, eight-connected contour
    neighbours (including zero-length repeats), never sparse approximations.
    This host assertion is not an alternate hardware geometry path.
    """
    if len(points) == 0 or not np.isfinite(points).all():
        raise ValueError("Expected a nonempty finite dense contour")
    if not np.equal(points, np.rint(points)).all():
        raise ValueError("Dense contour coordinates must be integer pixels")
    if np.any(np.abs(np.roll(points, -1, axis=0)-points) > 1):
        raise ValueError("Expected CHAIN_APPROX_NONE neighbour steps")


class TipMeasurementContext:
    """One shared measurement service with 16 frame-local candidate records.

    The contour is immutable for this frame. Keys use the exact quantization
    already performed by _measure_tip_impl, so tiny *pre-quantization* axis
    differences no longer trigger duplicate work. A candidate's record can
    be replaced if the identity stage uses a genuinely different quantized
    axis. This is 16 storage slots, not 16 arithmetic engines.

    At most 16 initial candidates, 16 optional needle checks, 6 cap checks,
    and 6 identity requests are allowed (44 requests). Needle/cap requests
    hit the same record; at most 16 + 6 computations are needed. Counters
    explicitly enforce both bounds; cache size alone is not a work bound.
    """

    def __init__(self, contour):
        if len(contour) > MAX_CONTOUR_POINTS:
            raise GeometryCapacityError("contour_capacity_exceeded", len(contour), MAX_CONTOUR_POINTS)
        self._source_contour = contour
        self._points = contour.reshape(-1, 2).astype(np.float64)
        self._points.setflags(write=False)
        _validate_dense_contour(self._points)
        self._points = _quantize(self._points, _PIXEL_SCALE)
        self._points.setflags(write=False)
        # No edge-code/edge-length RAM: steps are derived from existing points.
        self._measurements = {}
        self.requests = 0
        self.computations = 0
        self.cache_hits = 0

    @staticmethod
    def _parameter_key(tip, center, up, radius, config):
        # _measure_tip_impl converts each of these to float64 and rounds to
        # exactly these grids *before* using it. No numerical precision is
        # discarded by sharing equal rounded inputs.
        return (tuple(_quantize(tip, _PIXEL_SCALE)),
                tuple(_quantize(center, _PIXEL_SCALE)),
                tuple(_quantize(up, _NORMAL_SCALE)),
                float(_quantize(radius, _PIXEL_SCALE)), config)

    def measure(self, contour, tip, center, up, radius, config=DEFAULT_CONFIG, *, contour_index):
        if contour is not self._source_contour:
            raise ValueError("Measurement context belongs to another immutable contour")
        if self.requests >= MAX_MEASUREMENT_REQUESTS:
            raise GeometryCapacityError("measurement_request_capacity_exceeded", self.requests + 1, MAX_MEASUREMENT_REQUESTS)
        self.requests += 1
        index = int(contour_index)
        key = self._parameter_key(tip, center, up, radius, config)
        if index != contour_index or not 0 <= index < len(self._points) or not np.array_equal(self._points[index], key[0]):
            raise ValueError("Candidate contour index does not match its point")
        cached = self._measurements.get(index)
        if cached is not None and cached[0] == key:
            self.cache_hits += 1
            return deepcopy(cached[1])
        if cached is None and len(self._measurements) >= MAX_RAW_CANDIDATES:
            raise GeometryCapacityError("measurement_candidate_capacity_exceeded", len(self._measurements) + 1, MAX_RAW_CANDIDATES)
        if self.computations >= MAX_MEASUREMENT_COMPUTATIONS:
            raise GeometryCapacityError("measurement_computation_capacity_exceeded", self.computations + 1, MAX_MEASUREMENT_COMPUTATIONS)
        result = _measure_tip_impl(contour, tip, center, up, radius, config,
                                   points=self._points,
                                   start_index=index,
                                   points_quantized=True)
        self.computations += 1
        self._measurements[index] = (key, deepcopy(result))
        return result


def _measure_tip(contour, tip, center, up, radius, config, *, contour_index, measurement_context=None):
    if measurement_context is not None:
        return measurement_context.measure(contour, tip, center, up, radius, config, contour_index=contour_index)
    return _measure_tip_impl(contour, tip, center, up, radius, config, start_index=contour_index)


def _sample_tip_side(points, start, sign, targets):
    """Stream one contour side into fixed sampling slots, without chain RAM.

    Keep just two contour endpoints and one Q4 arc accumulator. A point is
    consumed at most once per side (including duplicate zero-length steps).
    Output follows np.interp endpoint/duplicate behaviour; quantization is
    still performed by the caller at the established pixel boundary.
    """
    count = len(points)
    if count > MAX_CONTOUR_POINTS:
        raise GeometryCapacityError("contour_capacity_exceeded", count, MAX_CONTOUR_POINTS)
    if len(targets) > MAX_TIP_SAMPLES:
        raise GeometryCapacityError("tip_sample_capacity_exceeded", len(targets), MAX_TIP_SAMPLES)
    sampled = np.empty((len(targets), 2), dtype=np.float64)
    output = 0
    previous = int(start)
    arc_code = 0
    for _ in range(count - 1):
        following = (previous + sign) % count
        x_same = points[previous, 0] == points[following, 0]
        y_same = points[previous, 1] == points[following, 1]
        step_code = 0 if x_same and y_same else (_ARC_SCALE if x_same or y_same else 23)
        next_code = arc_code + step_code
        if step_code:
            before = arc_code / _ARC_SCALE
            after = next_code / _ARC_SCALE
            slope = (points[following] - points[previous]) / (after - before)
            while output < len(targets) and targets[output] <= after:
                target = targets[output]
                if target == after:
                    sampled[output] = points[following]
                elif target == before:
                    sampled[output] = points[previous]
                else:
                    sampled[output] = points[previous] + slope * (target - before)
                output += 1
        previous, arc_code = following, next_code
        if output == len(targets):
            break
    if output < len(targets):
        sampled[output:] = points[previous]
    return sampled


def _measure_tip_impl(contour, tip, center, up, radius, config, *,
                      points=None,
                      start_index=None, points_quantized=False):
    # Keep the same rounding boundaries as the independently tested candidate.
    tip = _quantize(tip, _PIXEL_SCALE)
    center = _quantize(center, _PIXEL_SCALE)
    up = _quantize(up, _NORMAL_SCALE)
    radius = _quantize(radius, _PIXEL_SCALE)
    if radius <= 0:
        raise ValueError("Quantized palm radius is not positive")
    pts = points if points_quantized else _quantize(
        contour.reshape(-1, 2).astype(np.float64) if points is None else points,
        _PIXEL_SCALE)
    if start_index is None:
        raise ValueError("Core measurement requires the candidate contour index")
    start = int(start_index)
    if not 0 <= start < len(pts) or not np.array_equal(pts[start], tip):
        raise ValueError("Candidate contour index does not match its point")
    steps, width_samples, shaft_samples = _tip_sampling_template(config)
    sides = []
    for sign in (-1, 1):
        sides.append(_quantize(_sample_tip_side(pts, start, sign, steps * radius), _PIXEL_SCALE))
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











def _is_lateral_thumb_branch(feature, center, up, radius, config):
    """Recognize a separate thumb branch from its low, lateral attachment.

    The distal segment may bend inward; its tip angle need not point outward.
    A weak closed cap needs two observed shaft widths of branch length so a
    folded knuckle cannot gain extension from its lateral location alone.
    Existing Q5 length and width values suffice for this comparison. The
    detected branch base must still lie on the low, lateral palm, and the tip
    must remain in the thumb-side half-plane within one palm diameter.
    """
    if (not feature.get("extended") or not feature.get("has_branch_extent") or
            not feature.get("base_detected")):
        return False
    if feature["length_r"] < 2 * feature["width_r"]:
        return False
    x, y = feature["tip_local"]
    if (not np.isfinite([x, y]).all() or not -2.0 < y < 0 or
            x <= 0):
        return False
    right = np.array([-up[1], up[0]])
    delta = np.asarray(feature["base_point"], dtype=float) - center
    base_x, base_y = float(delta @ right / radius), float(-delta @ up / radius)
    return bool(np.isfinite([base_x, base_y]).all() and
                base_y > config.root_y_r and base_x > abs(base_y))



def _thumbward_candidate(features):
    best = None
    bx = by = 0
    for feature in features:
        x,y = (int(np.rint(v*32)) for v in feature['tip_local'])
        if abs(x)>1712 or abs(y)>1712:
            raise ValueError('250x180, R>=8 Q5 geometry contract exceeded')
        if best is None:
            best,bx,by=feature,x,y
            continue
        if y < 0 and by < 0:
            larger = x*(-by) > bx*(-y)
        elif y >= 0 and by >= 0:
            larger = x > bx
        elif y >= 0:
            larger = x > 0 or (x == 0 and bx < 0)
        else:
            larger = bx < 0 or (bx == 0 and x > 0)
        if larger: best,bx,by=feature,x,y
    return best


def recognize_fingers(mask, contour, palm_center, up, radius, tips, config=DEFAULT_CONFIG,
                      *, measurement_context=None):
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
                              contour_index=t["contour_index"], measurement_context=measurement_context)
                for t in tips]
    for index, feature in enumerate(features):
        feature["tip_index"] = index
    bounds = _palm_bounds(mask, center, up, radius, config)
    if bounds is None:
        return None, "palm_scan_unavailable"
    lo, hi = bounds
    # Select the most thumbward tip without an inverse trigonometric function.
    first = _thumbward_candidate(features)
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
    return measured, None
