"""Fixed black-background gesture reference for a 250x180 native-right core.

Camera input is fixed 320x180; the PC adapter physically crops x=40:290.
No board search, image rotation, ROI bitmap, shadow reconstruction or scene
calibration remains. Preprocessing is fixed luminance threshold, 3x3 opening,
one component filter and binary hole filling. Palm uses fixed 7x7 closing.
Five-finger geometry and the previously validated low precision are preserved.
DT uses 11-bit Q2 codes; geometry retains pixel/normal/arc/root grids.
This Python numerical reference still contains frame-based component labeling,
distance scans and contour geometry; it is not a finished one-pass RTL core.
Labels, filenames, playback state and past decisions never enter the core.
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


INPUT_MAX_WIDTH = 320


INPUT_MAX_HEIGHT = 180


CROP_X = 40


CROP_Y = 0


CROP_WIDTH = 250


CROP_HEIGHT = 180


DT_FRACTION_BITS = 2


DT_WORD_BITS = 11


DT_SCALE = 1 << DT_FRACTION_BITS


DT_INFINITY = (1 << DT_WORD_BITS) - 1


DT_FINITE_MAX = DT_INFINITY - 1


DT_AXIAL_WEIGHT = DT_SCALE


DT_DIAGONAL_WEIGHT = (14 * DT_SCALE + 5) // 10


DT_KNIGHT_WEIGHT = (21969 * DT_SCALE + 5000) // 10000


class DistanceWorkspace:
    """One lazily allocated DT state RAM owned by one recognition call.

    For 250x180, a finite axial path is at most 4*(249+179)=1712,
    below the 11-bit finite limit 2046. Q2 fractional precision is unchanged.
    The ndarray remains uint16 in Python; this is a hardware-sharing reference,
    not a claim that Python storage or synthesized RAM automatically uses 11 bits.

    A codes result borrows this state and is valid only until the next DT call
    with this workspace. The core runs peak first (only scalars escape), then
    codes, and performs no further DT while consuming those codes. Nothing is
    global or retained in RecognitionResult, so frames cannot share state.
    """

    def __init__(self, shape_hw):
        self.shape = tuple(int(v) for v in shape_hw)
        if len(self.shape) != 2 or min(self.shape) <= 0:
            raise ValueError("Distance workspace requires a nonempty 2D shape")
        self.state = None
        self.allocations = 0
        self.uses = 0

    def reset(self, shape_hw, initial):
        if tuple(shape_hw) != self.shape:
            raise ValueError("Distance workspace shape does not match input")
        height, width = self.shape
        if self.state is None:
            self.state = np.empty((height + 4, width + 4), dtype=np.uint16)
            self.allocations += 1
        self.state.fill(initial)
        self.uses += 1
        return self.state


def _distance_transform_fixed(mask: np.ndarray, *, diagnostics=None,
                           output="distance", valid_mask=None, threshold_pixels=None,
                           workspace=None):
    """Bounded integer chamfer5 recurrence at explicit logical precision.

    All weights, horizontal propagation, threshold comparison, result adapter
    and sentinels derive from DT_FRACTION_BITS and DT_WORD_BITS. uint16 is the
    Python storage container, not a claim that hardware must use 16 bits.
    int32 scratch models nonwrapping addition before bounded state writeback.
    Exact-final-cap and unreachable outputs raise, never silently pass as valid
    distances. Threshold mode intentionally caps at its comparison threshold.
    All-foreground input returns the established explicit infinity sentinel.
    """
    source = np.asarray(mask)
    if source.ndim != 2 or source.dtype != np.uint8 or source.size == 0:
        raise ValueError("Distance input must be a nonempty 2D uint8 mask")
    height, width = source.shape
    if max(height, width) > 32768:
        raise ValueError("Distance input dimensions exceed the bounded reference")
    if workspace is not None:
        if not isinstance(workspace, DistanceWorkspace):
            raise TypeError("Expected a DistanceWorkspace or None")
        if workspace.shape != source.shape:
            raise ValueError("Distance workspace shape does not match input")
    if output not in ("distance", "codes", "peak", "threshold"):
        raise ValueError("Unsupported distance consumer")
    if valid_mask is not None and (output != "peak" or np.shape(valid_mask) != source.shape):
        raise ValueError("Distance validity mask must match a peak input")
    threshold_code = None
    if output == "threshold":
        if threshold_pixels is None or np.ndim(threshold_pixels) != 0 or not np.isfinite(threshold_pixels):
            raise ValueError("A finite distance threshold is required")
        # Preserve NumPy's float32 result-array comparison, including its weak
        # Python scalar and explicit NumPy float64 scalar promotion rules.
        comparison_dtype = np.result_type(np.dtype(np.float32), threshold_pixels)
        normalized_threshold = np.asarray(threshold_pixels, dtype=comparison_dtype).item()
        threshold_code = int(np.ceil(float(normalized_threshold) * DT_SCALE))
        if not 0 <= threshold_code <= DT_FINITE_MAX:
            raise ValueError("Threshold exceeds the configured fixed-distance range")
    if diagnostics is not None:
        diagnostics["calls"] = diagnostics.get("calls", 0) + 1
        diagnostics["pixels"] = diagnostics.get("pixels", 0) + source.size
        diagnostics["configuration"] = [DT_FRACTION_BITS, DT_WORD_BITS,
                                        DT_AXIAL_WEIGHT, DT_DIAGONAL_WEIGHT, DT_KNIGHT_WEIGHT]
        diagnostics["python_storage_dtype"] = "uint16"
        diagnostics["logical_word_bits"] = DT_WORD_BITS
        diagnostics["fraction_bits"] = DT_FRACTION_BITS
        shapes = diagnostics.setdefault("input_shapes", {})
        shape_key = str((height, width))
        shapes[shape_key] = shapes.get(shape_key, 0) + 1
        manhattan = (height + width - 2) * DT_SCALE
        diagnostics["max_grid_manhattan_upper_code"] = max(diagnostics.get("max_grid_manhattan_upper_code", 0), manhattan)
        diagnostics["required_full_grid_logical_bits"] = max(diagnostics.get("required_full_grid_logical_bits", 0), (manhattan + 2).bit_length())
        diagnostics["max_finite_representable"] = DT_FINITE_MAX / DT_SCALE
        diagnostics[output + "_calls"] = diagnostics.get(output + "_calls", 0) + 1
        if output == "threshold":
            diagnostics["last_threshold_code"] = threshold_code
            diagnostics["max_threshold_state_bits"] = max(diagnostics.get("max_threshold_state_bits", 0), max(1, threshold_code.bit_length()))
            diagnostics["threshold_state_semantics"] = "min(distance_code, threshold_code); intentional truncation"
    if threshold_code == 0:
        if diagnostics is not None:
            diagnostics["predicate_true_pixels"] = diagnostics.get("predicate_true_pixels", 0) + source.size
        return np.ones(source.shape, dtype=bool)
    if not np.any(source == 0):
        if diagnostics is not None:
            diagnostics["all_foreground_calls"] = diagnostics.get("all_foreground_calls", 0) + 1
        if output == "threshold":
            if diagnostics is not None:
                diagnostics["predicate_true_pixels"] = diagnostics.get("predicate_true_pixels", 0) + source.size
            return np.ones(source.shape, dtype=bool)
        if output == "peak":
            if valid_mask is None:
                return (0, 0), float(np.finfo(np.float32).max)
            positions = np.flatnonzero(valid_mask)
            if not len(positions):
                return (-1, -1), 0.0
            y, x = divmod(int(positions[0]), width)
            return (x, y), float(np.finfo(np.float32).max)
        if output == "codes":
            if workspace is not None:
                state = workspace.reset(source.shape, DT_INFINITY)
                return state[2:height + 2, 2:width + 2]
            return np.full(source.shape, DT_INFINITY, np.uint16)
        return np.full(source.shape, np.finfo(np.float32).max, np.float32)

    infinity = DT_INFINITY
    finite_max = DT_FINITE_MAX if threshold_code is None else threshold_code
    big = 1 << 29  # Scratch sentinel, never stored as uint16.
    initial = infinity if threshold_code is None else threshold_code
    state = (np.full((height + 4, width + 4), initial, dtype=np.uint16)
             if workspace is None else workspace.reset(source.shape, initial))
    state[2:height + 2, 2:width + 2][source == 0] = 0
    offsets = np.arange(width, dtype=np.int32) * DT_SCALE
    work = np.empty(width, dtype=np.int32)
    saturation_events = 0
    max_preclip = 0
    peak_code, peak_location = -1, (-1, -1)
    final_capped = final_unreachable = observed_max = 0

    def read_row(index):
        row = state[index].astype(np.int32)
        row[row == infinity] = big
        return row

    def sweep(y, direction):
        nonlocal saturation_events, max_preclip
        base = state[y, 2:width + 2].astype(np.int32)
        base[base == infinity] = big
        near = read_row(y - direction)
        far = read_row(y - 2 * direction)
        for values, weight in ((near[2:width + 2], DT_AXIAL_WEIGHT),
                               (near[1:width + 1], DT_DIAGONAL_WEIGHT), (near[3:width + 3], DT_DIAGONAL_WEIGHT),
                               (near[:width], DT_KNIGHT_WEIGHT), (near[4:width + 4], DT_KNIGHT_WEIGHT),
                               (far[1:width + 1], DT_KNIGHT_WEIGHT), (far[3:width + 3], DT_KNIGHT_WEIGHT)):
            np.add(values, weight, out=work)
            np.minimum(base, work, out=base)
        ordered = base if direction == 1 else base[::-1]
        ordered -= offsets
        np.minimum.accumulate(ordered, out=ordered)
        ordered += offsets
        finite = base < big
        if diagnostics is not None:
            saturation_events += int(np.count_nonzero(finite & (base > finite_max)))
            max_preclip = max(max_preclip, int(np.max(base[finite], initial=0)))
        np.minimum(base, finite_max, out=base)
        base[~finite] = infinity
        state[y, 2:width + 2] = base

    for y in range(2, height + 2):
        sweep(y, 1)
    for y in range(height + 1, 1, -1):
        sweep(y, -1)
        if output == "peak":
            row = state[y, 2:width + 2]
            final_capped += int(np.count_nonzero(row == finite_max))
            final_unreachable += int(np.count_nonzero(row == infinity))
            observed_max = max(observed_max, int(np.max(row[row != infinity], initial=0)))
            valid = None if valid_mask is None else np.asarray(valid_mask[y - 2]) != 0
            if valid is not None and not np.any(valid):
                continue
            values = row if valid is None else np.where(valid, row.astype(np.int32), -1)
            x = int(np.argmax(values))
            value = int(values[x])
            # Backward traversal visits later rows first. Equality must select
            # the earlier row, matching minMaxLoc's first row-major maximum.
            if value >= peak_code:
                peak_code, peak_location = value, (x, y - 2)
    core = state[2:height + 2, 2:width + 2]
    if output == "threshold":
        predicate = core >= threshold_code
        if diagnostics is not None:
            diagnostics["intentional_threshold_clipped_updates"] = diagnostics.get("intentional_threshold_clipped_updates", 0) + saturation_events
            diagnostics["predicate_true_pixels"] = diagnostics.get("predicate_true_pixels", 0) + int(np.count_nonzero(predicate))
        return predicate
    capped = final_capped if output == "peak" else int(np.count_nonzero(core == finite_max))
    unreachable = final_unreachable if output == "peak" else int(np.count_nonzero(core == infinity))
    if diagnostics is not None:
        for key, value in (("saturated_scan_updates", saturation_events),
                           ("final_at_finite_cap_pixels", capped),
                           ("unreachable_final_pixels", unreachable)):
            diagnostics[key] = diagnostics.get(key, 0) + value
        diagnostics["max_preclip_code"] = max(diagnostics.get("max_preclip_code", 0), max_preclip)
        maximum = observed_max if output == "peak" else int(np.max(core[core != infinity], initial=0))
        diagnostics["max_observed_finite_code"] = max(diagnostics.get("max_observed_finite_code", 0), maximum)
        diagnostics["max_observed_finite_distance"] = diagnostics["max_observed_finite_code"] / DT_SCALE
    if capped or unreachable:
        raise ValueError(f"Distance outside configured Q{DT_FRACTION_BITS}/{DT_WORD_BITS}-bit range "
                         f"({DT_FINITE_MAX / DT_SCALE:g} px); capped={capped}, unreachable={unreachable}")
    if output == "peak":
        return peak_location, max(0, peak_code) / DT_SCALE
    if output == "codes":
        # Borrow the bounded integer state. Consumers scale isolated samples
        # or thresholds, never allocate a second floating-point distance map.
        return core
    result = core.astype(np.float32)
    result *= np.float32(1.0 / DT_SCALE)
    return result


PALM_MIN_RADIUS_PX = 8


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


COLOR_BOUNDARY = (0, 200, 255)


COLOR_TEXT = (255, 255, 255)


COLOR_PALM = (0, 165, 255)


COLOR_WRIST = (0, 255, 255)


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


def _active_rect(shape_hw, state):
    return (0, 0, int(shape_hw[1]), int(shape_hw[0]))


def segment_foreground(frame, state):
    """8-bit fixed-coefficient luminance and one fixed threshold on black background.

    The host uses OpenCV's integer BGR-to-gray conversion. RTL can implement
    its fixed coefficients with constant arithmetic; no color histogram,
    adaptive skin box, background bitmap or per-frame calibration is used.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mask = cv2.inRange(gray, FOREGROUND_MIN_GRAY, 255)
    state.flags['segment_stats'] = {'method': 'fixed_black_luminance',
                                  'active_area': int(mask.size)}
    return mask


HOLE_SCAN_ROUNDS = 2


def _fill_internal_holes(binary, *, inplace=False):
    """Fixed native-width state: 0 unseen background, 1 reachable, 255 foreground.

    These three states need two logical bits, independent of the host byte
    representation. Production may own/reuse the cleaned mask. All scratch
    below is row-local; no background plane, reachable plane or output copy.
    """
    state = binary if inplace else binary.copy()
    height, width = state.shape
    for edge in (state[0], state[-1], state[:, 0], state[:, -1]):
        edge[edge == 0] = 1
    for _ in range(HOLE_SCAN_ROUNDS):
        for direction in (1, -1):
            rows = range(height) if direction == 1 else range(height - 1, -1, -1)
            for y in rows:
                row = state[y]
                background = row != 255
                starts = background & np.concatenate((np.ones(1, dtype=bool), ~background[:-1]))
                indices = np.flatnonzero(starts)
                if indices.size == 0:
                    continue
                seeds = row == 1
                previous = y - direction
                if 0 <= previous < height:
                    seeds |= state[previous] == 1
                seeds &= background
                active = np.concatenate((np.zeros(1, dtype=bool), np.maximum.reduceat(seeds, indices)))
                # At width 250 the number of alternating runs is at most125.
                labels = np.cumsum(starts, dtype=np.uint16)
                row[background & active[labels]] = 1
    for row in state:
        row[:] = np.where(row == 1, 0, 255)
    return state


def clean_mask(mask, state):
    """Fixed 3x3 opening, one component filter, then bounded binary hole fill.

    The binary component filter is required to remove disconnected bright
    board-edge fragments before wrist and palm-width measurements. Its
    relative threshold is integer ceil(largest/4); absolute threshold is
    ceil(45000/500). Areas fit 16 unsigned bits for this fixed crop.
    """
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _OPEN_KERNEL)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(opened, 8)
    largest = int(stats[1:, cv2.CC_STAT_AREA].max()) if count > 1 else 0
    minimum = max(1, (CROP_WIDTH*CROP_HEIGHT + 499)//500, (largest + 3)//4)
    lookup = np.zeros(count, np.uint8)
    if count > 1:
        lookup[1:] = (stats[1:, cv2.CC_STAT_AREA] >= minimum).astype(np.uint8)*255
    kept = int(np.count_nonzero(lookup))
    state.flags['segment_stats'].update(kept_components=kept,
                                        min_area_used=minimum)
    cleaned = lookup[labels]
    del opened, labels
    return _fill_internal_holes(cleaned, inplace=True) if kept else cleaned


def _estimate_palm_robust(mask, *, workspace=None):
    """Fixed 7x7 palm-only closing followed by the existing Q2 peak scan.

    Closing does not change the silhouette used for finger geometry. The
    window is fixed: no image-area square root or variable-sized kernel.
    """
    work = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _PALM_CLOSE_KERNEL)
    location, radius = _distance_transform_fixed(work, output='peak', workspace=workspace)
    return location, radius


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


_TIP_MAX_COS = float(np.cos(np.radians(RECOG_TIP_MAX_ANGLE_DEG)))


_TIP_MIN_COS = float(np.cos(np.radians(RECOG_TIP_MIN_ANGLE_DEG)))


def _angle_compare_vectors(first, second, cosine):
    """Compare an angle with a fixed limit without sqrt/division/acos.

    Return -1, 0 or 1 for below/equal/above the limit. A zero vector has
    the old 180-degree meaning. Preserve signs before squaring so obtuse
    and acute angles cannot be confused.
    """
    aa, bb = float(first @ first), float(second @ second)
    if aa < 1e-12 or bb < 1e-12:
        return 1
    dot = float(first @ second)
    if cosine >= 0:
        if dot < 0:
            return 1
        lhs, rhs = dot * dot, cosine * cosine * aa * bb
        return int(lhs < rhs) - int(lhs > rhs)
    if dot >= 0:
        return -1
    lhs, rhs = dot * dot, cosine * cosine * aa * bb
    return int(lhs > rhs) - int(lhs < rhs)


def _tip_shape_tests(contour, index, window):
    """Threshold-only versions of max turn ratio and median of 3 angles."""
    here = _contour_point(contour, index)
    enough_turn = False
    above_max = below_min = 0
    for factor in (0.6, 1.0, 1.5):
        distance = max(2, int(round(window * factor)))
        before = _contour_point(contour, index - distance)
        after = _contour_point(contour, index + distance)
        chord = after - before
        chord2 = float(chord @ chord)
        offset = here - before
        cross = abs(float(offset[0] * chord[1] - offset[1] * chord[0]))
        enough_turn |= (chord2 >= 1e-12 and
                        cross >= RECOG_TIP_MIN_TURN_RATIO * chord2)
        first, second = before - here, after - here
        above_max += _angle_compare_vectors(first, second, _TIP_MAX_COS) > 0
        below_min += _angle_compare_vectors(first, second, _TIP_MIN_COS) < 0
    return not enough_turn, above_max >= 2, below_min >= 2


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
    threshold_code = int(np.ceil(float(np.float32(float(RECOG_TIP_PALM_THICK) * float(radius))) * DT_SCALE))
    hit = np.nonzero((values == DT_INFINITY) | (values >= threshold_code))[0]
    if hit.size == 0:
        return total / float(radius)
    return float(fractions[int(hit[0])]) * total / float(radius)


def _right_entry_wrist(mask, palm_center, radius, column):
    """Use a visible right-entry section only when no wrist valley is available.

    One column's longest foreground run rejects separated border fragments.
    A quarter-radius minimum baseline avoids amplifying a nearly coincident
    palm/section pair. This is point arithmetic; no image is rotated.
    """
    cx, cy = float(palm_center[0]), float(palm_center[1])
    if column <= cx or 4 * (column - cx) < float(radius):
        return None
    occupied = mask[:, column] != 0
    changes = np.diff(np.r_[False, occupied, False].astype(np.int8))
    starts, ends = np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)
    if not starts.size:
        return None
    widths = ends - starts
    chosen = int(np.argmax(widths))
    if int(widths[chosen]) < int(RECOG_WRIST_MIN_PIXELS):
        return None
    middle = 0.5 * float(starts[chosen] + ends[chosen] - 1)
    return np.array([float(column), middle], dtype=np.float64)


def _estimate_wrist(mask, palm_center, radius):
    """Measure column heights to the right of the palm; no image transpose."""
    column_heights = np.count_nonzero(mask, axis=0)
    columns = np.flatnonzero(column_heights)
    if not columns.size:
        return None
    last = int(columns[-1])
    cx = int(round(float(palm_center[0])))
    start = min(last, cx + int(round(0.8 * float(radius))))
    if last - start < 3:
        return _right_entry_wrist(mask, palm_center, radius, last)
    segment = column_heights[start:last + 1].astype(np.float32)
    kernel = max(3, int(round(0.2 * float(radius))) | 1)
    smoothed = cv2.blur(segment.reshape(-1, 1), (1, kernel)).ravel()
    peak = float(smoothed.max())
    if peak <= 0:
        return None
    left = np.r_[smoothed[0], smoothed[:-1]]
    right = np.r_[smoothed[1:], smoothed[-1]]
    candidates = np.flatnonzero((smoothed <= left) & (smoothed <= right) & (smoothed < 0.88 * peak))
    column = start + (int(candidates[0]) if candidates.size else int(np.argmin(smoothed)))
    ys = np.flatnonzero(mask[:, column])
    if ys.size < int(RECOG_WRIST_MIN_PIXELS) or column <= palm_center[0]:
        return _right_entry_wrist(mask, palm_center, radius, last)
    return np.array([float(column), 0.5 * float(ys.min() + ys.max())], dtype=np.float64)


def _truncate_forearm(mask: np.ndarray, palm_center, up, radius: float) -> np.ndarray:
    """Clip the wrist half-plane with row-local integer coordinate tests.

    There is no full-frame polygon-validity buffer. The retained side and
    rounded cut endpoints follow the prior wrist geometry. A half-pixel
    boundary allowance includes the rasterized cut edge, as a binary line.
    """
    height, width = mask.shape
    far = 2.0 * max(height, width)
    axis = np.asarray(up, dtype=np.float64)
    origin = np.asarray(palm_center, dtype=np.float64) - axis * (RECOG_TRUNCATE_FACTOR * radius)
    along = np.array([-axis[1], axis[0]])
    first, second = np.asarray([origin + along * far, origin - along * far], dtype=np.int32)
    dx, dy = (int(v) for v in second - first)
    sign = 1 if dx * axis[1] - dy * axis[0] >= 0 else -1
    allowance = max(abs(dx), abs(dy)) // 2
    xterm = -dy * (np.arange(width, dtype=np.int32) - int(first[0]))
    result = mask.copy()
    for y in range(height):
        values = dx * (y - int(first[1])) + xterm
        result[y, sign * values < -allowance] = 0
    return result


def _detect_fingertips(contour, mask: np.ndarray, palm_center, radius: float, up,
                       *, closed_candidates=None, measurement_context=None,
                       workspace=None):
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
    thickness = _distance_transform_fixed(mask, output="codes", workspace=workspace)
    candidates = []
    for raw_index in candidate_indices:
        index = int(raw_index)
        flat_turn, too_flat, too_sharp = _tip_shape_tests(contour, index, window)
        inner = points[index] + (np.asarray(palm_center) - points[index]) * \
            float(RECOG_TIP_THICKNESS_SAMPLE)
        sample_code = _sample_plane(thickness, inner)
        sample_pixels = (float(np.finfo(np.float32).max) if sample_code == DT_INFINITY
                         else sample_code / DT_SCALE)
        thickness_ratio = sample_pixels / float(radius)
        if flat_turn and \
                thickness_ratio > RECOG_TIP_THICKNESS_MAX:
            continue
        if too_flat:
            continue
        # 针状毛刺：夹角和厚度同时极端小 -> 不是指尖（真指尖是圆帽，两者都不会这么极端）
        if too_sharp and \
                thickness_ratio < RECOG_TIP_MIN_THICKNESS:
            # The radial sample can cross a finger gap. Rescue only a long
            # local shaft with opposing sides that do not markedly taper.
            shaft = FG._measure_tip(contour, points[index],
                                    np.asarray(palm_center, dtype=float),
                                    np.asarray(up, dtype=float), radius, FG.DEFAULT_CONFIG,
                                    measurement_context=measurement_context)
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
            "turn_ratio": None,
            "angle_deg": None,
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
                                      np.asarray(up, dtype=float), radius, FG.DEFAULT_CONFIG,
                                      measurement_context=measurement_context)
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
    workspace = getattr(state, "dt_workspace", None)
    palm_center = segment_stats.get("palm_center")
    radius = segment_stats.get("palm_radius")
    if palm_center is None or not radius:
        palm_center, radius = _estimate_palm_robust(mask, workspace=workspace)
        if float(radius) < float(PALM_MIN_RADIUS_PX):
            return None
    palm_center = np.array([float(palm_center[0]), float(palm_center[1])], dtype=np.float64)
    radius = float(radius)
    # A finite distance cannot exceed an axial path across this image.
    # An all-foreground palm mask has no boundary and returns an infinity
    # sentinel; reject it before converting that value into window sizes.
    if not np.isfinite(radius) or radius > (mask.shape[0] - 1) + (mask.shape[1] - 1):
        state.flags["geometry_rejected"] = "invalid_palm_radius"
        return None
    wrist = _estimate_wrist(mask, palm_center, radius)
    if wrist is None:
        up = np.array([-1.0, 0.0], dtype=np.float64)
    else:
        direction = palm_center - wrist
        norm = float(np.linalg.norm(direction))
        up = direction / norm if norm > 1e-06 else np.array([-1.0, 0.0], dtype=np.float64)
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
    # One immutable contour context is shared by collection and identification.
    # It is local to this frame and is never retained in RecognitionResult.
    measurement_context = FG.TipMeasurementContext(contour)
    tips, _profile = _detect_fingertips(
        contour, hand, palm_center, radius, up, closed_candidates=closed_caps,
        measurement_context=measurement_context, workspace=workspace
    )
    # Main geometry does not consume webs, hull, span or display finger lines.
    # Webs and display finger lines are PC-only detailed-render diagnostics.
    gaps = None
    rx0, ry0, rx1, ry1 = _active_rect(hand.shape, state)
    span_x = float(max(1, rx1 - rx0))
    span_y = float(max(1, ry1 - ry0))
    palm_rel = (float((palm_center[0] - rx0) / span_x), float((palm_center[1] - ry0) / span_y))
    result = {
        "palm_center": palm_center,
        "palm_radius": radius,
        "wrist": wrist,
        "up": up,
        "hand_mask": hand,
        "contour": contour,
        "hull": None,
        "solidity": None,
        "angle_span_deg": None,
        "palm_rel": palm_rel,
        "mean_length_ratio": None,
        "tips": tips,
        "gaps": [],
        "fingers": [],
        "finger_count": len(tips),
    }
    states, extensions, angles_used, match_detail, expect_len = _finger_states(
        hand, palm_center, up, radius, tips, gaps, contour=contour, closed_candidates=closed_caps,
        measurement_context=measurement_context
    )
    if states is None:
        state.flags["geometry_rejected"] = match_detail["geometry_fallback_reason"]
        return None
    state.flags.pop("geometry_rejected", None)
    result["_diagnostics_pending"] = True
    result["finger_states"] = states
    result["finger_extensions"] = extensions
    result["finger_ray_angles"] = angles_used
    result["finger_expect_len"] = expect_len
    result["finger_match"] = match_detail
    result["finger_names"] = list(FINGER_NAMES)
    result["finger_text"] = finger_states_text(states)
    return result


def _finger_states(
    mask: np.ndarray,
    palm_center,
    up,
    radius: float,
    tips,
    web_gaps=None,
    contour=None,
    closed_candidates=None,
    measurement_context=None,
):
    """Return geometry-based states or an explicit unavailable result.

    No angle-template, radial-profile or web-gap inference fallback exists.
    The remaining web_gaps argument is a compatibility input and is ignored.
    """
    geometry_failure = None
    if contour is None:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contour = max(contours, key=cv2.contourArea) if contours else None
    measured, geometry_failure = FG.recognize_fingers(
        mask, contour, palm_center, up, radius, tips or [], closed_candidates=closed_candidates,
        measurement_context=measurement_context
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
    # Geometry failure is unavailable recognition, never a fabricated fist.
    reason = geometry_failure or "geometry_unavailable"
    return (None, [0.0] * 5, [0.0] * 5,
            {"method": "geometry_unavailable", "geometry_fallback_reason": reason},
            [0.0] * 5)


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
            "局部长度 "
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
                else "局部指轴定位身份，宽度与细长程度判断伸缩"
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
        hud_lines.append(("wrist not found -> axis = image left", 18, (180, 180, 180)))
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


def prepare_hand_frame(frame):
    """Physical fixed crop; every algorithm plane is CROP_HEIGHT x CROP_WIDTH."""
    if frame.shape != (INPUT_MAX_HEIGHT, INPUT_MAX_WIDTH, 3):
        raise ValueError('Camera adapter must supply 320x180 BGR')
    cropped = np.ascontiguousarray(frame[CROP_Y:CROP_Y + CROP_HEIGHT, CROP_X:CROP_X + CROP_WIDTH])
    return cropped, {
        'input_size': (INPUT_MAX_WIDTH, INPUT_MAX_HEIGHT),
        'cropped_size': (CROP_WIDTH, CROP_HEIGHT),
        'crop_rect': (CROP_X, CROP_Y, CROP_X + CROP_WIDTH, CROP_Y + CROP_HEIGHT),
        'rotation_deg': 0.0, 'coordinate_orientation': 'native_right',
        'input_to_prepared': np.array([[1., 0., -CROP_X], [0., 1., -CROP_Y], [0., 0., 1.]]),
        'active_rect': (0, 0, CROP_WIDTH, CROP_HEIGHT),
    }


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


def recognize_cropped(frame):
    """Core entry: exactly 250x180 BGR, native hand entry from the right.

    No full-camera frame, resize, crop search, rotation or display image is
    allocated here. Local wrist inclination only defines point projections;
    it never rotates or resamples an image or mask.
    """
    if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8 or frame.shape != (CROP_HEIGHT, CROP_WIDTH, 3):
        raise ValueError('Core input must be exactly 250x180 BGR uint8')
    started = perf_counter()
    state = SimpleNamespace(flags={}, dt_workspace=DistanceWorkspace(frame.shape[:2]))
    mask = segment_foreground(frame, state)
    thresholded = perf_counter()
    mask = clean_mask(mask, state)
    cleaned = perf_counter()
    result = _recognize(mask, state)
    finished = perf_counter()
    timings = {'segmentation': (thresholded-started)*1000,
               'cleanup': (cleaned-thresholded)*1000,
               'finger_recognition': (finished-cleaned)*1000,
               'total': (finished-started)*1000}
    return RecognitionResult(frame, result, timings, {}, state.flags)


def recognize(frame):
    """PC adapter: camera resize, physical crop, core call, original-frame preview.

    FPGA camera input is already 320x180. Its crop gate passes x=40..289,
    y=0..179 directly to the 250x180 core; it does not need this PC adapter.
    """
    if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3 or not frame.size:
        raise ValueError('Expected a nonempty BGR uint8 image')
    height, width = frame.shape[:2]
    if width * 9 != height * 16:
        raise ValueError('The fixed camera contract requires a 16:9 input')
    started = perf_counter()
    camera = (frame if (width, height) == (INPUT_MAX_WIDTH, INPUT_MAX_HEIGHT) else
              cv2.resize(frame, (INPUT_MAX_WIDTH, INPUT_MAX_HEIGHT),
                         interpolation=cv2.INTER_AREA if width > INPUT_MAX_WIDTH else cv2.INTER_LINEAR))
    prepared, preprocessing = prepare_hand_frame(camera)
    prepared_at = perf_counter()
    result = recognize_cropped(prepared)
    preprocessing['input_to_display'] = np.diag([width / INPUT_MAX_WIDTH, height / INPUT_MAX_HEIGHT, 1.0])
    result.preprocess = preprocessing
    result.flags['display_frame'] = frame.copy()  # PC preview only, outside the core.
    result.timings_ms['fixed_crop'] = (prepared_at-started)*1000
    result.timings_ms['total'] = (perf_counter()-started)*1000
    return result


def _with_render_diagnostics(result):
    """PC-only geometry; call before transforming to original display pixels."""
    if result is None or not result.get("_diagnostics_pending"):
        return result
    result = dict(result)
    contour, tips = result["contour"], result["tips"]
    palm_center, radius, up = result["palm_center"], result["palm_radius"], result["up"]
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
    result.update(gaps=gaps, fingers=fingers, hull=hull, solidity=solidity,
                  angle_span_deg=angle_span_deg,
                  mean_length_ratio=float(np.mean([f["length_ratio"] for f in fingers])) if fingers else 0.0,
                  _diagnostics_pending=False)
    return result


def render_result(result, detailed=False):
    """Only presentation maps crop coordinates back to the original image."""
    display = result.flags['display_frame']
    translate = np.array([[1., 0., CROP_X], [0., 1., CROP_Y], [0., 0., 1.]])
    matrix = result.preprocess['input_to_display'] @ translate
    recognition = _with_render_diagnostics(result.recognition) if detailed else result.recognition
    drawing = _transform_recognition_for_display(recognition, matrix)
    rejected = bool(result.flags.get('geometry_rejected'))
    canvas = _draw_recognition(display.copy(), drawing, show_summary=not rejected, detailed=detailed)
    if rejected:
        draw_text_lines(canvas, [('手指：无法可靠判断', 26, (70, 100, 255))], (16, 14), panel=True)
    if detailed:
        scale = result.preprocess['input_to_display']
        x0, y0 = int(CROP_X*scale[0,0]), int(CROP_Y*scale[1,1])
        x1, y1 = int((CROP_X+CROP_WIDTH)*scale[0,0])-1, int((CROP_Y+CROP_HEIGHT)*scale[1,1])-1
        cv2.rectangle(canvas, (x0,y0), (x1,y1), COLOR_BOUNDARY, 2)
        draw_text_lines(canvas, [('固定裁剪 250×180；原生右侧进手', 18, COLOR_TEXT)],
                         (12, max(24, canvas.shape[0]-60)), panel=True)
    return canvas


FOREGROUND_MIN_GRAY = 40


_OPEN_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))


_PALM_CLOSE_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
