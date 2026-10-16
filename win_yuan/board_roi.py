"""Locate a pale, approximately axis-aligned board from image evidence.

The board is a scene constraint, not a skin-colour mask.  Its straight outer
edges are fitted across gaps, and the returned rectangle includes occluding
hands/arms.  No filenames, labels, saved crop coordinates or temporal state are
used.  Uncertain evidence deliberately leaves the original image intact.
"""
from __future__ import annotations

import cv2
import numpy as np


def _fit_edge(points, extent):
    """Deterministic consensus line: dependent coordinate = a * axis + b."""
    points = np.asarray(points, np.float64)
    if len(points) < 12:
        return None
    ids = np.unique(np.linspace(0, len(points) - 1, min(48, len(points))).astype(int))
    selected = points[ids]
    first, second = np.triu_indices(len(selected), 1)
    delta = selected[second] - selected[first]
    usable = np.abs(delta[:, 0]) >= 0.20 * np.ptp(points[:, 0])
    delta, first = delta[usable], first[usable]
    slopes = delta[:, 1] / delta[:, 0]
    intercepts = selected[first, 1] - slopes * selected[first, 0]
    usable = np.abs(slopes) <= np.tan(np.deg2rad(20.0))
    slopes, intercepts = slopes[usable], intercepts[usable]
    if not len(slopes):
        return None
    tolerance = max(1.5, 0.005 * extent)
    best = None
    for start in range(0, len(slopes), 128):
        aa, bb = slopes[start:start + 128], intercepts[start:start + 128]
        error = np.abs(points[None, :, 1] - aa[:, None] * points[None, :, 0] - bb[:, None])
        inside = error <= tolerance
        votes = inside.sum(axis=1)
        # Residual is a tie breaker; it does not reward image-specific positions.
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
    return {"a": float(slope), "b": float(intercept),
            "support": float(keep.mean()), "span": float(span),
            "error": float(np.median(residual[keep])), "tolerance": tolerance}


def _intersection(vertical, horizontal):
    # x = av*y+bv, y = ah*x+bh.
    av, bv = vertical["a"], vertical["b"]
    ah, bh = horizontal["a"], horizontal["b"]
    x = (av * bh + bv) / (1 - av * ah)
    return [x, ah * x + bh]


def _candidate(mask, height, width):
    result = {"rect": (0, 0, width, height), "detected": False,
              "confidence": 0.0, "reason": "no_pale_region", "lines": [], "polygon": []}
    edges = []
    for axis in (0, 1):
        oriented = mask if axis == 0 else mask.T
        rows = np.flatnonzero(np.any(oriented, axis=1))
        low = np.argmax(oriented, axis=1)
        high = oriented.shape[1] - 1 - np.argmax(oriented[:, ::-1], axis=1)
        for name, envelope in zip(("left", "right") if axis == 0 else ("top", "bottom"), (low, high)):
            edge = _fit_edge(np.column_stack((rows, envelope[rows])), oriented.shape[1])
            if edge is None or edge["support"] < 0.28 or edge["span"] < 0.28:
                result["reason"] = "insufficient_straight_edges"
                return result
            edge["name"] = name
            edges.append(edge)
    left, right, top, bottom = edges
    if sum(edge["support"] >= 0.60 and edge["span"] >= 0.70 for edge in edges) < 2:
        result["reason"] = "insufficient_unoccluded_edges"
        return result
    # Opposing sides should be approximately parallel.  Modest perspective is
    # allowed; a heavily tilted board should be re-positioned, not guessed.
    if abs(left["a"] - right["a"]) > 0.15 or abs(top["a"] - bottom["a"]) > 0.15:
        result["reason"] = "perspective_or_conflicting_edges"
        return result
    polygon = np.asarray([_intersection(left, top), _intersection(right, top),
                          _intersection(right, bottom), _intersection(left, bottom)], np.float32)
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
    # Reject image-border-only hypotheses: a uniform room wall is not a located
    # board.  Image-border lines remain valid when other board edges are visible.
    visible = 0
    for edge, extent, along in zip(edges, (sw, sw, sh, sh), (sh, sh, sw, sw)):
        values = edge["a"] * np.array([0.0, along - 1.0]) + edge["b"]
        clipped = np.all(values <= 2) or np.all(values >= extent - 3)
        edge["image_boundary"] = bool(clipped)
        visible += not clipped
    if visible < 2:
        result["reason"] = "board_edges_out_of_view"
        return result
    # No added black/white canvas: callers crop the original pixels so an arm
    # entering the board remains attached to the new image boundary.
    sx, sy = width / sw, height / sh
    original = polygon * np.array([sx, sy], np.float32)
    x0, y0 = np.floor(original.min(axis=0)).astype(int)
    x1, y1 = np.ceil((polygon.max(axis=0) + 1) * [sx, sy]).astype(int)
    rect = (max(0, int(x0)), max(0, int(y0)), min(width, int(x1)), min(height, int(y1)))
    if rect[2] - rect[0] < width * 0.20 or rect[3] - rect[1] < height * 0.20:
        result["reason"] = "board_too_narrow"
        return result
    confidence = float(np.clip(0.6 * min(e["support"] for e in edges) + 0.4 * coverage, 0, 1))
    lines = []
    for index, edge in enumerate(edges):
        ends = {0: (0, 3), 1: (1, 2), 2: (0, 1), 3: (3, 2)}[index]
        lines.append({"name": edge["name"], "points": original[list(ends)].tolist(),
                      "support": edge["support"], "image_boundary": edge["image_boundary"]})
    result.update(rect=rect, detected=True, confidence=confidence,
                  reason="straight_pale_board", lines=lines, polygon=original.tolist(),
                  surface_coverage=coverage)
    return result


def _regions(pale, height, width, restore=None):
    count, labels, stats, _ = cv2.connectedComponentsWithStats(pale, connectivity=8)
    candidates = []
    for label in (1 + np.argsort(stats[1:, cv2.CC_STAT_AREA])[::-1])[:4]:
        if stats[label, cv2.CC_STAT_AREA] < 0.12 * pale.size:
            break
        mask = labels == label
        if restore is not None:
            # Restore the thin strip removed by edge separation without merging
            # the already separated region identities back into one component.
            mask = (cv2.dilate(mask.astype(np.uint8), np.ones((5, 5), np.uint8)) != 0) & (restore != 0)
        candidate = _candidate(mask, height, width)
        candidate["region_area"] = float(stats[label, cv2.CC_STAT_AREA] / pale.size)
        candidates.append(candidate)
    return candidates


def _select(candidates, fallback):
    valid = sorted((r for r in candidates if r["detected"]),
                   key=lambda r: r["confidence"], reverse=True)
    if not valid:
        return candidates[0] if candidates else fallback
    if len(valid) > 1 and valid[1]["region_area"] >= 0.80 * valid[0]["region_area"]:
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
    fallback = {"rect": (0, 0, width, height), "detected": False,
                "confidence": 0.0, "reason": "no_pale_region", "lines": [], "polygon": []}
    if min(height, width) < 32:
        return dict(fallback, reason="image_too_small")
    scale = min(1.0, 640.0 / max(height, width))
    small = cv2.resize(frame, (round(width * scale), round(height * scale)),
                       interpolation=cv2.INTER_AREA) if scale < 1 else frame
    smooth = cv2.GaussianBlur(small, (5, 5), 0)
    gray = cv2.cvtColor(smooth, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(smooth, cv2.COLOR_BGR2HSV)
    threshold, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Relative brightness handles exposure; saturation is a material constraint
    # for a white/grey board, independent of the skin and gesture classifier.
    pale = ((hsv[:, :, 1] <= 45) & (gray >= max(45, threshold))).astype(np.uint8)
    pale = cv2.morphologyEx(pale, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    result = _select(_regions(pale, height, width), fallback)
    if result["detected"] or result["reason"] == "ambiguous_pale_regions":
        return result
    # A neutral background can connect to the board in a brightness mask.
    # Split at measured image edges, then validate each resulting region using
    # exactly the same geometry.  This preserves shadowed board pixels instead
    # of treating an arbitrary higher brightness threshold as a crop boundary.
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
