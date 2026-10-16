"""Deterministic still-image evaluation; no drawing or temporal holding.

Labels are used only by the evaluator, never passed into recognition.
Missing/unreadable labeled images remain in the denominator.
No hand is distinct from a recognized closed fist.
"""
from __future__ import annotations

import hashlib
import json
import platform
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

CODE_DIR = Path(__file__).resolve().parent
BASE = CODE_DIR.parent
ARTIFACT_DIR = BASE / "reports"
# Use qualified modules so another version can coexist in this process.
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from win_yuan import main as M, plugins as P
from win_yuan import finger_geometry as FG
FINGERS = "TIMRP"
OLD_TRUTH = {"00": "", "01": "I", "02": "IM", "03": "IMR", "04": "IMRP",
             "05": "TIMRP", "06": "T", "08": "TI", "33": "TIM",
             "66": "TP", "ye": "TIP"}
PROCESSING = ("mirror", "color_threshold", "mask_cleanup", "palm_reconstruct",
              "hand_recognize")


def load_truth(path):
    truth = {}
    for number, line in enumerate(Path(path).read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        fields = line.split("\t")
        if len(fields) < 2:
            raise ValueError("%s:%d: missing TAB" % (path, number))
        name, value = fields[0].strip(), fields[1].split("#", 1)[0].strip().upper()
        if not name or name in truth or any(c not in FINGERS for c in value):
            raise ValueError("%s:%d: invalid or duplicate label" % (path, number))
        if len(set(value)) != len(value):
            raise ValueError("%s:%d: repeated finger" % (path, number))
        truth[name] = "".join(c for c in FINGERS if c in value)
    if not truth:
        raise ValueError("Empty truth file: %s" % path)
    return truth


def samples(group, *, heldout=False):
    data = BASE / "data"
    if group == "test" and not heldout:
        raise ValueError("The held-out test set requires an explicit test evaluation")
    if group in ("new", "camera", "nonstd", "test"):
        folder = {"new": data / "new", "camera": data / "new",
                  "nonstd": data / "不标准", "test": data / "test"}[group]
        filename = "_不标准_真值.tsv" if group == "nonstd" else "_真值.tsv"
        truth = load_truth(folder / filename)
        if any(Path(name).name != name for name in truth):
            raise ValueError("Truth entries must be filenames inside the selected dataset")
        names = set(truth) | set(M.image_filenames(str(folder)))
        roi_enabled = bool(P.ROI_ENABLED)
        return [(folder / name, truth.get(name), roi_enabled)
                for name in sorted(names, key=M.image_sort_key)]
    if group == "old":
        data = data / "old"
        pattern = re.compile(r"gesture_(\d+|ye)(?:_(busy|s3))?\.jpg$")
        result = []
        manifest = json.loads((CODE_DIR / "evaluation_manifest.json").read_text(encoding="utf-8"))
        names = set(manifest["old_filenames"]) | {p.name for p in data.glob("gesture_*.jpg")}
        for path in (data / name for name in sorted(names)):
            match = pattern.fullmatch(path.name)
            result.append((path, OLD_TRUTH.get(match[1]) if match else None, bool(P.ROI_ENABLED)))
        if not result:
            raise ValueError("No old images found")
        return result
    if group == "unlabeled":
        return [(data / "old" / name, None, bool(P.ROI_ENABLED))
                for name in ("shou1.png", "shou2.png", "badmat_01.png")]
    raise ValueError("Unknown group: %s" % group)


def build_pipeline():
    registry = M.build_default_pipeline()
    P.register(registry)
    for name in registry.names():
        registry.enable(name, name in PROCESSING)
    return registry


def reset_image_state():
    P.reset_auto_rotate()
    P.set_roi_anchor(None)
    P.clear_roi_cache()
    P.reset_stability()


def infer_image(registry, raw, roi_enabled):
    reset_image_state()
    started = time.perf_counter()
    raw = M.fit_within(raw, M.IMAGE_MAX_WIDTH, M.IMAGE_MAX_HEIGHT)
    raw, preprocessing = P.prepare_hand_frame(raw)
    rotation = preprocessing["rotation_deg"]
    prepared = time.perf_counter()
    state = M.FrameState(plugins=registry.names(), mirror=M.MIRROR_DEFAULT, show_overlay=False)
    state.raw_frame = raw
    state.raw_size = (raw.shape[1], raw.shape[0])
    state.flags["preprocess"] = preprocessing
    state.flags["roi_enabled"] = bool(roi_enabled)
    state.flags["roi_fallback_enabled"] = bool(roi_enabled)
    output = registry.run(raw.copy(), state, strict=True)
    finished = time.perf_counter()
    if output is None:
        raise RuntimeError("Pipeline dropped the image")
    rec = state.flags.get("recognize")
    stats = state.flags.get("segment_stats") or {}
    prediction = None if rec is None else "".join(
        c for c, on in zip(FINGERS, rec["finger_states"]) if on)
    details = {
        "prediction": prediction,
        "status": "no_hand" if rec is None else "recognized",
        "rotation_deg": float(rotation),
        "preprocess_ms": (prepared - started) * 1000,
        "pipeline_ms": (finished - prepared) * 1000,
        "inference_ms": (finished - started) * 1000,
        "palm_radius": stats.get("palm_radius"),
        "foreground_ratio": stats.get("foreground_ratio"),
        "board": preprocessing["board"],
        "roi_effective": P.roi_enabled_now(state),
    }
    if rec is not None:
        details.update({
            "palm_center": rec["palm_center"], "up": rec["up"], "wrist": rec["wrist"],
            "tip_count": len(rec["tips"]), "gap_count": len(rec["gaps"]),
            "tip_local": [t["local"] for t in rec["tips"]],
            "gap_local": [g["local"] for g in rec["gaps"]],
            "match": rec["finger_match"],
        })
    return details, state


def is_correct(prediction, truth):
    return truth is not None and prediction is not None and prediction == truth


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summarize(rows):
    labeled = [r for r in rows if r["truth"] is not None]
    timings = [r["inference_ms"] for r in rows if "inference_ms" in r]
    available = [r for r in labeled if r["prediction"] is not None]
    per_finger = {}
    for c in FINGERS:
        per_finger[c] = {
            "tp": sum(c in r["truth"] and c in r["prediction"] for r in available),
            "fp": sum(c not in r["truth"] and c in r["prediction"] for r in available),
            "fn": sum(c in r["truth"] and c not in r["prediction"] for r in available),
            "tn": sum(c not in r["truth"] and c not in r["prediction"] for r in available),
            "unavailable": len(labeled) - len(available),
        }
    correct = sum(r["correct"] for r in labeled)
    return {
        "images": len(rows), "labeled": len(labeled), "correct": correct,
        "exact_accuracy": correct / len(labeled) if labeled else None,
        "class_counts": dict(Counter(r["truth"] for r in labeled)),
        "status_counts": dict(Counter(r["status"] for r in rows)),
        "mean_inference_ms": float(np.mean(timings)) if timings else None,
        "p95_inference_ms": float(np.percentile(timings, 95)) if timings else None,
        "per_finger": per_finger,
    }


def evaluate(groups, *, reverse=False, heldout=False):
    groups = tuple(groups)
    if "test" in groups and (not heldout or groups != ("test",)):
        raise ValueError("Evaluate the held-out test set explicitly and separately")
    registry = build_pipeline()
    rows = []
    for group in groups:
        entries = samples(group, heldout=heldout)
        if reverse:
            entries = list(reversed(entries))
        for path, truth, roi in entries:
            row = {"group": group, "file": str(path.relative_to(BASE)), "truth": truth,
                   "roi_enabled": roi, "prediction": None, "status": "error"}
            try:
                row["sha256"] = sha256(path)
                raw = M.read_image(str(path))
                if raw is None:
                    raise ValueError("Unreadable image")
                result, _state = infer_image(registry, raw, roi)
                row.update(result)
            except Exception as error:
                row["error"] = "%s: %s" % (type(error).__name__, error)
            row["correct"] = is_correct(row["prediction"], truth)
            rows.append(row)
    config = {name: getattr(P, name) for name in dir(P)
              if name.isupper() and isinstance(getattr(P, name), (str, int, float, bool, tuple))}
    truth_sources = {
        "new": "data/new/_真值.tsv",
        "camera": "data/new/_真值.tsv",
        "old": "win_yuan/evaluation_manifest.json",
        "nonstd": "data/不标准/_不标准_真值.tsv",
        "test": "data/test/_真值.tsv",
    }
    return {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {"python": platform.python_version(), "opencv": cv2.__version__,
                        "numpy": np.__version__},
        "code_sha256": {name: sha256(BASE / name) for name in
                        ("win_yuan/main.py", "win_yuan/plugins.py", "win_yuan/board_roi.py", "win_yuan/finger_geometry.py", "win_yuan/evaluation.py",
                         "win_yuan/evaluate.py", "win_yuan/regression.py", "win_yuan/calibrate_template.py")},
        "truth_sha256": {truth_sources[group]: sha256(BASE / truth_sources[group])
                         for group in groups if group in truth_sources},
        "config": config,
        "geometry_config": FG.config_dict(),
        "protocol": {"mode": "independent_still_images", "drawing": False,
                     "temporal_gate": False, "mirror": M.MIRROR_DEFAULT,
                     "timing": "resize + auto rotation + processing; excludes disk IO and drawing",
                     "reverse": reverse, "groups": list(groups),
                     "heldout": "test" in groups},
        "summary": {g: summarize([r for r in rows if r["group"] == g]) for g in groups},
        "rows": rows,
    }


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError("Not JSON serializable: %s" % type(value).__name__)


def artifact_path(path):
    """Resolve relative tool output paths under repository reports/, independent of cwd."""
    path = Path(path)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0].lower() == ARTIFACT_DIR.name.lower():
        path = Path(*path.parts[1:])
    return ARTIFACT_DIR / path


def save_report(report, path):
    path = artifact_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=json_default,
                               allow_nan=False), encoding="utf-8")


def compare_reports(before, after):
    old = {(r["group"], r["file"]): r for r in before["rows"]}
    new = {(r["group"], r["file"]): r for r in after["rows"]}
    if old.keys() != new.keys():
        raise ValueError("Cannot compare different sample sets")
    changes = {"improved": [], "regressed": [], "changed_wrong": []}
    for key, row in new.items():
        previous = old[key]
        if any(row.get(k) != previous.get(k) for k in ("truth", "sha256", "roi_enabled")):
            raise ValueError("Sample, truth or ROI changed: %s" % (key,))
        if row["truth"] is None or previous["prediction"] == row["prediction"]:
            continue
        category = "improved" if row["correct"] else (
            "regressed" if previous["correct"] else "changed_wrong")
        changes[category].append({"group": key[0], "file": key[1], "truth": row["truth"],
                                  "before": previous["prediction"], "after": row["prediction"]})
    return changes
