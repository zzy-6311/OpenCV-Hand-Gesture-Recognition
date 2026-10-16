"""Run: python -B -m win_yuan.evaluate new --json current.json

`all` includes data/new development images, old images (including hand backs),
nonstandard images and three explicitly unlabeled diagnostic images.
The held-out test set is excluded from all; select test explicitly to evaluate it.
"""
import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from win_yuan import evaluation as E

P = E.P


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("group", nargs="?", default="new",
                        choices=("camera", "new", "old", "nonstd", "unlabeled", "all", "test"))
    parser.add_argument("--json", dest="report_path",
                        help="Save report; relative paths are resolved inside repository reports/")
    parser.add_argument("--compare", help="Compare with an earlier JSON report; relative paths are inside repository reports/")
    parser.add_argument("--identity-mode", choices=("geometry", "legacy"),
                        help="Select identity algorithm for a controlled comparison")
    parser.add_argument("--reverse", action="store_true", help="Check image-order independence")
    parser.add_argument("--ablate", choices=("none", "no_web", "no_contiguity", "angles_only"),
                        default=None, help="Diagnostic only; never writes production parameters")
    args = parser.parse_args()
    if args.group == "test" and not E.M.image_filenames(str(E.BASE / "data" / "test")):
        parser.error("data/test 当前没有静态测试照片，请先补充独立测试素材；视频不能直接用此照片评测入口评分。")
    if args.group == "test" and (args.identity_mode is not None or args.ablate is not None):
        parser.error("The held-out test set cannot be used for identity-mode or ablation experiments")
    args.ablate = args.ablate or "none"
    if args.identity_mode:
        P.FINGER_IDENTITY_MODE = args.identity_mode
    if args.ablate != "none":
        if args.identity_mode == "geometry":
            parser.error("Weight ablations apply to --identity-mode legacy")
        P.FINGER_IDENTITY_MODE = "legacy"
    groups = ("new", "old", "nonstd", "unlabeled") if args.group == "all" else (args.group,)
    if args.ablate in ("no_web", "angles_only"):
        P.FINGER_W_WEB = 0.0
    if args.ablate in ("no_contiguity", "angles_only"):
        P.FINGER_CONTIGUOUS_PENALTY = 0.0
    try:
        report = E.evaluate(groups, reverse=args.reverse, heldout=args.group == "test")
    except FileNotFoundError as error:
        missing = error.filename or str(error)
        parser.error(f"缺少评测所需的素材或真值文件：{missing}。请准备所选 data 目录中的照片及真值 TSV；随仓库提供的视频不能替代照片评测集。")
    except ValueError as error:
        parser.error(f"评测输入无效：{error}")
    report["protocol"]["ablation"] = args.ablate
    report["protocol"]["identity_mode"] = P.FINGER_IDENTITY_MODE
    if args.compare:
        report["comparison"] = E.compare_reports(
            json.loads(E.artifact_path(args.compare).read_text(encoding="utf-8")), report)
    for row in report["rows"]:
        mark = "?" if row["truth"] is None else ("OK" if row["correct"] else "XX")
        pred = "<unavailable>" if row["prediction"] is None else (row["prediction"] or "<fist>")
        truth = "<unlabeled>" if row["truth"] is None else (row["truth"] or "<fist>")
        print("%s %-9s %-30s truth=%-11s got=%-13s %s" % (
            mark, row["group"], row["file"].split("\\")[-1], truth, pred, row["status"]))
    for group, stats in report["summary"].items():
        accuracy = stats["exact_accuracy"]
        print("%-9s %d/%d  %s  mean_inference_ms=%s" % (
            group, stats["correct"], stats["labeled"],
            "unlabeled" if accuracy is None else "%.1f%%" % (100 * accuracy),
            "n/a" if stats["mean_inference_ms"] is None else "%.1f" % stats["mean_inference_ms"]))
    if "comparison" in report:
        for category, changes in report["comparison"].items():
            print("%s: %d" % (category, len(changes)))
            for change in changes:
                print("  %s %s: %s -> %s (truth %s)" % (
                    change["group"], Path(change["file"]).name, change["before"],
                    change["after"], change["truth"]))
    if args.report_path:
        E.save_report(report, args.report_path)
        print("Report: %s" % E.artifact_path(args.report_path))
    return 2 if any(r["status"] == "error" for r in report["rows"]) else 0


if __name__ == "__main__":
    sys.exit(main())
