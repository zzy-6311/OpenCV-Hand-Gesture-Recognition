# -*- coding: utf-8 -*-
"""
模板自标定 —— 用【带真值的素材】重新测出 FINGER_TEMPLATE_DEG / _LEN / FINGER_WEB_X。

用法：
    python calibrate_template.py                 只测量、出报告（不动代码）
    python calibrate_template.py --write         测完直接写回 plugins.py
    python calibrate_template.py --dir data/new --truth _真值.tsv

data/test 是留出测试集，不允许用于模板测量或参数写回。

★★ 核心技巧：用【真值 + 解剖学顺序】给指尖贴标签 ★★
   现在的识别流程里"哪个指尖对应哪根手指"就是难点本身 ✗
   但标定时我们有真值 ✓ 而"手指从拇指到小指沿角度依次排列"是解剖学固定事实 ✓
   所以：把检出的指尖【按角度降序】排、把真值里伸开的手指【按 T→P】排，
         然后**一一对应** ✓ —— 完全绕开分配问题 ✓

⚠️ 掌轴（up）本身可能还带一点系统偏差。脚本会：
   ① 先按"四指（食/中/无名/小）的实测角均值"估出每张图的整体偏移 δ
   ② 报告 δ 的分布 —— 如果 δ 普遍不为 0，说明掌轴还有偏，标定值要按这个修正
"""

import glob
import io
import os
import sys
from pathlib import Path

import cv2
import numpy as np

CODE_DIR = Path(__file__).resolve().parent
BASE = str(CODE_DIR.parent)
if BASE not in sys.path:
    sys.path.insert(0, BASE)
from win_yuan import main as M, plugins as P  # noqa: E402
from win_yuan import evaluation as E  # noqa: E402

FINGERS = "TIMRP"
CN = {"T": "拇指", "I": "食指", "M": "中指", "R": "无名指", "P": "小指"}


def development_path(path):
    """Reject held-out data, including paths through directory links or '..'."""
    candidate = Path(path)
    resolved = (candidate if candidate.is_absolute() else Path(BASE) / candidate).resolve()
    heldout = (Path(BASE) / "data" / "test").resolve()
    if resolved == heldout or heldout in resolved.parents:
        raise ValueError("留出测试集 data/test 不允许用于标定：%s" % path)
    return resolved


def load_truth(path):
    path = str(development_path(path))
    d = {}
    if not os.path.exists(path):
        return d
    for line in io.open(path, encoding="utf-8"):
        s = line.rstrip("\n")
        if not s or s.startswith("#"):
            continue
        a = s.split("\t")
        if len(a) >= 2:
            d[a[0].strip()] = "".join(c for c in a[1].split("#")[0].strip().upper()
                                      if c in FINGERS)
    return d


def measure_one(path, truth_str):
    """量一张图。返回 (每根伸开手指的 (下标, 角度, 长度), 指缝横向位置列表, δ估计)。"""
    path = str(development_path(path))
    img = M.read_image(path)
    if img is None:
        return None
    # Reuse the still-image path, including resize, board crop, scene colour
    # calibration and rotation. Supervised label matching starts only after
    # inference; no filename or truth value enters the recognition pipeline.
    _details, st = E.infer_image(E.build_pipeline(), img, bool(P.ROI_ENABLED))
    rec = st.flags.get("recognize")
    if rec is None:
        return None
    tips = rec["tips"]
    gap_list = rec["gaps"]
    if not tips:
        return None

    # ★ 指尖按角度降序（拇指侧在前）= 解剖学顺序 T→P
    tip_items = []
    for t in tips:
        x = float(t["local"][0])
        y = float(t["local"][1])
        tip_items.append({"angle": float(np.degrees(np.arctan2(x, -y))),
                          "len": float(np.hypot(x, y))})
    tip_items.sort(key=lambda q: -q["angle"])

    # 真值里伸开的手指，按 T→P 顺序
    want = [i for i, c in enumerate(FINGERS) if c in truth_str]
    if len(want) != len(tip_items):
        return None                       # 检出的指尖数和真值对不上，跳过这张

    pairs = []
    for slot, item in zip(want, tip_items):
        pairs.append((slot, item["angle"], item["len"]))

    # δ 估计：用四指（1..4）里伸开的那几根
    deltas = []
    for slot, ang, _ln in pairs:
        if slot != 0:
            deltas.append(ang - float(P.FINGER_TEMPLATE_DEG[slot]))
    delta = float(np.median(deltas)) if deltas else 0.0

    webs = sorted([float(g["local"][0]) for g in gap_list], reverse=True)
    return {"pairs": pairs, "webs": webs, "delta": delta, "n_webs": len(webs)}


def main():
    if "--help" in sys.argv or "-h" in sys.argv:
        print(__doc__)
        print("报告保存在仓库 reports/calibration_report.tsv；--write 才会修改 win_yuan/plugins.py。")
        return 0
    write_back = "--write" in sys.argv
    folder = os.path.join(BASE, "data", "new")
    tname = "_真值.tsv"
    if "--dir" in sys.argv:
        folder = sys.argv[sys.argv.index("--dir") + 1]
        if not os.path.isabs(folder):
            folder = os.path.join(BASE, folder)
    if "--truth" in sys.argv:
        tname = sys.argv[sys.argv.index("--truth") + 1]
    try:
        folder = str(development_path(folder))
        truth_path = development_path(os.path.join(folder, tname))
        truth = load_truth(truth_path)
        # Validate every referenced photo before processing any sample. An external
        # truth file must not use absolute paths or '..' to point into data/test.
        image_paths = {name: str(development_path(os.path.join(folder, name)))
                       for name in truth}
    except ValueError as error:
        print("  %s" % error)
        return 1
    if not truth:
        print("  没读到真值 %s" % os.path.join(folder, tname))
        return 1

    print("=" * 96)
    print("模板自标定  素材目录 %s   真值 %d 条" % (folder, len(truth)))
    print("=" * 96)
    print("  旧模板（在旧素材上标的，带着旧素材掌轴的偏差）：")
    print("    角度 %s" % (P.FINGER_TEMPLATE_DEG,))
    print("    长度 %s" % (P.FINGER_TEMPLATE_LEN,))
    print("    指缝 %s" % (P.FINGER_WEB_X,))
    old_four = [P.FINGER_TEMPLATE_DEG[i] for i in (1, 2, 3, 4)]
    print("    四指角度均值 = %.1f 度  <- 掌轴无偏的话应当接近 0" % (sum(old_four) / 4))
    print()

    rows = []
    for name in sorted(truth):
        p = image_paths[name]
        if not os.path.exists(p):
            continue
        r = measure_one(p, truth[name])
        if r is None:
            print("  %-16s 真值 %-6s  跳过（检出指尖数与真值对不上）" % (name, truth[name]))
            continue
        rows.append((name, truth[name], r))
        got = " ".join("%s(%+.1f°,%.2fR)" % (FINGERS[s], a, l)
                       for s, a, l in r["pairs"])
        print("  %-16s 真值 %-6s δ=%+6.1f  缝%d条   %s"
              % (name, truth[name], r["delta"], r["n_webs"], got))

    if not rows:
        print("\n  没有可用样本")
        return 1

    # ---- 汇总：每根手指的角度/长度 ----
    print()
    print("=" * 96)
    print("① 每根手指实测角度（相对当前掌轴，未做 δ 修正）")
    print("=" * 96)
    print("  %-8s %-6s %-10s %-10s %-10s %-10s %s"
          % ("手指", "样本", "均值", "中位", "标准差", "最小~最大", "旧模板值"))
    new_deg = list(P.FINGER_TEMPLATE_DEG)
    for i in range(5):
        vals = [a for _n, _t, r in rows for s, a, _l in r["pairs"] if s == i]
        if not vals:
            print("  %-8s %-6d  (没有样本，保留旧值 %+.1f)" % (CN[FINGERS[i]], 0, new_deg[i]))
            continue
        v = np.array(vals)
        new_deg[i] = float(np.median(v))
        print("  %-8s %-6d %+-10.1f %+-10.1f %-10.1f %+.1f~%+.1f   %+.1f"
              % (CN[FINGERS[i]], len(v), v.mean(), np.median(v), v.std(),
                 v.min(), v.max(), P.FINGER_TEMPLATE_DEG[i]))

    # ---- δ 分布 ----
    print()
    print("=" * 96)
    print("② 每张图估出来的整体旋转偏差 δ（手掌轴相对模板的偏转）")
    print("=" * 96)
    ds = np.array([r["delta"] for _n, _t, r in rows])
    print("  样本 %d   均值 %+.1f   中位 %+.1f   标准差 %.1f   范围 %+.1f ~ %+.1f"
          % (len(ds), ds.mean(), np.median(ds), ds.std(), ds.min(), ds.max()))
    print("  旧模板四指均值 = %+.1f" % (sum(old_four) / 4))
    print("  [!] δ 普遍不为 0 -> 说明掌轴还有系统偏差，新模板角度应当把 δ 一并吸收进去")
    # ★ 把 δ 吸收进新模板：new = median(实测角) - median(δ)
    d_med = float(np.median(ds))
    print()
    print("  -> 新模板角度 = 实测中位 - δ中位(%+.1f)，让 δ 归零后模板才自洽" % d_med)

    # ---- 汇总：长度 ----
    print()
    print("=" * 96)
    print("③ 每根手指实测长度（× R）")
    print("=" * 96)
    print("  %-8s %-6s %-10s %-10s %-10s %s" % ("手指", "样本", "均值", "中位", "标准差", "旧模板值"))
    new_len = list(P.FINGER_TEMPLATE_LEN)
    for i in range(5):
        vals = [l for _n, _t, r in rows for s, _a, l in r["pairs"] if s == i]
        if not vals:
            print("  %-8s %-6d  (没有样本，保留旧值 %.2f)" % (CN[FINGERS[i]], 0, new_len[i]))
            continue
        v = np.array(vals)
        new_len[i] = float(np.median(v))
        print("  %-8s %-6d %-10.2f %-10.2f %-10.2f %.2f"
              % (CN[FINGERS[i]], len(v), v.mean(), np.median(v), v.std(),
                 P.FINGER_TEMPLATE_LEN[i]))

    # ---- 汇总：指缝 ----
    print()
    print("=" * 96)
    print("④ 指缝的横向位置（× R，掌心为 0，正 = 拇指侧）")
    print("=" * 96)
    print("  只统计【检出正好 4 条缝】的图（顺序天然对应 T-I / I-M / M-R / R-P）")
    four = [r["webs"] for _n, _t, r in rows if r["n_webs"] == 4]
    new_web = list(P.FINGER_WEB_X)
    if four:
        m = np.array(four)
        print("  样本 %d 张" % len(four))
        for k in range(4):
            col = m[:, k]
            new_web[k] = float(np.median(col))
            print("    缝%d  均值 %+.2f  中位 %+.2f  标准差 %.2f  范围 %+.2f ~ %+.2f   旧值 %+.2f"
                  % (k + 1, col.mean(), np.median(col), col.std(), col.min(), col.max(),
                     P.FINGER_WEB_X[k]))
    else:
        print("  没有一张检出正好 4 条缝，保留旧值")

    # ---- 输出 ----
    new_deg_abs = [float(np.median([a for _n, _t, r in rows
                                    for s, a, _l in r["pairs"] if s == i]))
                   if any(s == i for _n, _t, r in rows for s, _a, _l in r["pairs"])
                   else float(P.FINGER_TEMPLATE_DEG[i]) for i in range(5)]
    # 让 δ 归零：整体减去 δ 中位
    new_deg_final = tuple(round(v - d_med, 1) for v in new_deg_abs)

    print()
    print("=" * 96)
    print("★★ 标定结果（可直接粘进 plugins.py）★★")
    print("=" * 96)
    print("FINGER_TEMPLATE_DEG = (%s)" % ", ".join("%.1f" % v for v in new_deg_final))
    print("FINGER_TEMPLATE_LEN = (%s)" % ", ".join("%.2f" % v for v in new_len))
    print("FINGER_WEB_X = (%s)" % ", ".join("%.2f" % v for v in new_web))
    print()
    print("  （角度已整体减去 δ 中位 %.1f 度，让整套模板自洽）" % d_med)

    report_path = Path(BASE) / "reports" / "calibration_report.tsv"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with io.open(report_path, "w", encoding="utf-8") as f:
        f.write("# 模板自标定报告  素材=%s  样本=%d 张\n" % (folder, len(rows)))
        f.write("# 旧角度\t%s\n" % (P.FINGER_TEMPLATE_DEG,))
        f.write("# 新角度\t%s\n" % (new_deg_final,))
        f.write("# 旧长度\t%s\n" % (P.FINGER_TEMPLATE_LEN,))
        f.write("# 新长度\t%s\n" % (tuple(round(v, 2) for v in new_len),))
        f.write("# 旧指缝\t%s\n" % (P.FINGER_WEB_X,))
        f.write("# 新指缝\t%s\n" % (tuple(round(v, 2) for v in new_web),))
        f.write("# δ 中位\t%.1f\n" % d_med)
        f.write("文件\t真值\tdelta\t" + "\t".join("%s角度" % CN[c] for c in FINGERS)
                + "\t" + "\t".join("%s长度" % CN[c] for c in FINGERS) + "\t指缝\n")
        for name, t, r in rows:
            ang = {s: a for s, a, _l in r["pairs"]}
            ln = {s: l for s, _a, l in r["pairs"]}
            f.write("%s\t%s\t%.1f\t" % (name, t, r["delta"]))
            f.write("\t".join("%.1f" % ang[i] if i in ang else "" for i in range(5)))
            f.write("\t")
            f.write("\t".join("%.2f" % ln[i] if i in ln else "" for i in range(5)))
            f.write("\t%s\n" % " ".join("%.2f" % v for v in r["webs"]))
    print("  报告已存 %s" % report_path)

    if write_back:
        p = str(CODE_DIR / "plugins.py")
        src = io.open(p, encoding="utf-8").read()
        import re
        src = re.sub(r"FINGER_TEMPLATE_DEG = \([^)]*\)",
                     "FINGER_TEMPLATE_DEG = (%s)"
                     % ", ".join("%.1f" % v for v in new_deg_final), src)
        src = re.sub(r"FINGER_TEMPLATE_LEN = \([^)]*\)",
                     "FINGER_TEMPLATE_LEN = (%s)"
                     % ", ".join("%.2f" % v for v in new_len), src)
        src = re.sub(r"FINGER_WEB_X = \([^)]*\)",
                     "FINGER_WEB_X = (%s)"
                     % ", ".join("%.2f" % v for v in new_web), src)
        io.open(p, "w", encoding="utf-8").write(src)
        print("  ★ 已写回 plugins.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
