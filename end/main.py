"""Browse photos; run the fixed recognition pipeline only when a photo changes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True

import cv2
import numpy as np

from recognizer import RecognitionResult, recognize


DEFAULT_IMAGE = Path(__file__).resolve().parent.parent / "data/new_camera/shot_01.png"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
WINDOW_NAME = "Finger recognition"
NEXT_KEYS = {ord("n"), ord("N"), ord(" "), 0x270000, 65363, 16777236}
PREVIOUS_KEYS = {ord("p"), ord("P"), 0x250000, 65361, 16777234}


def collect_images(path: Path) -> tuple[list[Path], int]:
    """Naturally sort sibling photos and start at the explicitly selected file."""
    path = path.resolve()
    folder = path if path.is_dir() else path.parent
    if not path.is_dir() and not path.is_file():
        raise FileNotFoundError(f"Image does not exist: {path}")
    photos = [p for p in folder.iterdir()
              if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]
    if path.is_file() and path not in photos:
        photos.append(path)  # Let the decoder handle an explicitly named image.
    photos.sort(key=lambda p: [int(part) if part.isdigit() else part.casefold()
                               for part in re.split(r"(\d+)", p.name)])
    if not photos:
        raise ValueError(f"No photos found in: {folder}")
    return photos, 0 if path.is_dir() else photos.index(path)


def read_image(path: Path) -> np.ndarray:
    """imdecode supports Windows paths containing Chinese characters."""
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot decode image: {path}")
    return image


def draw_result(result: RecognitionResult) -> np.ndarray:
    """Draw the result after inference has completed."""
    canvas = result.image.copy()
    rec = result.recognition
    if rec is not None:
        cv2.drawContours(canvas, [rec["contour"]], -1, (60, 210, 60), 2)
        palm = tuple(np.rint(rec["palm_center"]).astype(int))
        cv2.circle(canvas, palm, int(round(rec["palm_radius"])), (0, 165, 255), 2)
        for tip in rec["tips"]:
            point = tuple(np.rint(tip["point"]).astype(int))
            cv2.circle(canvas, point, 6, (0, 0, 255), -1)
    width = min(canvas.shape[1], 720)
    cv2.rectangle(canvas, (0, 0), (width, min(105, canvas.shape[0])), (30, 30, 30), -1)
    states = result.finger_states
    # Match the left-to-right view of a right palm; the API stays in TIMRP order.
    label = "NO HAND" if states is None else "  ".join(
        f"{finger}:{int(on)}" for finger, on in zip("PRMIT", reversed(states)))
    cv2.putText(canvas, label, (14, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.85,
                (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, f"Inference: {result.timings_ms['total']:.1f} ms", (14, 77),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2, cv2.LINE_AA)
    return canvas


def save_image(path: Path, image: np.ndarray) -> None:
    extension = path.suffix or ".png"
    ok, encoded = cv2.imencode(extension, image)
    if not ok:
        raise ValueError(f"Cannot encode image: {path}")
    encoded.tofile(path)


def process_image(path: Path, roi_enabled: bool) -> RecognitionResult:
    result = recognize(read_image(path), roi_enabled=roi_enabled)
    output = {"file": str(path), **result.to_dict()}
    print(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False))
    return result


def browse_images(photos: list[Path], index: int, result: RecognitionResult,
                  *, roi_enabled: bool) -> None:
    """UI event loop; recognition runs once on entry and once per navigation."""
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    try:
        while True:
            if result is None:
                annotated = np.zeros((400, 720, 3), np.uint8)
                cv2.putText(annotated, "Cannot read/process this photo. Press N or P.",
                            (15, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                            (100, 100, 255), 1, cv2.LINE_AA)
            else:
                annotated = draw_result(result)
            # Scale only the display copy, with room left for the keyboard help.
            height, width = annotated.shape[:2]
            scale = min(1280 / width, 760 / height, 1.0)
            shown = cv2.resize(annotated, (max(1, round(width * scale)),
                                          max(1, round(height * scale))),
                               interpolation=cv2.INTER_AREA) if scale < 1.0 else annotated
            # Keep the help legible even for a very narrow input image.
            shown = cv2.copyMakeBorder(shown, 0, 38, 0, max(0, 620 - shown.shape[1]),
                                       cv2.BORDER_CONSTANT, value=(30, 30, 30))
            help_text = f"{index + 1}/{len(photos)}   N/Right/Space: next   P/Left: prev   Q/Esc: quit"
            cv2.putText(shown, help_text, (10, shown.shape[0] - 13),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow(WINDOW_NAME, shown)
            cv2.setWindowTitle(WINDOW_NAME, f"{index + 1}/{len(photos)} - {photos[index].name} | N: next  P: previous  Q: quit")

            # Waiting and irrelevant keys never re-run the recognizer.
            while True:
                key = cv2.waitKeyEx(50)
                if key in (ord("q"), ord("Q"), 27):
                    return
                try:
                    if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                        return
                except cv2.error:  # The window can disappear between UI events.
                    return
                step = 1 if key in NEXT_KEYS else -1 if key in PREVIOUS_KEYS else 0
                next_index = (index + step) % len(photos)
                if next_index != index:
                    index = next_index
                    try:
                        result = process_image(photos[index], roi_enabled)
                    except (OSError, ValueError, cv2.error) as error:
                        print(f"Error ({photos[index]}): {error}", file=sys.stderr)
                        result = None
                    break
    finally:
        cv2.destroyAllWindows()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="照片浏览与五指识别；仅在切换照片时运行识别")
    parser.add_argument("image", nargs="?", type=Path, default=DEFAULT_IMAGE,
                        help="图片或文件夹路径；默认浏览 ../data/new_camera/ 的照片")
    parser.add_argument("--roi", action="store_true",
                        help="启用原旧照片的居中 ROI；标准摄像头照片默认不启用")
    display = parser.add_mutually_exclusive_group()
    display.add_argument("--show", dest="show", action="store_true",
                         help="浏览照片（默认）：N/右方向键下一张，P/左方向键上一张，Q/Esc退出")
    display.add_argument("--no-show", dest="show", action="store_false",
                         help="不打开照片窗口，仅输出 JSON 或保存标注图")
    parser.set_defaults(show=True)
    parser.add_argument("--output", type=Path, help="保存启动时首张照片的标注图，浏览切换时不覆盖")
    args = parser.parse_args(argv)
    try:
        if args.show or args.image.is_dir():
            photos, index = collect_images(args.image)
        else:
            photos, index = [args.image.resolve()], 0
        result = process_image(photos[index], args.roi)
        if args.output:
            save_image(args.output, draw_result(result))
        if args.show:
            browse_images(photos, index, result, roi_enabled=args.roi)
        return 0
    except (OSError, ValueError, cv2.error) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
