"""Video only: read one frame, recognize it, display it, then read the next."""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import re
import sys
from time import perf_counter

sys.dont_write_bytecode = True

import cv2
import numpy as np

import recognizer as recognizer_core
from recognizer import draw_text_lines, recognize, render_result


DEFAULT_VIDEO = Path(__file__).resolve().parent.parent / "data/test"
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".wmv", ".webm", ".mts"}
WINDOW_NAME = "Finger recognition - video"
FRAME_HOLD_MS = 0  # Additional minimum display time, after this frame is recognized.
NEXT_KEYS = {ord("n"), ord("N"), 0x270000, 65363, 16777236}
PREVIOUS_KEYS = {ord("p"), ord("P"), 0x250000, 65361, 16777234}


def collect_videos(path: Path) -> tuple[list[Path], int]:
    """Sort sibling videos naturally; an explicit file selects the starting clip."""
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Video or folder does not exist: {path}")
    folder = path if path.is_dir() else path.parent
    videos = [item for item in folder.iterdir()
              if item.is_file() and item.suffix.lower() in VIDEO_SUFFIXES]
    if path.is_file() and path not in videos:
        raise ValueError(f"Unsupported video extension: {path.suffix}")
    videos.sort(key=lambda item: [int(part) if part.isdigit() else part.casefold()
                                 for part in re.split(r"(\d+)", item.name)])
    if not videos:
        raise ValueError(f"No videos found in: {folder}")
    return videos, 0 if path.is_dir() else videos.index(path)


def _positive_number(value: float, fallback: float) -> float:
    return value if math.isfinite(value) and value > 0 else fallback


def _player_view(canvas: np.ndarray, videos: list[Path], index: int,
                 frame_no: int, frame_count: int, paused: bool,
                 error: str) -> np.ndarray:
    """Add a small player footer without covering the recognition overlay."""
    height, width = canvas.shape[:2]
    scale = min(1280 / width, 710 / height, 1.0)
    if scale < 1:
        canvas = cv2.resize(canvas, (max(1, round(width * scale)),
                                    max(1, round(height * scale))),
                            interpolation=cv2.INTER_AREA)
    footer_height = 90 if error else 58
    shown = cv2.copyMakeBorder(canvas, 0, footer_height,
                              0, max(0, 860 - canvas.shape[1]),
                              cv2.BORDER_CONSTANT, value=(25, 25, 25))
    top = canvas.shape[0]
    total = str(frame_count) if frame_count else "?"
    status = "已停止" if error else "已暂停" if paused else "播放中"
    lines = [f"{index + 1}/{len(videos)}  {videos[index].name}  "
             f"帧 {frame_no}/{total}  {status}  | 循环当前视频",
             "N/右箭头 下一视频   P/左箭头 上一视频   空格 暂停   R 重播   H 详情   Q/Esc 退出"]
    if error:
        lines.append(f"{error}  | Press N/P/R or Q.")
    draw_text_lines(shown,
                    [(line, 16, (90, 120, 255) if row == 2 else (230, 230, 230))
                     for row, line in enumerate(lines)],
                    (10, top + 6))
    return shown


def play_videos(videos: list[Path], index: int = 0, *, recognize_frame=None,
                render_frame=None, capture_factory=None, clock=perf_counter) -> int:
    """Sequential playback. EOF repeats this clip; only N/P selects another clip.

    The callbacks/factory permit testing playback without a GUI or video decoder.
    Pausing and toggling details reuse the last result; they never run inference.
    Decode/inference errors freeze this clip for inspection and manual navigation.
    """
    if not videos or not 0 <= index < len(videos):
        raise ValueError("The starting video index is invalid.")
    recognize_frame = recognize if recognize_frame is None else recognize_frame
    render_frame = render_result if render_frame is None else render_frame
    capture_factory = cv2.VideoCapture if capture_factory is None else capture_factory
    capture = None
    paused = False
    detailed = False
    open_clip = True
    read_frame = True
    redraw = True
    result = None
    canvas = np.zeros((480, 860, 3), np.uint8)
    frame_no = frame_count = 0
    period_ms = 1000 / 30
    error = ""
    had_error = False

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    try:
        while True:
            started = clock()
            processed_frame = False
            if open_clip:
                if capture is not None:
                    capture.release()
                    capture = None
                frame_no = frame_count = 0
                result = None
                error = ""
                canvas = np.zeros((480, 860, 3), np.uint8)
                try:
                    capture = capture_factory(str(videos[index]))
                    if not capture.isOpened():
                        raise ValueError("Cannot open video")
                    fps = _positive_number(float(capture.get(cv2.CAP_PROP_FPS)), 30.0)
                    period_ms = 1000 / fps
                    frame_count = int(_positive_number(
                        float(capture.get(cv2.CAP_PROP_FRAME_COUNT)), 0.0))
                    read_frame = True  # Switching/restarting shows frame one even while paused.
                except Exception as exc:
                    error = f"Cannot open video: {exc}"
                    had_error = True
                    print(f"{videos[index]}: {error}", file=sys.stderr)
                    read_frame = False
                open_clip = False
                redraw = True

            if read_frame and not error:
                raw = None
                try:
                    ok, raw = capture.read()
                    if not ok or raw is None:
                        if frame_no == 0:
                            raise ValueError("No decodable frames")
                        if frame_count and frame_no < frame_count:
                            raise ValueError(f"Decode stopped early at frame {frame_no}/{frame_count}")
                        # Some containers do not report a frame count. A successful
                        # pass may loop; an empty pass always stops, avoiding a spin.
                        open_clip = True
                        continue
                    frame_no += 1
                    result = recognize_frame(raw)
                    canvas = render_frame(result, detailed=detailed)
                    processed_frame = True
                except Exception as exc:
                    if raw is not None:
                        canvas = raw.copy()
                    result = None
                    error = f"Frame {frame_no}: {exc}"
                    had_error = True
                    print(f"{videos[index]}: {error}", file=sys.stderr)
                read_frame = False
                redraw = True

            if redraw:
                shown = _player_view(canvas, videos, index, frame_no,
                                     frame_count, paused, error)
                cv2.imshow(WINDOW_NAME, shown)
                redraw = False

            # Source FPS is a maximum, not a deadline: never skip a decoded frame
            # to catch up. Slow inference simply makes playback take longer.
            elapsed_ms = (clock() - started) * 1000
            delay_ms = max(1, FRAME_HOLD_MS, math.ceil(period_ms - elapsed_ms)) \
                if processed_frame else 30
            key = cv2.waitKeyEx(delay_ms)
            if key in (27, ord("q"), ord("Q")):
                break
            try:
                if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break

            if key in NEXT_KEYS or key in PREVIOUS_KEYS:
                index = (index + (1 if key in NEXT_KEYS else -1)) % len(videos)
                open_clip = True
            elif key in (ord("r"), ord("R")):
                open_clip = True
            elif key == ord(" "):
                paused = not paused
                redraw = True
                read_frame = not paused and not error
            elif key in (ord("h"), ord("H")):
                detailed = not detailed
                if result is not None and not error:
                    try:
                        canvas = render_frame(result, detailed=detailed)
                    except Exception as exc:
                        error = f"Cannot display result: {exc}"
                        had_error = True
                        print(f"{videos[index]}: {error}", file=sys.stderr)
                redraw = True
            else:
                read_frame = not paused and not error
    finally:
        if capture is not None:
            capture.release()
        try:
            cv2.destroyWindow(WINDOW_NAME)
        except cv2.error:
            pass
    return 1 if had_error else 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="逐帧识别视频；当前视频默认循环，按 N 才播放下一段。")
    parser.add_argument("video", nargs="?", type=Path, default=DEFAULT_VIDEO,
                        help="视频或文件夹路径；默认 ../data/test，文件按自然顺序排列")
    parser.add_argument("--resolution", choices=("320x180", "640x360"),
                        default="320x180",
                        help="启动时选择整条识别通路的分辨率；默认 320x180")
    args = parser.parse_args(argv)
    try:
        # One startup profile; all algorithm stages use this same fixed grid.
        recognizer_core.INPUT_MAX_WIDTH, recognizer_core.INPUT_MAX_HEIGHT = (
            int(value) for value in args.resolution.split("x"))
        videos, index = collect_videos(args.video)
        print(f"识别计算分辨率：{args.resolution}；固定顺时针90度逻辑坐标。")
        print("N/右箭头 下一视频；P/左箭头 上一视频；空格 暂停；R 重播；H 详情；Q/Esc 退出。")
        print(f"当前视频：{videos[index]}（默认循环此视频）")
        return play_videos(videos, index)
    except (OSError, ValueError, cv2.error) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
