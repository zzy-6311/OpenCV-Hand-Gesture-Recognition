# -*- coding: utf-8 -*-
"""视频处理骨架：图像/摄像头输入 -> 插件式中间处理 -> 窗口显示。

设计目标
--------
把「输入 / 处理 / 显示」彻底解耦，中间的每一步都是一个可插拔的函数：

    输入（单张图片 或 摄像头 2K）  ->  [插件 1] -> [插件 2] -> ...  ->  显示

* 输入：有两种来源，用下面的 `SOURCE` 切换：
    - ``"image"``（当前默认）：读磁盘上的单张图片，**在循环里反复处理同一张**，
      方便一边看叠加结果一边开关插件、调参数。按 n / p 切换图片，按 R 重载。
    - ``"camera"``：原来的摄像头路径（2K 采集），代码完整保留，随时切回去。
* 处理：默认把**原图（如 2560x1440）整帧**交给插件，不做裁切，
  这样指尖/指缝这类细节不会被提前缩小。
  想回到「先裁成 1280x720 再交给插件」的旧行为，把 `PLUGIN_FULL_RESOLUTION`
  改成 False 即可。
* 显示：窗口固定 1280x720（WINDOW_NORMAL 会把大图按比例缩放进窗口显示）。

插件来源
--------
1. 本文件内置的三个示例插件（镜像、ROI 框、FPS/信息叠加）；
2. 同目录下的 `plugins.py`（如果存在）：里面定义 `register(api)`，用它拿到的
   `api.add(name, func, enabled=...)` 注册任意多个处理函数。

插件函数约定
------------
    def my_step(frame: np.ndarray, state: FrameState) -> np.ndarray | None

* `frame`：**插件工作帧**。默认是原图整帧（当前 2560x1440），
  若 `PLUGIN_FULL_RESOLUTION = False` 则是裁切后的 720p。
  想知道本帧尺寸请读 `state.raw_size`，不要写死。
  注意 `frame` 是一份**拷贝**，就地修改不会污染 `state.raw_frame`；
* `state`：本帧的共享状态（帧号、时间、FPS、原始帧、裁切区域、运行时开关），
  插件之间靠它传递数据和开关，不需要全局变量；
* 返回 `None`：表示「本帧不处理/丢弃」，主循环会跳过后续插件与显示（用于省算力）。

运行方式：执行 main.py。
键位：q 退出，空格暂停/继续，d 显示信息叠加，m 镜像开关，f 全屏开关，
      l 列出插件，r ROI限制 开关；图片模式下另有 n 下一张、p 上一张、R 从磁盘重载。
"""

from __future__ import annotations

import collections
import importlib
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# 配置区
# ---------------------------------------------------------------------------

# --- 输入源 -----------------------------------------------------------------
# "image"  = 读磁盘单张图片，循环反复处理（调算法时用这个）
# "camera" = 原来的摄像头路径
SOURCE = "image"

# 图片模式下要处理的图片（相对本文件所在目录）。按 n / p 在列表里循环切换。
# 列表里不存在的文件会在启动时提示并跳过；一个都没有就直接报错退出。
IMAGE_FILES = [
    "shou1.png",
    "shou2.png",
]
IMAGE_START_INDEX = 0                  # 启动时先用列表里的第几张（从 0 开始）

# --- 插件工作分辨率 ---------------------------------------------------------
# True  = 把原图整帧交给插件（不裁切、不缩放），细节保留最好
# False = 沿用旧行为：先按显示比例裁切并缩放到 DISPLAY_WIDTH x DISPLAY_HEIGHT
PLUGIN_FULL_RESOLUTION = True

# --- 循环节流 ---------------------------------------------------------------
# 图片模式下一帧处理很快，不加节流会空转烧 CPU。这个值同时决定按键响应速度。
FRAME_DELAY_MS_IMAGE = 30              # 图片模式：约 33 次/秒
FRAME_DELAY_MS_CAMERA = 1              # 摄像头模式：只要能泵事件即可

CAMERA_INDEX = 0

CAPTURE_WIDTH = 2560                   # 期望采集宽度（2K）
CAPTURE_HEIGHT = 1440                  # 期望采集高度（2K）
CAPTURE_FPS = 30.0

DISPLAY_WIDTH = 1280                   # 显示窗口宽度（720p）
DISPLAY_HEIGHT = 720                   # 显示窗口高度（720p）
WINDOW_NAME = "video pipeline"

MIRROR_DEFAULT = True                  # 默认水平镜像，符合照镜子式交互习惯
SHOW_OVERLAY_DEFAULT = True            # 默认显示 FPS / 分辨率等信息
MAX_CONSECUTIVE_READ_FAILURES = 30     # 连续读不到帧多少次后退出

# --- 显示模式（数字键切换）--------------------------------------------------
# 每项是 (按键, 模式名, 这一模式下要启用的插件名)。
# 按对应数字键时：列表里出现过的插件中，只启用本模式列出的那些，其余全部关掉。
#
# 六种模式是**递进**的：
#     1 原图       -> 一个处理插件都不开，画面就是原始帧
#     2 掩膜       -> 只开颜色阈值，看到的是原始二值掩膜
#     3 清理后     -> 再开清理层（开运算 + 面积过滤 + 填洞）
#     4 掌心重建   -> 再开重建层（从掌心核心做颜色约束重建）
#     5 识别(掩膜) -> 再开识别骨架，并把结果画在掩膜上
#     6 识别(原图) -> 识别骨架，结果画在原图上（同一份结果，只是底图不同）
#
# 想加新模式：在 plugins.py 注册插件，然后在这里追加一项、分配一个没被占用的
# 按键即可，主循环不用改。
#
# ⚠️ 这里的开关只作用于列表里出现的这些"处理类"插件；
#    ROI 边界、信息叠加这些绘制类插件不受影响，任何模式下都照常工作。
VIEW_MODES = (
    (ord("1"), "原图",       ()),
    (ord("2"), "掩膜",       ("color_threshold",)),
    (ord("3"), "清理后",     ("color_threshold", "mask_cleanup")),
    (ord("4"), "掌心重建",   ("color_threshold", "mask_cleanup", "palm_reconstruct")),
    (ord("5"), "识别(掩膜)", ("color_threshold", "mask_cleanup", "palm_reconstruct",
                              "hand_recognize", "recognize_on_mask")),
    (ord("6"), "识别(原图)", ("color_threshold", "mask_cleanup", "palm_reconstruct",
                              "hand_recognize", "recognize_on_original")),
)

# 默认模式（索引，5 = VIEW_MODES 里的第六项「识别(原图)」）
VIEW_MODE_DEFAULT = 5

# 每个显示模式下"画面里是不是掩膜"。决定要不要做触边检查 + ROI 外擦黑 ——
# 模式 1（原图）和模式 6（原图+标注）都不是掩膜，不能把 ROI 外擦黑。
VIEW_MODE_HAS_MASK = (False, True, True, True, True, False)


# ---------------------------------------------------------------------------
# 每帧共享状态：插件之间靠它传递数据与开关
# ---------------------------------------------------------------------------
@dataclass
class FrameState:
    """一帧在流水线中流转时携带的状态。

    字段说明：
      index         当前帧序号（从 0 开始）
      time          本帧开始处理的时刻，time.perf_counter()
      dt            与上一帧的间隔（秒），第一帧为 0
      fps           平滑后的实时帧率
      raw_frame     采集到的原始帧（通常 2K），只读参考，不要就地改
      raw_size      原始帧尺寸 (宽, 高)
      display_size  显示尺寸 (宽, 高)
      crop          本帧从原始帧裁切出的区域 (x, y, w, h)
      mirror        当前是否水平镜像
      show_overlay  当前是否绘制信息叠加
      paused        主循环是否处于暂停状态
      plugins       已注册插件的名字列表（只读，便于插件了解自身位置）
      flags         插件自由使用的字典，用于自定义开关，例如 flags["roi"] = True
    """

    index: int = 0
    time: float = 0.0
    dt: float = 0.0
    fps: float = 0.0
    raw_frame: Optional[np.ndarray] = None
    raw_size: tuple[int, int] = (0, 0)
    display_size: tuple[int, int] = (DISPLAY_WIDTH, DISPLAY_HEIGHT)
    crop: tuple[int, int, int, int] = (0, 0, DISPLAY_WIDTH, DISPLAY_HEIGHT)
    mirror: bool = MIRROR_DEFAULT
    show_overlay: bool = SHOW_OVERLAY_DEFAULT
    paused: bool = False
    plugins: tuple[str, ...] = ()
    flags: dict[str, Any] = field(default_factory=dict)


# 插件函数签名：输入 (720p 帧, 状态)，输出处理后的帧或 None
PluginFunc = Callable[[np.ndarray, FrameState], Optional[np.ndarray]]


class PluginRegistry:
    """有序插件表：注册、启停、按顺序执行。"""

    def __init__(self) -> None:
        self._items: list[list[Any]] = []          # [name, func, enabled]

    def add(self, name: str, func: PluginFunc, enabled: bool = True) -> None:
        """注册一个处理函数；同名会覆盖原来的函数，保持原有顺序。"""
        for item in self._items:
            if item[0] == name:
                item[1] = func
                item[2] = bool(enabled)
                return
        self._items.append([name, func, bool(enabled)])

    def remove(self, name: str) -> None:
        """按名字移除插件。"""
        self._items = [item for item in self._items if item[0] != name]

    def enable(self, name: str, enabled: bool) -> None:
        """开启或关闭某个插件，运行中可以随时切换。"""
        for item in self._items:
            if item[0] == name:
                item[2] = bool(enabled)
                return

    def names(self) -> tuple[str, ...]:
        """返回全部插件名（含被关闭的），顺序即执行顺序。"""
        return tuple(item[0] for item in self._items)

    def enabled_names(self) -> list[str]:
        """返回当前开启的插件名。"""
        return [item[0] for item in self._items if item[2]]

    def run(self, frame: np.ndarray, state: FrameState) -> Optional[np.ndarray]:
        """按顺序执行所有已启用插件；任一插件返回 None 则整条流水线短路。"""
        current: Optional[np.ndarray] = frame
        for name, func, enabled in self._items:
            if not enabled or current is None:
                continue
            try:
                result = func(current, state)
            except Exception as error:  # 单个插件出错不应该让整个程序崩掉
                print("[plugin:%s] 处理失败：%s: %s" % (name, type(error).__name__, error))
                continue
            if result is None:
                print("[plugin:%s] 丢弃本帧" % name)
                return None
            current = result
        return current


# ---------------------------------------------------------------------------
# 图片输入：读单张图，循环反复交给流水线（用于算法调试）
# ---------------------------------------------------------------------------
class ImageSource:
    """把磁盘上的图片伪装成"视频源"，接口与 ``cv2.VideoCapture`` 对齐。

    设计要点：

    * ``read()`` **每次返回一份拷贝**。因为插件约定里允许"就地修改 frame"，
      如果反复交出同一个数组，第一帧画上去的叠加层会残留在第二帧上、
      镜像插件会把上一次的结果再翻一次 —— 所以必须拷贝。
    * 读不到文件时**保留上一张好图**并记下原因，避免突然黑屏，
      让使用者还有机会按 R 重载或按 n 换一张。
    """

    def __init__(self, paths: list[str], start_index: int = 0) -> None:
        self.paths = list(paths)
        self.index = int(start_index) % len(self.paths) if self.paths else 0
        self.frame: Optional[np.ndarray] = None
        self.loaded_path: Optional[str] = None
        self.message = ""
        if self.paths:
            self.load()

    # --- 查询 ---
    @property
    def current_path(self) -> str:
        return self.paths[self.index] if self.paths else ""

    @property
    def current_name(self) -> str:
        return os.path.basename(self.current_path) if self.current_path else "(无)"

    def position_text(self) -> str:
        """形如 ``shou1.png (1/2)``，给信息面板用。"""
        if not self.paths:
            return "(无图片)"
        return "%s (%d/%d)" % (self.current_name, self.index + 1, len(self.paths))

    # --- 载入 ---
    def load(self, path: Optional[str] = None) -> bool:
        """载入指定图片（默认当前张）。成功返回 True，失败保留原图并返回 False。"""
        target = path if path is not None else self.current_path
        if not target:
            self.message = "没有可用的图片路径"
            return False

        image = cv2.imread(target, cv2.IMREAD_COLOR)
        if image is None or image.size == 0:
            self.message = "读不到图片：%s" % target
            return False

        self.frame = image
        self.loaded_path = target
        self.message = "已载入 %s  %dx%d" % (
            os.path.basename(target), image.shape[1], image.shape[0])
        return True

    # --- 与 VideoCapture 对齐的接口 ---
    def read(self) -> tuple[bool, Optional[np.ndarray]]:
        """返回 ``(ok, frame)``；frame 是拷贝，随便插件怎么改。"""
        if self.frame is None:
            return False, None
        return True, self.frame.copy()

    def release(self) -> None:
        self.frame = None

    # --- 切换 ---
    def step(self, delta: int) -> bool:
        """在图片列表里前后切换（delta = +1 / -1），到头就绕回。"""
        if len(self.paths) <= 1:
            self.message = "列表里只有一张图，没有可切换的" if self.paths else "没有可用图片"
            return False
        self.index = (self.index + int(delta)) % len(self.paths)
        return self.load()

    def reload(self) -> bool:
        """从磁盘重新读当前张（改了图不用重启程序）。"""
        return self.load()


def resolve_image_paths(names: list[str], base_dir: str) -> list[str]:
    """把配置里的文件名解析成绝对路径，并报告缺失的文件。"""
    found: list[str] = []
    missing: list[str] = []
    for name in names:
        path = name if os.path.isabs(name) else os.path.join(base_dir, name)
        if os.path.isfile(path):
            found.append(path)
        else:
            missing.append(name)
    if missing:
        print("[image] 跳过不存在的图片：%s" % "、".join(missing))
    return found


# ---------------------------------------------------------------------------
# 采集：优先 2K，拿不到就退而求其次
# ---------------------------------------------------------------------------
def _try_open(index: int, backend: Optional[int], width: int, height: int,
              fps: float) -> Optional[cv2.VideoCapture]:
    """尝试用指定后端打开摄像头并读取一帧；失败返回 None。"""
    cap = cv2.VideoCapture(index) if backend is None else cv2.VideoCapture(index, backend)
    if cap is None or not cap.isOpened():
        if cap is not None:
            cap.release()
        return None
    try:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
        cap.set(cv2.CAP_PROP_FPS, float(fps))
    except Exception:
        pass
    ok, frame = cap.read()
    if not ok or frame is None or frame.size == 0:
        cap.release()
        return None
    return cap


def open_camera(index: int = CAMERA_INDEX,
                width: int = CAPTURE_WIDTH,
                height: int = CAPTURE_HEIGHT,
                fps: float = CAPTURE_FPS) -> cv2.VideoCapture:
    """打开摄像头并尽量配置成给定分辨率。

    输出：可用的 ``cv2.VideoCapture``。
    失败：所有后端都拿不到有效帧时抛出 ``RuntimeError``。
    """
    backends = [(cv2.CAP_DSHOW, "DirectShow"), (cv2.CAP_MSMF, "MediaFoundation"),
                (None, "默认后端")]
    problems: list[str] = []
    for backend, label in backends:
        cap = _try_open(index, backend, width, height, fps)
        if cap is None:
            problems.append(label)
            continue
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        actual_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        print("[camera] 已打开（%s）实际分辨率 %dx%d @ %.1f fps"
              % (label, actual_w, actual_h, actual_fps))
        if actual_w < width or actual_h < height:
            print("[camera] 注意：摄像头没有提供 %dx%d，实际只有 %dx%d，"
                  "显示会按实际画面裁切" % (width, height, actual_w, actual_h))
        return cap

    raise RuntimeError("摄像头 index=%d 打不开或读不到画面（尝试过后端：%s）"
                       % (index, "、".join(problems)))


def compute_crop(raw_size: tuple[int, int],
                 display_size: tuple[int, int] = (DISPLAY_WIDTH, DISPLAY_HEIGHT),
                 anchor: tuple[float, float] = (0.5, 0.5)) -> tuple[int, int, int, int]:
    """按显示宽高比从原始画面中裁出最大区域，返回 ``(x, y, 宽, 高)``。

    ``anchor`` 是裁切窗口在原始画面中的相对位置，(0.5, 0.5) 表示居中；
    后续做手部跟踪时，把它往左上/右下挪就能让 ROI 跟着手移动。
    """
    raw_w, raw_h = int(raw_size[0]), int(raw_size[1])
    disp_w, disp_h = int(display_size[0]), int(display_size[1])
    if raw_w <= 0 or raw_h <= 0:
        return (0, 0, disp_w, disp_h)
    target_ratio = disp_w / float(disp_h)
    raw_ratio = raw_w / float(raw_h)
    if raw_ratio >= target_ratio:          # 原始画面更宽：裁左右
        crop_w = int(round(raw_h * target_ratio))
        crop_h = raw_h
    else:                                  # 原始画面更高：裁上下
        crop_w = raw_w
        crop_h = int(round(raw_w / target_ratio))
    crop_w = max(1, min(crop_w, raw_w))
    crop_h = max(1, min(crop_h, raw_h))
    x = int(round((raw_w - crop_w) * float(anchor[0])))
    y = int(round((raw_h - crop_h) * float(anchor[1])))
    x = max(0, min(x, raw_w - crop_w))
    y = max(0, min(y, raw_h - crop_h))
    return (x, y, crop_w, crop_h)


def crop_for_display(frame: np.ndarray,
                     display_size: tuple[int, int] = (DISPLAY_WIDTH, DISPLAY_HEIGHT),
                     anchor: tuple[float, float] = (0.5, 0.5)) -> np.ndarray:
    """把原始帧裁切成显示尺寸；宽高比不一致时才做一次缩放兜底。"""
    x, y, crop_w, crop_h = compute_crop(frame.shape[1::-1], display_size, anchor)
    patch = frame[y:y + crop_h, x:x + crop_w]
    disp_w, disp_h = int(display_size[0]), int(display_size[1])
    if patch.shape[1] != disp_w or patch.shape[0] != disp_h:
        patch = cv2.resize(patch, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(patch)


# ---------------------------------------------------------------------------
# 内置示例插件：演示「在这三行里插入任何处理」
# ---------------------------------------------------------------------------
def step_mirror(frame: np.ndarray, state: FrameState) -> np.ndarray:
    """水平镜像，让画面像照镜子。"""
    if state.mirror:
        return cv2.flip(frame, 1)
    return frame


def step_roi_outline(frame: np.ndarray, state: FrameState) -> np.ndarray:
    """画一个居中的 50%x80% 矩形 ROI，仅作示例；state.flags["roi"] 控制开关。"""
    if not state.flags.get("roi", True):
        return frame
    height, width = frame.shape[:2]
    x0 = int(width * 0.25)
    y0 = int(height * 0.10)
    x1 = int(width * 0.75)
    y1 = int(height * 0.90)
    cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 255), 2)
    cv2.putText(frame, "ROI (example)", (x0 + 6, y0 + 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
    return frame


def step_info_overlay(frame: np.ndarray, state: FrameState) -> np.ndarray:
    """在左上角显示采集分辨率、显示分辨率、FPS、帧号与插件数量。"""
    if not state.show_overlay:
        return frame
    lines = [
        "capture %dx%d" % state.raw_size,
        "display %dx%d" % state.display_size,
        "fps %.1f" % state.fps,
        "frame %d" % state.index,
        "plugins %d" % len(state.plugins),
    ]
    if state.paused:
        lines.append("PAUSED")
    height = frame.shape[0]
    for line_index, text in enumerate(lines):
        cv2.putText(frame, text, (12, 26 + line_index * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "q quit / space pause / d overlay / m mirror / f fullscreen",
                (12, height - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (230, 230, 230), 1, cv2.LINE_AA)
    return frame


def build_default_pipeline() -> PluginRegistry:
    """建立默认流水线里的**处理类**插件（先执行，会改变图像数据）。

    这里只放真正处理图像的东西。绘制类插件（ROI 参考框、信息叠加）在
    :func:`add_display_plugins` 里，它们要等到外部处理插件都注册完之后才加，
    这样画上去的标注不会被后续处理（例如二值化）当成图像内容吃掉。
    """
    registry = PluginRegistry()
    registry.add("mirror", step_mirror)
    return registry


def add_display_plugins(registry: PluginRegistry) -> None:
    """注册**绘制类**插件（最后执行，只往画面上叠标注）。

    ⚠️ 必须排在所有处理类插件之后。
    否则它们画出来的东西会被后面的插件当成图像内容处理掉 ——
    表现就是「按 d / 按 r 画面没反应」。这个坑踩过一次。
    """
    # roi_outline 只是画一个参考框，它不裁任何东西。
    # 真正生效的识别区域限制在 plugins.py 的 ROI_RELATIVE 里，
    # 所以这里默认关闭，免得画面上出现两个框分不清。按 r 可以打开作参考。
    registry.add("roi_outline", step_roi_outline, enabled=False)
    registry.add("info_overlay", step_info_overlay)


# ---------------------------------------------------------------------------
# 外部插件：同目录 plugins.py 定义 register(api) 即可接入
# ---------------------------------------------------------------------------
class _PluginApi:
    """交给外部插件的注册接口。"""

    def __init__(self, registry: PluginRegistry) -> None:
        self._registry = registry

    def add(self, name: str, func: PluginFunc, enabled: bool = True) -> None:
        """注册一个处理函数（通常追加在最后）。"""
        self._registry.add(name, func, enabled)

    def remove(self, name: str) -> None:
        """移除某个插件。"""
        self._registry.remove(name)

    def build_pipeline(self) -> PluginRegistry:
        """返回内部插件表，供高级用法直接操作。"""
        return self._registry


def load_external_plugins(registry: PluginRegistry,
                          module_name: str = "plugins") -> bool:
    """导入同目录的 ``plugins.py`` 并调用它的 ``register(api)``。

    返回是否成功接入；文件不存在时返回 False 并给出提示，不影响程序运行。
    """
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError:
        print("[plugins] 未找到 %s.py，当前只运行内置插件（可新建该文件来插入处理）"
              % module_name)
        return False
    except Exception as error:
        print("[plugins] 导入 %s.py 失败：%s: %s" % (module_name, type(error).__name__, error))
        return False

    register = getattr(module, "register", None)
    if not callable(register):
        print("[plugins] %s.py 里没有可调用的 register(api)，已跳过" % module_name)
        return False
    try:
        register(_PluginApi(registry))
    except Exception as error:
        print("[plugins] register(api) 执行失败：%s: %s" % (type(error).__name__, error))
        return False
    print("[plugins] 已接入 %s.py，当前插件：%s" % (module_name, list(registry.names())))
    return True


# ---------------------------------------------------------------------------
# 主循环
# ---------------------------------------------------------------------------
def _prepare_window() -> None:
    """创建窗口并让它正好显示 720p 画面。

    只调用一次 imshow/waitKey 时，WIN32UI 后端还沿用系统记忆的窗口尺寸（本机
    1920x1080）；必须多泵几次事件循环、再显式 resizeWindow，客户区才会稳定在
    1280x720。实测顺序：imshow -> 泵两次 -> resize -> 再泵一次。
    """
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    placeholder = np.zeros((DISPLAY_HEIGHT, DISPLAY_WIDTH, 3), dtype=np.uint8)
    cv2.imshow(WINDOW_NAME, placeholder)
    cv2.waitKey(1)
    cv2.waitKey(1)
    cv2.resizeWindow(WINDOW_NAME, DISPLAY_WIDTH, DISPLAY_HEIGHT)
    cv2.waitKey(1)


def _set_fullscreen(enabled: bool) -> None:
    """切换全屏；退出全屏时恢复成与图像等大的窗口。"""
    if enabled:
        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.waitKey(1)
        return
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
    cv2.waitKey(1)
    cv2.destroyWindow(WINDOW_NAME)
    _prepare_window()


def _select_view_mode(key: int, state: FrameState,
                      registry: PluginRegistry) -> bool:
    """数字键切换显示模式（原图 / 掩膜 / 清理后）。

    返回 True 表示这个按键已经被处理掉了，调用方不用再往下判断。

    做法：把 VIEW_MODES 里出现过的所有插件先全部关掉，再打开本模式列出的那些。
    没在 VIEW_MODES 里出现过的插件（ROI 边界、信息叠加等）一律不动。
    """
    for index, (key_code, label, plugin_names) in enumerate(VIEW_MODES):
        if key != key_code:
            continue

        # 先检查本模式需要的插件是否都注册了，避免切过去却发现什么都没开
        missing = [name for name in plugin_names if name not in registry.names()]
        if missing:
            print("[main] 模式「%s」需要的插件没注册：%s（检查 plugins.py）"
                  % (label, "、".join(missing)))
            return True

        for _, _, names in VIEW_MODES:
            for name in names:
                registry.enable(name, name in plugin_names)
        state.flags["view_mode"] = label
        state.flags["view_mode_has_mask"] = bool(
            VIEW_MODE_HAS_MASK[index] if index < len(VIEW_MODE_HAS_MASK) else bool(plugin_names))
        print("[main] 显示模式 -> %s（启用：%s）"
              % (label, "、".join(plugin_names) if plugin_names else "无，原图透传"))
        return True
    return False


def _handle_key(key: int, state: FrameState, registry: PluginRegistry) -> Optional[str]:
    """处理一个按键，返回 "quit"/"fullscreen" 等动作，未识别时返回 None。"""
    if key in (ord("q"), 27):
        return "quit"
    # 数字键切换显示模式（1 原图 / 2 掩膜 / 3 清理后）。放前面，优先于其他开关。
    if _select_view_mode(key, state, registry):
        return None
    if key == ord(" "):
        state.paused = not state.paused
        print("[main] %s" % ("已暂停" if state.paused else "已继续"))
    elif key == ord("d"):
        state.show_overlay = not state.show_overlay
        print("[main] 信息叠加：%s" % ("开" if state.show_overlay else "关"))
    elif key == ord("m"):
        state.mirror = not state.mirror
        print("[main] 镜像：%s" % ("开" if state.mirror else "关"))
    elif key == ord("r"):
        # 控制**真正生效**的 ROI 限制（实现在 plugins.py 里），
        # 而不是那个"只画不裁"的示例框。按一下就能对比限制/不限制的效果。
        state.flags["roi_enabled"] = not state.flags.get("roi_enabled", True)
        print("[main] ROI 限制：%s" % ("开" if state.flags["roi_enabled"] else "关"))
    elif key == ord("l"):
        print("[main] 插件顺序：%s" % " -> ".join(registry.names()))
        print("[main] 已启用：%s" % ", ".join(registry.enabled_names()))
    elif key == ord("f"):
        return "fullscreen"
    return None


def main() -> int:
    """按 SOURCE 选择输入源，跑插件流水线并显示，直到用户退出。

    返回 0 表示正常退出；图片不可用 / 摄像头不可用 / 读取连续失败时返回 1。

    图片模式（SOURCE = "image"）下会**循环反复处理同一张图**，
    每次循环都从磁盘缓存里拷一份出来，所以插件可以放心就地修改。
    按 n / p 在 IMAGE_FILES 列表里切换，按 R 从磁盘重读当前张。

    摄像头模式（SOURCE = "camera"）下行为与原来一致。
    """
    # 1) 建默认流水线里的处理类插件（镜像），顺序即执行顺序
    registry = build_default_pipeline()
    # 2) 若同目录有 plugins.py，就把它的 register(api) 里注册的处理函数追加进来
    load_external_plugins(registry)
    # 2.5) ★最后才加绘制类插件（ROI 参考框、信息叠加）。
    #      放在外部处理插件之后，它们画的标注才不会被后续处理吃掉。
    add_display_plugins(registry)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    source_kind = SOURCE.strip().lower()

    cap: Optional[cv2.VideoCapture] = None
    image_source: Optional[ImageSource] = None

    # 3) 准备输入源
    if source_kind == "image":
        paths = resolve_image_paths(IMAGE_FILES, base_dir)
        if not paths:
            print("[error] 图片模式下一张可用图片都没有。")
            print("        检查 IMAGE_FILES 配置，或把 SOURCE 改成 \"camera\"。")
            return 1
        image_source = ImageSource(paths, IMAGE_START_INDEX)
        if image_source.frame is None:
            print("[error] %s" % image_source.message)
            return 1
        print("[image] 共 %d 张：%s"
              % (len(paths), "、".join(os.path.basename(p) for p in paths)))
        print("[image] %s" % image_source.message)
        print("[image] 循环处理同一张；n 下一张，p 上一张，R 从磁盘重载")
        frame_delay_ms = FRAME_DELAY_MS_IMAGE
    else:
        try:
            cap = open_camera()
        except RuntimeError as error:
            print("[error] %s" % error)
            return 1
        print("[camera] 循环采集")
        frame_delay_ms = FRAME_DELAY_MS_CAMERA

    # 4) 建立这一轮运行共享的状态对象（帧号、FPS、裁切区、各种开关都在里面）
    state = FrameState(plugins=registry.names())
    state.flags["roi"] = True              # 内置 ROI 示例框默认开启
    # 图片模式默认不做镜像：磁盘上的照片本身就是真实方向，再翻一次会左右颠倒，
    # 调试坐标时容易看错。摄像头模式保留"照镜子"的习惯。
    state.mirror = False if image_source is not None else MIRROR_DEFAULT
    # 启动时按 VIEW_MODE_DEFAULT 把显示模式设好（等价于自动按一下那个数字键）
    if VIEW_MODES:
        _index = min(max(VIEW_MODE_DEFAULT, 0), len(VIEW_MODES) - 1)
        _key = VIEW_MODES[_index][0]
        _bootstrap = FrameState(plugins=registry.names())
        _select_view_mode(_key, _bootstrap, registry)
        state.flags["view_mode"] = _bootstrap.flags.get("view_mode")
        state.flags["view_mode_has_mask"] = _bootstrap.flags.get("view_mode_has_mask")
    # 5) 创建 1280x720 的显示窗口，并先把客户区尺寸校准好
    _prepare_window()
    fullscreen = False
    recent_dt: collections.deque = collections.deque(maxlen=30)   # 用于平滑算 FPS
    last_time = time.perf_counter()
    failures = 0                           # 连续读帧失败计数
    print("[main] 按键：q 退出，空格暂停，1~6 切换显示模式，d 叠加，m 镜像，r ROI限制，f 全屏，l 列出插件"
          + ("，n/p 换图，R 重载" if image_source is not None else ""))

    try:
        while True:
            # 5.0 暂停时跳过取帧和处理，但仍然往下走到 waitKey ——
            #     否则界面不泵事件，按键会彻底没反应（这正是之前"暂停像死机"的原因）。
            #     窗口会保留最后一帧画面。
            if not state.paused:
                # 5.1 取一帧原始图
                if image_source is not None:
                    ok, raw = image_source.read()
                    if not ok or raw is None:
                        # 载入失败在启动时就已经退出了，正常不会走到这里
                        time.sleep(0.05)
                        continue
                else:
                    ok, raw = cap.read()
                    if not ok or raw is None or raw.size == 0:
                        failures += 1
                        if failures >= MAX_CONSECUTIVE_READ_FAILURES:
                            print("[error] 连续 %d 帧读取失败，退出" % failures)
                            return 1
                        continue
                    failures = 0

                # 5.2 计算本帧时间间隔，并用最近 30 帧的平均间隔换算平滑 FPS
                now = time.perf_counter()
                state.dt = 0.0 if state.index == 0 else now - last_time
                last_time = now
                if state.dt > 0.0:
                    recent_dt.append(state.dt)
                    average_dt = sum(recent_dt) / len(recent_dt)
                    state.fps = 1.0 / average_dt if average_dt > 0 else 0.0

                # 5.3 把本帧信息写进共享状态，插件通过 state 读取原图尺寸和裁切区
                state.raw_frame = raw
                state.raw_size = (raw.shape[1], raw.shape[0])

                if PLUGIN_FULL_RESOLUTION:
                    # 整帧交给插件，不裁不缩，指尖/指缝这类细节不被提前缩小。
                    # 必须拷贝：插件允许就地修改 frame，而 raw 同时就是 state.raw_frame，
                    # 直接把同一个数组交出去会被插件改坏。
                    state.crop = (0, 0, state.raw_size[0], state.raw_size[1])
                    frame = raw.copy()
                else:
                    # 旧行为：按 16:9 算出居中裁切区，再压到显示尺寸
                    state.crop = compute_crop(state.raw_size, state.display_size)
                    frame = crop_for_display(raw, state.display_size)

                # 5.4 把工作帧按顺序交给所有插件；
                #     registry.run 返回 None 表示某个插件要求丢弃本帧
                frame = registry.run(frame, state)
                # 5.5 有可显示的帧就送进窗口（窗口是 NORMAL，大图会按比例缩放进 1280x720）
                if frame is not None:
                    cv2.imshow(WINDOW_NAME, frame)

                # 5.6 帧号加一，然后处理键盘：waitKey 需要每帧调用一次来驱动界面事件。
            #     图片模式下用较大的延时，既避免空转烧 CPU，也保证按键能响应。
                # 只有真正处理过一帧，帧号才加一（暂停时不涨）
                state.index += 1

            # 5.7 键盘。这一句在 if 外面，暂停时也要执行，否则界面会僵住。
            key = cv2.waitKey(frame_delay_ms) & 0xFF

            # 5.8 图片模式专有按键：换图 / 从磁盘重载
            if image_source is not None and key in (ord("n"), ord("p"), ord("R")):
                if key == ord("n"):
                    image_source.step(+1)
                elif key == ord("p"):
                    image_source.step(-1)
                else:
                    image_source.reload()
                print("[image] %s" % image_source.message)
                continue

            # 5.9 _handle_key：把按键翻译成开关动作；返回 "quit"/"fullscreen" 时需额外处理
            #     空格键会翻转 state.paused，下一轮循环的 5.0 就会真的停下处理。
            action = _handle_key(key, state, registry)
            if action == "quit":
                return 0
            if action == "fullscreen":
                fullscreen = not fullscreen
                # _set_fullscreen：在全屏和 720p 窗口之间切换
                _set_fullscreen(fullscreen)
    finally:
        # 6) 无论正常退出还是异常退出，都释放输入源并销毁窗口
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
