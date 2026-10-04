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

# --- 素材目录 ---------------------------------------------------------------
# 所有图片/视频素材都放在这个子目录里（相对本文件所在目录）。
# 代码文件和素材分开，目录看起来清爽，也避免素材文件混进 git 之类的场景。
#
# ⚠️ 改了这个常量，下面 GESTURE_SAMPLES 和 IMAGE_FILES 里的文件名**不用动** ——
#    解析路径时会统一拼上这个前缀（见 resolve_image_paths 的调用处）。
#    写成 "" 就退回"素材和代码放同一个目录"的老做法。
DATA_DIR = "data"

# --- 输入源 -----------------------------------------------------------------
# "image"  = 读磁盘单张图片，循环反复处理（调算法时用这个）
# "camera" = 摄像头实时采集
# "video"  = 读磁盘上的**视频文件**（走和摄像头同一条 cv2.VideoCapture 路径，
#            区别只在"读到结尾怎么办"，见下面的 VIDEO_LOOP）
SOURCE = "image"

# --- 视频文件模式（SOURCE = "video" 时生效）---------------------------------
VIDEO_FILE = "gesture.mp4"             # 相对 DATA_DIR；也可写绝对路径
VIDEO_LOOP = True                      # True = 播完从头循环；False = 播完自动退出
# 视频模式下每帧的等待时间：用 FRAME_DELAY_MS_CAMERA(=1)，让处理速度决定节奏。
# 实测 1920x1440 下整条链约 115~135 ms/帧（约 8 fps）—— 见文件末尾的性能说明。

# --- 手势素材数据集 ---------------------------------------------------------
# 中国式数字手势素材。目前有**三组**（同一个手势在不同背景/光照下各拍一次）：
#   第一组  gesture_<NN>.jpg        白板干净背景、明亮白光、横幅
#   第二组  gesture_<NN>_busy.jpg   斑驳桌面背景、明亮、横幅（手离镜头更近）
#   第三组  gesture_<NN>_s3.jpg     印花床单背景 + 暗暖光、竖幅、手在画面左侧
# 三组放在同一行，方便直接对比"换背景/换光照之后算法有没有受影响"。
# 按 n / p 浏览时会三组依次出现（00 → 00_busy → 00_s3 → 01 → …），正好对着看。
#
# 每项是 (手势标签, 这个手势是怎么比的, 样张文件名元组)。
#
# ★ 标签沿用你拍照时的编号，没改 —— 它**直接当分类结果用**。
#   ⚠️ 标签不一定是数字：不是数字手势的（比如 "ye"）就写个短名字，
#      classify_gesture() 返回什么标签，这里就写什么。
# ★ 第 2 列（手势说明）是我**按照片实际内容**写的。如果和你的本意不符，
#   只改文字就行，文件名不用动。
# ★ 加新样张：在对应手势的元组里追加文件名即可，命名沿用
#   gesture_<编号>[_<场景>].jpg（编号可以是数字，也可以是短名字，如 gesture_ye.jpg）。
#
# ★ 编号 3/33 和 6/66 分别是**同一个数字的两种比法**，编号不同但分类结果相同：
#      3   = 食指 + 中指 + 无名指（拇指收在掌前）
#      33  = 拇指 + 食指 + 中指（最左边那根是拇指）        -> 也判 3
#      6   = 点赞，只伸拇指
#      66  = 拇指 + 小指（"六"的另一种比法）              -> 也判 6
#    分类器的主判据是"伸出手指的根数"，所以两种比法**自然归成一类、不用写特例**。
#    想合并成一行，把 33 / 66 的条目并到 3 / 6 的元组里即可。
#
# （第二组最初把"8"的文件名误写成了 7.jpg，已按你确认改为 gesture_08_busy.jpg。）
#
# ⚠️ 第三组的 gesture_01_s3.jpg（真值 1）已移到 ./不能识别/ 目录，**故意不加载**：
#    它会把"只伸食指"误判成 2，原因是床单上的粉色花纹紧贴食指、颜色和肤色撞车，
#    在掩膜里和手指连成一片、多出一个假指尖。详见 不能识别/说明.txt。
GESTURE_SAMPLES = (
    (0,  "握拳",                  ("gesture_00.jpg", "gesture_00_busy.jpg", "gesture_00_s3.jpg")),
    (1,  "只伸食指",              ("gesture_01.jpg", "gesture_01_busy.jpg")),
    (2,  "食指+中指（剪刀）",      ("gesture_02.jpg", "gesture_02_busy.jpg", "gesture_02_s3.jpg")),
    (3,  "三指（西式：食+中+无名）", ("gesture_03.jpg", "gesture_03_busy.jpg", "gesture_03_s3.jpg")),
    (4,  "四指",                  ("gesture_04.jpg", "gesture_04_busy.jpg", "gesture_04_s3.jpg")),
    (5,  "五指张开",              ("gesture_05.jpg", "gesture_05_busy.jpg", "gesture_05_s3.jpg")),
    (6,  "点赞（只伸拇指）",       ("gesture_06.jpg", "gesture_06_busy.jpg", "gesture_06_s3.jpg")),
    (8,  "手枪（拇指+食指）",      ("gesture_08.jpg", "gesture_08_busy.jpg", "gesture_08_s3.jpg")),
    (33, "三指（中式：拇+食+中）",  ("gesture_33.jpg", "gesture_33_busy.jpg", "gesture_33_s3.jpg")),
    (66, "六（拇指+小指）",        ("gesture_66_s3.jpg",)),
    ("ye", "耶（食指+小指+拇指）",  ("gesture_ye.jpg",)),
)

# 图片模式下要处理的图片（相对本文件所在目录）。按 n / p 在列表里循环切换。
# 直接由上面的手势素材表展平而来 —— 这样只有一处需要维护，
# 加素材只改 GESTURE_SAMPLES。老的 shou1/shou2（布背景、五指张开）
# 放在最后，留着当"难例"做对比。
IMAGE_FILES = (
    [name for _label, _desc, names in GESTURE_SAMPLES for name in names]
    + ["shou1.png", "shou2.png"]
)
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

# --- 图片载入时的降采样（"压到 2K"）------------------------------------------
# 相机/手机的原图往往 4000x3000 以上，整条链直接跑会很慢：
#   实测 4096x3072 要 360~470 ms/帧，而 2560x1440 只要 ~139 ms。
# 这里在**载入的时候压一次**，之后每帧都直接用压好的小图，不会重复计算。
#
# 规则：按 (IMAGE_MAX_WIDTH, IMAGE_MAX_HEIGHT) 装框，**保持宽高比**，只缩不放。
#   4096x3072 -> 1920x1440    （4:3 装进 16:9 的框，受高度限制）
#   2560x1440 -> 原样不动      （本来就在框内）
# 所以老的 shou1/shou2.png 不受影响。
#
# 想改大小就改这两个数（比如两个都乘 2 就回到全分辨率）。
IMAGE_MAX_WIDTH = 2560
IMAGE_MAX_HEIGHT = 1440

DISPLAY_WIDTH = 1280                   # 显示窗口宽度（720p）
DISPLAY_HEIGHT = 720                   # 显示窗口高度（720p）

# --- 显示适配（★ 必须自己做，不能交给 cv2.imshow）----------------------------
# ⚠️ 踩过的坑：窗口客户区固定是 DISPLAY_WIDTH x DISPLAY_HEIGHT，而
#    **cv2.imshow 会把画面拉满整个窗口**。素材宽高比五花八门
#    （4:3 横幅 / 9:16 竖幅 / 16:9），直接丢给 imshow 就会被**非等比拉伸**：
#      实测 810x1440 的竖幅素材在 1280x720 的窗口里被横向拉满，手明显变扁。
#
#    一开始以为 WINDOW_NORMAL 会保持宽高比（它的常量值确实和 WINDOW_KEEPRATIO
#    相同，都是 0），但**后端实际并不保证** —— 实测就是拉满了。
#    所以**不要依赖 imshow 的行为**。
#
# ★ 修法：先自己把画面**等比缩放**到能装进窗口，再**居中补黑边**凑满整块画布，
#    然后才 imshow。这样无论什么宽高比都不会变形。
DISPLAY_FIT = True                     # False = 退回老行为（直接丢给 imshow，会拉伸）
DISPLAY_BG = (0, 0, 0)                 # 补边的颜色（BGR），黑边

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
def fit_within(image: np.ndarray, max_width: int, max_height: int) -> np.ndarray:
    """把图片按比例缩到 (max_width, max_height) 这个框内。只缩不放。

    **保持宽高比**（这一点很重要）：绝对不能用 cv2.resize 直接拉到固定尺寸，
    那样会把 4:3 的图压成 16:9，手部被非等比拉伸，后面所有"按 R 归一化"的
    长度判据都会失真。

    用 INTER_AREA 而不是默认的 INTER_LINEAR：缩小时 INTER_AREA 是抗混叠的，
    等价于先低通再抽取，不会在掩膜上产生锯齿/摩尔纹。
    """
    height, width = image.shape[:2]
    if width <= 0 or height <= 0:
        return image
    scale = min(float(max_width) / float(width), float(max_height) / float(height))
    if scale >= 1.0:                      # 本来就在框内，原样返回
        return image
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)


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
        """载入指定图片（默认当前张），并按 IMAGE_MAX_* 压到 2K。成功返回 True。"""
        target = path if path is not None else self.current_path
        if not target:
            self.message = "没有可用的图片路径"
            return False

        image = cv2.imread(target, cv2.IMREAD_COLOR)
        if image is None or image.size == 0:
            self.message = "读不到图片：%s" % target
            return False

        original_h, original_w = image.shape[:2]
        image = fit_within(image, IMAGE_MAX_WIDTH, IMAGE_MAX_HEIGHT)

        self.frame = image
        self.loaded_path = target
        if (image.shape[1], image.shape[0]) != (original_w, original_h):
            self.message = "已载入 %s  %dx%d -> 压到 %dx%d" % (
                os.path.basename(target), original_w, original_h,
                image.shape[1], image.shape[0])
        else:
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
    """在**右上角**显示采集分辨率、显示分辨率、FPS、帧号与插件数量。

    ⚠️ 原来画在左上角，会和插件里那块"手势分类"面板叠在一起（实测严重重叠）。
       挪到右上角，两边各占一边，互不干扰。
    """
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

    height, width = frame.shape[:2]
    scale = 0.6
    thickness = 1
    line_h = 24
    pad = 10
    text_w = max(cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0][0]
                 for t in lines)
    panel_w = text_w + pad * 2
    panel_h = line_h * len(lines) + pad * 2
    x1 = width - 12
    x0 = max(0, x1 - panel_w)
    y0 = 12
    y1 = min(height, y0 + panel_h)

    # 半透明底衬，白底/花背景上也能看清
    region = frame[y0:y1, x0:x1]
    dark = np.full_like(region, 16)
    cv2.addWeighted(region, 0.38, dark, 0.62, 0.0, dst=region)
    cv2.rectangle(frame, (x0, y0), (x1 - 1, y1 - 1), (95, 95, 95), 1)

    for line_index, text in enumerate(lines):
        color = (0, 140, 255) if text == "PAUSED" else (0, 255, 0)
        baseline = y0 + pad + line_h * line_index + 18
        cv2.putText(frame, text, (x0 + pad, baseline), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
        cv2.putText(frame, text, (x0 + pad, baseline), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, color, thickness, cv2.LINE_AA)

    # 底部按键提示（整宽，单独一条，和上面的面板不冲突）
    hint = "q quit | space pause | 1-6 view | d overlay | m mirror | r ROI | f fullscreen | l plugins"
    cv2.putText(frame, hint, (12, height - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, hint, (12, height - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
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
def _display_frame(frame: np.ndarray) -> np.ndarray:
    """把画面**等比缩放 + 居中补边**，凑成一块 DISPLAY_WIDTH x DISPLAY_HEIGHT 的画布。

    ★ 为什么要自己做（而不是直接 imshow）：cv2.imshow 会把画面**拉满窗口**，
      宽高比不一致就会**非等比拉伸**。实测 810x1440 的竖幅素材在 1280x720 的
      窗口里被横向拉满、手明显变扁 —— 详见 DISPLAY_FIT 那段注释。

    返回的一定是 (DISPLAY_HEIGHT, DISPLAY_WIDTH, 3) 的 BGR 图，
    所以主循环里 imshow 的图像尺寸恒定，窗口不用跟着变。
    """
    if not DISPLAY_FIT:
        return frame
    canvas_h, canvas_w = int(DISPLAY_HEIGHT), int(DISPLAY_WIDTH)
    height, width = frame.shape[:2]
    if height <= 0 or width <= 0:
        return frame
    if frame.ndim == 2:                       # 单通道也允许传进来
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    scale = min(canvas_w / float(width), canvas_h / float(height))
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))
    if (new_w, new_h) == (width, height):
        resized = frame
    else:
        # 缩小用 INTER_AREA（抗混叠），放大用 INTER_LINEAR
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        resized = cv2.resize(frame, (new_w, new_h), interpolation=interp)
    if (new_w, new_h) == (canvas_w, canvas_h):
        return resized                        # 比例刚好一致，不用补边
    canvas = np.full((canvas_h, canvas_w, 3), DISPLAY_BG, dtype=np.uint8)
    x0 = (canvas_w - new_w) // 2
    y0 = (canvas_h - new_h) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    return canvas


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


def _draw_paused_badge(frame: np.ndarray) -> None:
    """在画面上方中央画一个 PAUSED 提示（就地修改）。

    ⚠️ 为什么需要它：暂停时主循环不重绘，画面完全冻结 —— 用户分不清
       "按了空格暂停"和"程序卡死"，而且按 n 换图也看不到任何变化
       （终端提示变了、画面纹丝不动，这是实际踩到的坑）。
       有了这个标记，暂停状态一眼可见。
    """
    text = "PAUSED   (press space to resume)"
    scale, thickness = 0.85, 2
    (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX,
                                          scale, thickness)
    height, width = frame.shape[:2]
    x0 = max(0, (width - text_w) // 2 - 18)
    x1 = min(width, x0 + text_w + 36)
    y0 = 66
    y1 = min(height, y0 + text_h + 28)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return
    region = frame[y0:y1, x0:x1]
    dark = np.full_like(region, 20)
    cv2.addWeighted(region, 0.35, dark, 0.65, 0.0, dst=region)
    cv2.rectangle(frame, (x0, y0), (x1 - 1, y1 - 1), (0, 165, 255), 2)
    cv2.putText(frame, text, (x0 + 18, y1 - 14), cv2.FONT_HERSHEY_SIMPLEX,
                scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, text, (x0 + 18, y1 - 14), cv2.FONT_HERSHEY_SIMPLEX,
                scale, (0, 165, 255), thickness, cv2.LINE_AA)


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
    # ★ 素材统一放在 DATA_DIR 子目录里（见文件开头的配置）。
    #   DATA_DIR 写成 "" 时 os.path.join 会给回 base_dir 本身，行为退回老样子。
    data_dir = os.path.join(base_dir, DATA_DIR) if DATA_DIR else base_dir
    source_kind = SOURCE.strip().lower()
    is_video = (source_kind == "video")

    cap: Optional[cv2.VideoCapture] = None
    image_source: Optional[ImageSource] = None

    # 3) 准备输入源
    if source_kind == "image":
        paths = resolve_image_paths(IMAGE_FILES, data_dir)
        if not paths:
            print("[error] 图片模式下一张可用图片都没有。")
            print("        找素材的目录是：%s" % data_dir)
            print("        （由 main.py 开头的 DATA_DIR 决定，当前是 \"%s\"）" % DATA_DIR)
            print("        请把素材放进去，或把 SOURCE 改成 \"camera\"。")
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
    elif is_video:
        video_path = VIDEO_FILE if os.path.isabs(VIDEO_FILE) \
            else os.path.join(data_dir, VIDEO_FILE)
        if not os.path.exists(video_path):
            print("[error] 视频文件不存在：%s" % video_path)
            print("        改 VIDEO_FILE（它是相对 DATA_DIR=\"%s\" 的），"
                  "或把 SOURCE 换回 \"image\"。" % DATA_DIR)
            return 1
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("[error] 打不开视频：%s（编码可能不被 OpenCV 支持）" % video_path)
            return 1
        video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print("[video] %s  %dx%d  %.1f fps  共 %d 帧"
              % (os.path.basename(video_path),
                 int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                 int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                 cap.get(cv2.CAP_PROP_FPS), video_frames))
        print("[video] %s" % ("播完自动循环" if VIDEO_LOOP else "播完自动退出"))
        frame_delay_ms = FRAME_DELAY_MS_CAMERA
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
    last_displayed: Optional[np.ndarray] = None    # 最近一帧"处理完、还没做显示适配"的画面
    refresh_requested = False              # 暂停时按 n/p/R 也要强制刷新一帧
    print("[main] 按键：q 退出，空格暂停，1~6 切换显示模式，d 叠加，m 镜像，r ROI限制，f 全屏，l 列出插件"
          + ("，n/p 换图，R 重载" if image_source is not None else ""))

    try:
        while True:
            # 5.0 这一轮要不要处理一帧？
            #     没暂停                  -> 正常处理
            #     暂停了、但刚按过 n/p/R   -> 也处理一帧，否则换了图看不到变化
            #     （实际踩过的坑：暂停时按 n，终端文件名变了、画面纹丝不动）
            if (not state.paused) or refresh_requested:
                refresh_requested = False
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
                        # 视频文件读到结尾：要么回到第一帧，要么退出
                        if is_video:
                            if VIDEO_LOOP:
                                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                                time.sleep(0.02)      # 防止 seek 失败时空转
                                continue
                            print("[video] 播放结束")
                            return 0
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
                # 5.5 有可显示的帧就送进窗口。
                #     ★ 送之前先过一遍 _display_frame：等比缩放 + 补黑边，
                #       保证任何宽高比的素材都不会被 imshow 拉变形（见该函数注释）。
                if frame is not None:
                    last_displayed = frame
                    cv2.imshow(WINDOW_NAME, _display_frame(frame))

                # 5.6 帧号加一，然后处理键盘：waitKey 需要每帧调用一次来驱动界面事件。
            #     图片模式下用较大的延时，既避免空转烧 CPU，也保证按键能响应。
                # 只有真正处理过一帧，帧号才加一（暂停时不涨）
                state.index += 1
            else:
                # 5.6b 暂停中：**也要重绘**。
                #      以前暂停就完全不 imshow，结果画面冻结、连"PAUSED"都画不出来，
                #      用户分不清是暂停还是卡死；按 n 换图也只是终端变了、画面不动。
                if last_displayed is not None:
                    # ★ 先做显示适配，再画暂停标记 —— 这样标记是在 1280x720 的
                    #   画布上画的，字号恒定，不会因为原图大小而忽大忽小。
                    paused_canvas = _display_frame(last_displayed)
                    _draw_paused_badge(paused_canvas)      # 就地修改，无返回值
                    cv2.imshow(WINDOW_NAME, paused_canvas)

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
                print("[image] %s%s" % (image_source.message,
                                        "  [暂停中，已强制刷新一帧]" if state.paused else ""))
                refresh_requested = True      # ★ 暂停时也要立刻把新图显示出来
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
