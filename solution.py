# -*- coding: utf-8 -*-
"""数字 5 手势识别第一阶段（单文件版本）。

本程序面向一名固定用户和一个固定室内场景，采用传统 OpenCV 流程：

    摄像头预热 -> 空背景采集 -> 肤色标定 ->
    实时识别（分割 / 几何分析 / 时间投票）-> 退出

程序只使用 Python 标准库、opencv-python 和 numpy，不使用 CNN、
MediaPipe、深度学习运行时或云端服务。

本文件统一遵循以下约定：
  * 二值掩膜均为单通道 ``uint8``，背景值为 0，手部值为 255；
  * 几何阈值尽量按手掌尺度 R 或 ROI 面积归一化，只有形态学核尺寸和
    少量安全边界使用固定像素；
  * 只分析一个固定 ROI，因此 ROI 外的脸部、衣物和背景不会参与识别。

运行方式：执行 solution.py。
快捷键：q 表示在任意阶段退出，r 表示重新采集背景并标定肤色。
"""

from __future__ import annotations

import collections
import time
from typing import Any, Optional

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# 配置区：所有可调阈值集中在这里，并在行尾注明其含义。
# 标为“×R”的长度已按手掌尺度 R（像素）归一化，降低拍摄距离变化的影响。
# ---------------------------------------------------------------------------

# --- 摄像头与感兴趣区域 ROI -------------------------------------------------
CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30.0
WARMUP_FRAMES = 30                     # 丢弃这些帧，等待自动曝光趋于稳定
MAX_CONSECUTIVE_READ_FAILURES = 30     # 连续读取失败达到此值后终止运行
ROI_RELATIVE = (0.25, 0.06, 0.50, 0.80)  # 固定 ROI：左、上、宽、高（相对比例）
DEBUG_WINDOW_NAME = "gesture5 stage1"
MASK_THUMB_WIDTH = 200                 # 仅用于显示的掩膜缩略图宽度

# --- 背景与肤色标定 ---------------------------------------------------------
BACKGROUND_FRAMES = 30                 # 用于计算固定背景平均值的帧数
SKIN_SAMPLE_FRAMES = 60                # 用于建立肤色模型的采样帧数
SKIN_GRID_SIZE = 3                     # 3×3，共 9 个互不重叠的采样格
SKIN_GRID_RELATIVE = 0.4               # 采样方阵边长与 ROI 短边的比例
SKIN_MIN_SAMPLES = 500                 # 样本不足时要求重新标定
SKIN_COVARIANCE_REG = 40.0             # 添加到协方差矩阵上的正则项
SKIN_MAHALANOBIS_MAX = 9.0             # 马氏距离初始阈值
SKIN_Y_MIN = 40                        # 排除过暗像素
SKIN_Y_MAX = 245                       # 排除严重过曝像素
SKIN_SAT_MIN = 30                      # 排除饱和度过低的像素
SKIN_HSV_PERCENTILE_LOW = 2.0          # 根据样本分位数确定 HSV 下边界
SKIN_HSV_PERCENTILE_HIGH = 98.0

# --- 手部分割 ---------------------------------------------------------------
BLUR_KERNEL_SIZE = 5                   # 高斯预平滑核尺寸
BACKGROUND_DIFF_THRESHOLD = 20         # 多通道 |当前帧-背景帧| 阈值
BACKGROUND_FOREGROUND_DILATE = 3       # 背景前景掩膜的膨胀核尺寸（像素）
BACKGROUND_FOREGROUND_DILATE_ITER = 1
MORPH_OPEN_KERNEL = 3                  # 核必须较小，避免破坏指缝
MORPH_CLOSE_KERNEL = 5                 # 核必须较小，避免把相邻手指粘连
MIN_COMPONENT_AREA_RATIO = 0.005       # 小于 ROI 面积 0.5% 的区域视为噪声
MIN_SUBJECT_AREA_RATIO = 0.03          # 主体候选至少占 ROI 面积 3%
MAX_SUBJECT_AREA_RATIO = 0.70          # 主体候选最多占 ROI 面积 70%
PREVIOUS_CENTROID_TOLERANCE_FACTOR = 6.0  # 跟踪窗口，单位为 sqrt(面积/π)

# --- 掌心与尺度 -------------------------------------------------------------
PALM_CORE_KERNEL = 9                   # 用腐蚀操作分离手掌主体的核尺寸
MIN_PALM_RADIUS_PX = 12                # R 小于此像素值时判为无效
MAX_PALM_RADIUS_ROI_RATIO = 0.35       # R 不得超过 ROI 短边的 35%
WRIST_BAND_RATIO = 0.06                # 用于估计手腕的手部底端带状区域比例
WRIST_MIN_PIXELS = 12                  # 手腕像素少于此值时判为低质量帧
TIP_MIN_DISTANCE_FACTOR = 1.35         # 指尖必须满足 |指尖-C| >= 1.35R
TIP_TOP_FACTOR = 0.8                   # “手掌上方”区域阈值：局部纵坐标达到 0.8R
TIP_MIN_IN_UPPER = 1                   # 至少有一个指尖到达上述区域
TIP_LATERAL_SPAN_FACTOR = 2.0          # 指尖横向跨度至少达到 2.0R
TIP_MIN_PALM_Y = -6.0                  # 指尖局部纵坐标允许的最小值
PALM_CENTER_FACTOR = 0.25              # 排除距离掌心小于 0.25R 的候选点

# --- 前臂截断 ---------------------------------------------------------------
FOREARM_TRUNCATE_FACTOR = 1.5          # 去除掌心 C 下方超过 1.5R 的部分
TRUNCATE_POLYGON_MARGIN = 4.0          # 截断多边形的安全边距（以 R 为单位）

# --- 轮廓、凸包与凸缺陷 -----------------------------------------------------
CONTOUR_MIN_POINTS = 10                # 点数不足时轮廓不可用
CONTOUR_POLY_MIN_POINTS = 16           # 多边形近似后仍需保留足够细节
CONTOUR_TRUNCATED_MIN_AREA_RATIO = 0.01  # 截断前臂后仍需保留的最小面积比例
POLY_EPSILON_RATIO = 0.005             # 多边形近似误差取轮廓周长的 0.5%
SKELETON_MAX_ITERATIONS = 30           # 指尖骨架细化的最大迭代次数
INTERIOR_PEAK_MIN_PIXELS = 4.0         # 可视为内部峰值的最小厚度
INTERIOR_PEAK_LIMIT = 16               # 每帧最多检查的内部峰值数量
THIN_PART_SUPPRESSION = 0.55           # 非极大值抑制半径（×R）
THIN_PART_ENDPOINT_FACTOR = 0.6        # 内部峰值与骨架端点的匹配半径（×R）
DEFECT_MAX_ANGLE_DEG = 100.0           # 角度更大时不视为指缝
DEFECT_MIN_DEPTH_FACTOR = 0.25         # 深度小于 0.25R 时不视为指缝
SHARPNESS_WINDOW = 10                  # 计算曲率时使用的轮廓邻域大小
SHARPNESS_WINDOW_FACTORS = (0.3, 0.6, 1.0)  # 多尺度曲率测量比例
TIP_PEAK_SPAN = 4                      # 寻找指尖峰值时检查的轮廓跨度
TIP_MIN_TURNING_RATIO = 0.15           # 弯曲深度/弦长，低于此值近似直线
TIP_THICKNESS_SAMPLE_FACTOR = 0.2      # 向轮廓内部取厚度样本的位置（×R）
TIP_THICKNESS_FACTOR = 0.45            # 钝厚区域与手掌尺度的比例阈值
TIP_MAX_ANGLE_DEG = 150.0              # 仍可作为指尖的最大轮廓夹角

# --- 指尖聚类与分类 ---------------------------------------------------------
TIP_CLUSTER_DISTANCE_FACTOR = 0.35     # 合并距离小于 0.35R 的候选点
TIP_MERGE_DISTANCE_FACTOR = 0.42       # 距离小于此值的候选视为同一凸起
MAX_TIP_COUNT = 5                      # 指尖数量限制在 0～5
MAX_GAP_COUNT = 4                      # 去重后的指缝数量不得超过 4
MIN_SOLIDITY = 0.55                    # 实心度下限：轮廓面积/凸包面积
MAX_SOLIDITY = 0.90                    # 实心度上限
CANDIDATE_MIN_CONFIDENCE = 0.60        # 单帧候选最低置信度
ENABLE_FINGERTIP_RESIDUAL_CHECK = False  # 实验功能，默认关闭
FINGERTIP_RESIDUAL_FACTOR = 0.50       # 实验性残差容差（×R）

# --- 连续帧确认 -------------------------------------------------------------
HISTORY_LENGTH = 12                    # 单帧识别结果滑动窗口长度
FIVE_FRAMES_TO_CONFIRM = 8             # 12 帧中至少 8 帧为 FIVE 才确认
FIVE_MEAN_CONFIDENCE_TO_CONFIRM = 0.65  # FIVE 帧的平均置信度下限
FIVE_FRAMES_TO_HOLD = 4                # FIVE 帧少于 4 时释放稳定状态
MAX_CONSECUTIVE_UNKNOWN_TO_HOLD = 5    # 连续丢失超过 5 帧时释放状态
LATENCY_WARNING_SECONDS = 3.0          # 确认耗时超过此值时发出警告
LATENCY_TARGET_SECONDS = 1.0           # 目标确认延迟

# --- 调试叠加层颜色（BGR）---------------------------------------------------
COLOR_ROI = (0, 255, 255)
COLOR_CONTOUR = (0, 255, 0)
COLOR_HULL = (255, 128, 0)
COLOR_PALM = (0, 165, 255)
COLOR_TIP = (0, 0, 255)
COLOR_GAP = (255, 0, 255)
COLOR_OK = (0, 255, 0)
COLOR_BAD = (0, 0, 255)
COLOR_TEXT = (255, 255, 255)


# ===========================================================================
# 基础辅助函数
# ===========================================================================
def _blank_mask(shape_hw: tuple[int, int]) -> np.ndarray:
    """创建指定尺寸、全为 0 的单通道 uint8 掩膜。"""
    return np.zeros((int(shape_hw[0]), int(shape_hw[1])), dtype=np.uint8)


def compute_roi(frame_shape: Any) -> tuple[int, int, int, int]:
    """根据画面尺寸计算固定 ROI，返回 ``(左, 上, 宽, 高)``。"""
    height, width = int(frame_shape[0]), int(frame_shape[1])
    left = int(round(ROI_RELATIVE[0] * width))
    top = int(round(ROI_RELATIVE[1] * height))
    roi_w = int(round(ROI_RELATIVE[2] * width))
    roi_h = int(round(ROI_RELATIVE[3] * height))
    left = max(0, min(left, width - 1))
    top = max(0, min(top, height - 1))
    roi_w = max(1, min(roi_w, width - left))
    roi_h = max(1, min(roi_h, height - top))
    return (left, top, roi_w, roi_h)


def _roi_slice(roi: tuple[int, int, int, int]) -> tuple[slice, slice]:
    """把 ROI 矩形转换为可直接索引 numpy 数组的行、列切片。"""
    left, top, roi_w, roi_h = (int(value) for value in roi)
    return (slice(top, top + roi_h), slice(left, left + roi_w))


def _as_uint8_mask(mask: np.ndarray) -> np.ndarray:
    """将任意近似二值数组统一为单通道 0/255 uint8 掩膜。

    二维数组只重新二值化；三通道数组先按 BGR 图像转换为灰度图。
    """
    if mask is None:
        return np.zeros((1, 1), dtype=np.uint8)
    array = mask
    if array.ndim == 3:
        array = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
    else:
        array = np.asarray(array).reshape(array.shape[0], array.shape[1])
    return np.where(array > 0, 255, 0).astype(np.uint8)


def _unit(vector: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    """返回 ``vector`` 的单位向量；向量退化时返回备用方向。"""
    length = float(np.linalg.norm(vector))
    if length < 1e-9:
        return np.asarray(fallback, dtype=np.float64)
    return np.asarray(vector, dtype=np.float64) / length


def _point_angle_deg(point: np.ndarray, forward: np.ndarray, backward: np.ndarray) -> float:
    """计算两条方向在 ``point`` 处的夹角，结果范围为 0～180 度。"""
    first = _unit(forward, np.array([1.0, 0.0]))
    second = _unit(backward, np.array([1.0, 0.0]))
    value = float(np.clip(np.dot(first, second), -1.0, 1.0))
    return float(np.degrees(np.arccos(value)))


def _interior_thickness(mask: np.ndarray) -> np.ndarray:
    """计算掩膜的 L2 距离变换，可将结果理解为内部厚度图。"""
    hand = _as_uint8_mask(mask)
    if int(np.count_nonzero(hand)) == 0:
        return np.zeros(hand.shape, dtype=np.float32)
    return cv2.distanceTransform(hand, cv2.DIST_L2, 5)


def _contour_point(contour: np.ndarray, index: int) -> np.ndarray:
    """以 float64 返回轮廓第 ``index`` 个点，索引自动循环。"""
    count = int(contour.shape[0])
    return contour[int(index) % count].reshape(2).astype(np.float64)


def _to_local(points: np.ndarray, origin: np.ndarray,
              up: np.ndarray, scale: float) -> np.ndarray:
    """将 ROI 点转换到手部局部坐标系并按尺度归一化：x 表示左右方向。"""
    if points.size == 0:
        return np.zeros((0, 2), dtype=np.float64)
    flat = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    delta = flat - np.asarray(origin, dtype=np.float64).reshape(1, 2)
    right = np.array([-up[1], up[0]], dtype=np.float64)
    local_x = delta @ right
    local_y = -(delta @ up)
    return np.stack([local_x / scale, local_y / scale], axis=1)


# ===========================================================================
# 阶段 1：摄像头
# ===========================================================================
def open_camera(index: int = CAMERA_INDEX) -> cv2.VideoCapture:
    """打开并配置 USB 摄像头。

    输入：摄像头编号 ``index``。
    输出：可用的 ``cv2.VideoCapture``；请求的尺寸和帧率只是期望值。
    失败：设备无法打开或不能提供有效帧时，先释放捕获对象，再抛出带原因的
    ``RuntimeError``。
    """
    cap = cv2.VideoCapture(int(index))
    if cap is None or not cap.isOpened():
        if cap is not None:
            cap.release()
        raise RuntimeError(
            "camera index %d could not be opened; check the USB connection or "
            "change CAMERA_INDEX" % int(index)
        )
    try:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(CAMERA_WIDTH))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(CAMERA_HEIGHT))
        cap.set(cv2.CAP_PROP_FPS, float(CAMERA_FPS))
    except Exception:  # pragma: no cover - 某些摄像头后端不支持 set()
        pass

    for _ in range(3):
        ok, frame = cap.read()
        if ok and frame is not None and getattr(frame, "size", 0) > 0:
            return cap
    cap.release()
    raise RuntimeError(
        "camera index %d produced no valid frame; the device may be busy or "
        "blocked by another program" % int(index)
    )


def warm_up_camera(cap: cv2.VideoCapture, frames: int = WARMUP_FRAMES) -> bool:
    """丢弃 ``frames`` 帧，等待自动曝光和自动白平衡趋于稳定。

    只要用户按下 q 或关闭窗口就返回 False，本函数不主动抛出异常。这里暂时
    容忍读取失败，后续阶段还会执行严格的帧读取检查。
    """
    for _ in range(int(frames)):
        ok, frame = cap.read()
        if ok and frame is not None and getattr(frame, "size", 0) > 0:
            preview = frame.copy()
            cv2.putText(preview, "warming up camera ...", (12, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_TEXT, 2)
            _draw_hint_banner(preview, "STAGE 1/3  camera warm-up",
                              "wait a moment, then press q to quit")
            cv2.imshow(DEBUG_WINDOW_NAME, preview)
        if _handle_ui_event() == "quit":
            return False
    return True


# ===========================================================================
# 阶段 2：固定背景
# ===========================================================================
def capture_background(
    cap: cv2.VideoCapture,
    roi: tuple[int, int, int, int],
    sample_frames: int = BACKGROUND_FRAMES,
) -> Optional[np.ndarray]:
    """平均无手状态下的 ``sample_frames`` 个 ROI 帧，得到固定背景。

    输入：已打开的捕获对象、ROI 矩形和采样帧数。
    输出：形状为 ``(roi_h, roi_w, 3)`` 的 uint8 BGR 背景参考图。
    失败：用户按 q 或完全未读到帧时返回 ``None``，绝不返回伪造背景。
    """
    rows, cols = _roi_slice(roi)
    left, top, roi_w, roi_h = (int(value) for value in roi)
    accumulator: Optional[np.ndarray] = None
    collected = 0
    for _ in range(int(sample_frames)):
        ok, frame = cap.read()
        if not ok or frame is None or getattr(frame, "size", 0) == 0:
            continue
        patch = frame[rows, cols]
        if patch.shape[0] != roi_h or patch.shape[1] != roi_w:
            continue
        patch_f = patch.astype(np.float32)
        accumulator = patch_f.copy() if accumulator is None else accumulator + patch_f
        collected += 1

        preview = frame.copy()
        cv2.rectangle(preview, (left, top), (left + roi_w - 1, top + roi_h - 1),
                      COLOR_ROI, 2)
        cv2.putText(preview, "keep hands OUT of the ROI: %d/%d"
                    % (collected, int(sample_frames)), (12, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_ROI, 2)
        _draw_hint_banner(preview, "STAGE 2/3  background capture %d/%d"
                          % (collected, int(sample_frames)),
                          "keep hands out of the yellow ROI, then press q to quit",
                          COLOR_ROI)
        cv2.imshow(DEBUG_WINDOW_NAME, preview)
        if _handle_ui_event() == "quit":
            return None

    if accumulator is None or collected <= 0:
        return None
    return np.clip(accumulator / float(collected), 0.0, 255.0).astype(np.uint8)


# ===========================================================================
# 阶段 3：固定用户肤色模型
# ===========================================================================
def _skin_grid_cells(roi: tuple[int, int, int, int]) -> list[tuple[int, int, int, int]]:
    """返回九个互不重叠、采用 ROI 局部坐标的肤色采样矩形。"""
    _, _, roi_w, roi_h = (int(value) for value in roi)
    side = max(SKIN_GRID_SIZE * 8, int(round(SKIN_GRID_RELATIVE * min(roi_w, roi_h))))
    side = min(side, roi_w, roi_h)
    cell = max(1, side // SKIN_GRID_SIZE)
    square = cell * SKIN_GRID_SIZE
    left = max(0, (roi_w - square) // 2)
    top = max(0, (roi_h - square) // 2)
    cells: list[tuple[int, int, int, int]] = []
    for row in range(SKIN_GRID_SIZE):
        for col in range(SKIN_GRID_SIZE):
            cells.append((left + col * cell, top + row * cell, cell, cell))
    return cells


def _skin_model_from_samples(
    chroma: np.ndarray,
    hsv_samples: np.ndarray,
) -> Optional[dict[str, np.ndarray | float]]:
    """根据采集到的 Cr/Cb 与 HSV 样本建立肤色模型字典。"""
    if chroma is None or chroma.shape[0] < SKIN_MIN_SAMPLES:
        return None
    mean = chroma.mean(axis=0)
    centred = chroma - mean.reshape(1, 2)
    covariance = np.cov(centred, rowvar=False)
    covariance = np.asarray(covariance, dtype=np.float64).reshape(2, 2)
    covariance = covariance + np.eye(2, dtype=np.float64) * SKIN_COVARIANCE_REG
    if not np.all(np.isfinite(covariance)):
        return None
    try:
        inverse = np.linalg.inv(covariance)
    except np.linalg.LinAlgError:
        return None
    if not np.all(np.isfinite(inverse)):
        return None

    hue = hsv_samples[:, 0].astype(np.float64)
    circular = np.concatenate([hue, hue + 180.0, hue - 180.0])
    h_low, h_high = np.percentile(circular,
                                  [SKIN_HSV_PERCENTILE_LOW, SKIN_HSV_PERCENTILE_HIGH])
    s_low, s_high = np.percentile(hsv_samples[:, 1].astype(np.float64),
                                  [SKIN_HSV_PERCENTILE_LOW, SKIN_HSV_PERCENTILE_HIGH])
    v_low, v_high = np.percentile(hsv_samples[:, 2].astype(np.float64),
                                  [SKIN_HSV_PERCENTILE_LOW, SKIN_HSV_PERCENTILE_HIGH])
    h_low = float(h_low) % 180.0
    h_high = float(h_high) % 180.0
    if float(h_high - h_low) >= 178.0:  # 色相范围退化时使用完整色相环
        h_low, h_high = 0.0, 179.0
    return {
        "cr_cb_mean": mean.astype(np.float64),
        "cr_cb_inv_cov": inverse.astype(np.float64),
        "h_low": h_low,
        "h_high": h_high,
        "s_low": float(max(SKIN_SAT_MIN, s_low)),
        "s_high": float(min(255.0, s_high)),
        "v_low": float(max(SKIN_Y_MIN, v_low)),
        "v_high": float(min(255.0, v_high)),
        "sample_count": float(chroma.shape[0]),
    }


def calibrate_skin(
    cap: cv2.VideoCapture,
    roi: tuple[int, int, int, int],
    sample_frames: int = SKIN_SAMPLE_FRAMES,
) -> Optional[dict[str, np.ndarray | float]]:
    """通过 3×3 采样格采集固定用户的掌心颜色。

    输入：已打开的捕获对象、ROI 矩形和采样帧数；只有九个采样格中的像素
    会进入模型。
    输出：包含 ``cr_cb_mean``（float64[2]）、``cr_cb_inv_cov``
    （float64[2, 2]）以及 HSV 范围字段的字典。
    失败：用户退出、没有有效帧、样本不足或协方差无效时返回 ``None``；
    调用方应提示重新标定，不得继续识别。
    """
    left, top, roi_w, roi_h = (int(value) for value in roi)
    rows, cols = _roi_slice(roi)
    cells = _skin_grid_cells(roi)
    chroma_samples: list[np.ndarray] = []
    hsv_sample_list: list[np.ndarray] = []
    collected = 0

    for _ in range(int(sample_frames)):
        ok, frame = cap.read()
        if not ok or frame is None or getattr(frame, "size", 0) == 0:
            continue
        patch = frame[rows, cols]
        if patch.shape[0] != roi_h or patch.shape[1] != roi_w:
            continue

        ycrcb = cv2.cvtColor(patch, cv2.COLOR_BGR2YCrCb)
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        for (cell_x, cell_y, cell_w, cell_h) in cells:
            cell_ycrcb = ycrcb[cell_y:cell_y + cell_h, cell_x:cell_x + cell_w]
            cell_hsv = hsv[cell_y:cell_y + cell_h, cell_x:cell_x + cell_w]
            if cell_ycrcb.size == 0:
                continue
            chroma_samples.append(cell_ycrcb.reshape(-1, 3)[:, 1:3].astype(np.float64))
            hsv_sample_list.append(cell_hsv.reshape(-1, 3).astype(np.float64))
        collected += 1

        preview = frame.copy()
        cv2.rectangle(preview, (left, top), (left + roi_w - 1, top + roi_h - 1),
                      COLOR_ROI, 2)
        for (cell_x, cell_y, cell_w, cell_h) in cells:
            cv2.rectangle(preview, (left + cell_x, top + cell_y),
                          (left + cell_x + cell_w - 1, top + cell_y + cell_h - 1),
                          COLOR_OK, 1)
        cv2.putText(preview, "cover the 9 cells with your PALM: %d/%d"
                    % (collected, int(sample_frames)), (12, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_OK, 2)
        _draw_hint_banner(preview, "STAGE 3/3  skin calibration %d/%d"
                          % (collected, int(sample_frames)),
                          "cover all 9 green cells with your palm, then press q",
                          COLOR_OK)
        cv2.imshow(DEBUG_WINDOW_NAME, preview)
        if _handle_ui_event() == "quit":
            return None

    if not chroma_samples or not hsv_sample_list:
        return None
    return _skin_model_from_samples(np.vstack(chroma_samples), np.vstack(hsv_sample_list))


# ===========================================================================
# 阶段 4a：手部分割
# ===========================================================================
def _skin_candidate_mask(
    roi_bgr: np.ndarray,
    skin_model: dict[str, np.ndarray | float],
) -> Optional[np.ndarray]:
    """根据固定用户肤色模型生成 ROI 的 0/255 肤色候选掩膜。"""
    try:
        mean = np.asarray(skin_model["cr_cb_mean"], dtype=np.float64).reshape(2)
        inverse = np.asarray(skin_model["cr_cb_inv_cov"], dtype=np.float64).reshape(2, 2)
    except (KeyError, ValueError, TypeError):
        return None
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(inverse)):
        return None

    ycrcb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2YCrCb).astype(np.float64)
    diff = ycrcb[:, :, 1:3] - mean.reshape(1, 1, 2)
    mahalanobis = np.einsum("...i,ij,...j->...", diff, inverse, diff)
    distance = np.sqrt(np.maximum(mahalanobis, 0.0))
    candidate = (distance <= float(SKIN_MAHALANOBIS_MAX)).astype(np.uint8) * 255

    luma = ycrcb[:, :, 0]
    luma_ok = (luma >= float(skin_model.get("v_low", SKIN_Y_MIN))) & \
              (luma <= float(skin_model.get("v_high", SKIN_Y_MAX)))

    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0].astype(np.float64)
    h_low = float(skin_model.get("h_low", 0.0))
    h_high = float(skin_model.get("h_high", 179.0))
    if h_low <= h_high:
        hue_ok = (hue >= h_low - 1.0) & (hue <= h_high + 1.0)
    else:  # 色相范围跨过 OpenCV HSV 的 0/179 分界线
        hue_ok = (hue >= h_low - 1.0) | (hue <= h_high + 1.0)
    saturation = hsv[:, :, 1].astype(np.float64)
    saturation_ok = (saturation >= float(skin_model.get("s_low", SKIN_SAT_MIN))) & \
                    (saturation <= float(skin_model.get("s_high", 255.0)))

    candidate[(~luma_ok) | (~saturation_ok) | (~hue_ok)] = 0
    candidate = cv2.morphologyEx(
        candidate, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                  (MORPH_OPEN_KERNEL, MORPH_OPEN_KERNEL)))
    return _as_uint8_mask(candidate)


def _background_foreground_mask(
    roi_bgr: np.ndarray,
    background_bgr: np.ndarray,
) -> np.ndarray:
    """生成固定背景差分前景掩膜，并进行一次轻微膨胀。"""
    if background_bgr is None or background_bgr.shape != roi_bgr.shape:
        return np.zeros(roi_bgr.shape[:2], dtype=np.uint8)
    reference = background_bgr.astype(np.int16)
    current = roi_bgr.astype(np.int16)
    channel_max = np.max(np.abs(current - reference), axis=2)
    foreground = (channel_max >= BACKGROUND_DIFF_THRESHOLD).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (BACKGROUND_FOREGROUND_DILATE, BACKGROUND_FOREGROUND_DILATE))
    return _as_uint8_mask(cv2.dilate(foreground, kernel,
                                     iterations=BACKGROUND_FOREGROUND_DILATE_ITER))


def segment_hand(
    roi_bgr: np.ndarray,
    background_bgr: np.ndarray,
    skin_model: dict[str, np.ndarray | float],
    previous_centroid: Optional[tuple[int, int]] = None,
) -> np.ndarray:
    """为单个 ROI 帧生成清理后的单主体手部二值掩膜。

    输入：ROI 的 BGR 图像、固定背景参考、固定用户肤色模型，以及可选的
    上一帧质心（仅作为跟踪提示）。
    输出：单通道 uint8 掩膜，背景为 0，手部为 255。
    失败：没有合法主体时返回同尺寸全黑掩膜，不因普通识别失败抛出异常。
    """
    if roi_bgr is None or getattr(roi_bgr, "size", 0) == 0:
        return _blank_mask((1, 1))
    roi_h, roi_w = roi_bgr.shape[:2]
    roi_area = float(roi_h * roi_w)

    smoothed = cv2.GaussianBlur(roi_bgr, (BLUR_KERNEL_SIZE, BLUR_KERNEL_SIZE), 0)
    skin = _skin_candidate_mask(smoothed, skin_model)
    background_foreground = _background_foreground_mask(smoothed, background_bgr)

    if skin is None:
        combined = np.zeros((roi_h, roi_w), dtype=np.uint8)
    else:
        combined = _as_uint8_mask(cv2.bitwise_and(skin, background_foreground))

    combined = cv2.morphologyEx(
        combined, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                  (MORPH_OPEN_KERNEL, MORPH_OPEN_KERNEL)))
    combined = cv2.morphologyEx(
        combined, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                  (MORPH_CLOSE_KERNEL, MORPH_CLOSE_KERNEL)))

    count, labels, stats, centroids = cv2.connectedComponentsWithStats(combined, 8)
    minimum_area = MIN_COMPONENT_AREA_RATIO * roi_area
    best_label = -1
    best_area = -1.0
    best_distance = float("inf")
    for label in range(1, int(count)):
        area = float(stats[label, cv2.CC_STAT_AREA])
        if area < minimum_area:
            continue
        if area < MIN_SUBJECT_AREA_RATIO * roi_area or area > MAX_SUBJECT_AREA_RATIO * roi_area:
            continue
        cx, cy = float(centroids[label][0]), float(centroids[label][1])
        distance = 0.0
        if previous_centroid is not None:
            distance = float(np.hypot(cx - float(previous_centroid[0]),
                                      cy - float(previous_centroid[1])))
            tolerance = PREVIOUS_CENTROID_TOLERANCE_FACTOR * float(np.sqrt(area / np.pi))
            if distance > tolerance:
                continue
        if distance < best_distance - 1e-9 or (
                abs(distance - best_distance) <= 1e-9 and area > best_area):
            best_distance = distance
            best_area = area
            best_label = label

    if best_label < 0:
        return _blank_mask((roi_h, roi_w))
    return np.where(labels == best_label, 255, 0).astype(np.uint8)


# ===========================================================================
# 阶段 4b：掌心、手腕与手部局部坐标系
# ===========================================================================
def _estimate_palm_center(mask: np.ndarray) -> Optional[tuple[np.ndarray, float]]:
    """通过距离变换估计掌心中心 C 和手掌尺度 R。

    距离变换前先进行一次小尺度腐蚀，以突出手掌主体，避免过宽的手腕区域
    占据距离变换最大值。
    """
    roi_h, roi_w = mask.shape[:2]
    core_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                            (PALM_CORE_KERNEL, PALM_CORE_KERNEL))
    core = cv2.erode(mask, core_kernel, iterations=1)
    if int(np.count_nonzero(core)) == 0:
        core = _as_uint8_mask(mask)
    distance = cv2.distanceTransform(core, cv2.DIST_L2, 5)
    _, max_value, _, max_location = cv2.minMaxLoc(distance)
    radius = float(max_value)
    if not np.isfinite(radius) or radius < float(MIN_PALM_RADIUS_PX):
        return None
    short_side = float(min(roi_h, roi_w))
    if radius > MAX_PALM_RADIUS_ROI_RATIO * short_side:
        return None
    center = np.array([float(max_location[0]), float(max_location[1])], dtype=np.float64)
    return center, radius


def _pulp_safe(value: float) -> float:
    """返回严格为正的尺度值，防止手掌半径为零导致除零错误。"""
    return float(value) if abs(float(value)) > 1e-6 else 1.0


def _estimate_wrist_center(
    mask: np.ndarray,
    palm_center: np.ndarray,
    palm_radius: float,
) -> Optional[np.ndarray]:
    """取手部底端带状区域的质心，作为手腕中心 W。"""
    occupied_rows = np.where(np.any(mask > 0, axis=1))[0]
    if occupied_rows.size == 0:
        return None
    top_row = int(occupied_rows[0])
    bottom_row = int(occupied_rows[-1])
    if bottom_row <= top_row:
        return None
    band_height = max(3, int(round(WRIST_BAND_RATIO * (bottom_row - top_row + 1))))
    band_top = max(top_row, bottom_row - band_height + 1)
    band = mask[band_top:bottom_row + 1, :]
    ys, xs = np.nonzero(band)
    if xs.size < int(WRIST_MIN_PIXELS):
        return None
    center = np.array([float(xs.mean()), float(band_top + ys.mean())], dtype=np.float64)
    if float(np.linalg.norm(center - palm_center)) < 0.15 * _pulp_safe(palm_radius):
        return None
    return center


def _truncate_forearm(
    mask: np.ndarray,
    palm_center: np.ndarray,
    up: np.ndarray,
    palm_radius: float,
) -> np.ndarray:
    """去除前臂：删除掌心下方距离超过 1.5R 的区域。"""
    roi_h, roi_w = mask.shape[:2]
    margin = TRUNCATE_POLYGON_MARGIN * float(max(roi_h, roi_w))
    origin = palm_center + (-up) * (FOREARM_TRUNCATE_FACTOR * palm_radius)
    along = np.array([-float(up[1]), float(up[0])], dtype=np.float64)
    polygon = np.array([
        origin + along * margin,
        origin - along * margin,
        origin - along * margin + up * margin,
        origin + along * margin + up * margin,
    ], dtype=np.int32)
    allowed = np.zeros((roi_h, roi_w), dtype=np.uint8)
    cv2.fillConvexPoly(allowed, polygon, 255)
    return _as_uint8_mask(cv2.bitwise_and(mask, allowed))


def _touch_edge(points: np.ndarray, shape_hw: tuple[int, int]) -> bool:
    """点集接触 ROI 左边、右边或上边时返回 True。"""
    array = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if array.size == 0:
        return False
    height, width = int(shape_hw[0]), int(shape_hw[1])
    return bool(np.any(array[:, 0] <= 1.0) or np.any(array[:, 0] >= width - 2.0)
                or np.any(array[:, 1] <= 1.0))


def _sharpness_at(contour: np.ndarray, index: int, window: int = SHARPNESS_WINDOW) -> float:
    """计算轮廓 ``index`` 处的夹角；角度越小，凸起越尖锐。"""
    here = _contour_point(contour, index)
    before = _contour_point(contour, index - window)
    after = _contour_point(contour, index + window)
    return _point_angle_deg(here, before - here, after - here)


def _turning_ratio_at(contour: np.ndarray, index: int, window: int) -> float:
    """计算 ``index`` 处与尺度无关的曲率：弯曲深度除以弦长。

    直线段的得分接近 0，缓慢弯曲的掌缘得分较低，指尖圆帽得分明显更高。
    与原始夹角不同，该指标不依赖轮廓的像素尺寸，因此无需绝对像素阈值。
    """
    here = _contour_point(contour, index)
    before = _contour_point(contour, index - window)
    after = _contour_point(contour, index + window)
    chord = after - before
    chord_length = float(np.linalg.norm(chord))
    if chord_length < 1e-6:
        return 1.0
    offset = here - before
    along = float(np.dot(offset, chord)) / chord_length
    perpendicular = float(abs(offset[0] * chord[1] - offset[1] * chord[0])) / chord_length
    del along
    return float(np.clip(perpendicular / chord_length, 0.0, 1.0))


def _tip_sharpness(contour: np.ndarray, index: int) -> tuple[float, float]:
    """返回候选指尖的（转折比例、夹角），两项均考虑尺度影响。"""
    windows = [max(2, int(round(SHARPNESS_WINDOW * factor)))
               for factor in SHARPNESS_WINDOW_FACTORS]
    ratios = [_turning_ratio_at(contour, index, window) for window in windows]
    angles = [_sharpness_at(contour, index, window) for window in windows]
    # 较小窗口能看到指尖圆帽的明显弯曲，较大窗口会把圆帽平均掉；
    # 因此转折比例采用最尖锐读数，角度采用多尺度中位数。
    return (float(np.max(ratios)), float(np.median(angles)))


def _fingertip_candidates_from_contour(
    contour: np.ndarray,
    mask_for_thickness: np.ndarray,
    palm_center: np.ndarray,
    palm_radius: float,
) -> list[dict[str, Any]]:
    """扫描轮廓，寻找同时具备指尖形态的局部距离极大值。

    只考虑相对掌心距离的峰值，因为指尖通常是对应手指上离掌心最远的轮廓点。
    峰值还必须呈现明显弯曲的尖峰，或者比手掌主体薄很多、具有细长手指形态。
    两种判断都相对手掌尺度计算，因此同一组参数可适应不同拍摄距离。
    """
    count = int(contour.shape[0])
    if count < 3:
        return []
    points = contour.reshape(count, 2).astype(np.float64)
    distances = np.linalg.norm(points - palm_center.reshape(1, 2), axis=1)
    # 只在较短的轮廓跨度内寻找局部峰值。张开五指时，各指尖到掌心的距离
    # 并不完全相同；若搜索跨度过大，较近的指尖会被相邻较远指尖掩盖。
    neighbourhood = max(1, int(TIP_PEAK_SPAN))
    peak_floor = distances.copy()
    for offset in range(1, neighbourhood + 1):
        peak_floor = np.maximum(peak_floor, np.roll(distances, offset))
        peak_floor = np.maximum(peak_floor, np.roll(distances, -offset))

    hand = _as_uint8_mask(mask_for_thickness)
    thickness = _interior_thickness(hand)
    candidates: list[dict[str, Any]] = []
    for index in range(count):
        value = float(distances[index])
        if value <= palm_radius:
            continue
        if peak_floor[index] > value + 1e-9:
            continue  # 邻域中存在离掌心更远的轮廓点
        ratio, angle = _tip_sharpness(contour, index)
        # 厚度采样点向轮廓内部移动少许，因为轮廓像素在距离变换中的值
        # 总是接近零，无法代表手指真实厚度。
        inner = points[index] + (palm_center - points[index]) * TIP_THICKNESS_SAMPLE_FACTOR
        local_thickness = _thickness_at(thickness, inner) / palm_radius
        if ratio < TIP_MIN_TURNING_RATIO and local_thickness > TIP_THICKNESS_FACTOR:
            continue
        if angle > TIP_MAX_ANGLE_DEG:
            continue
        candidates.append({
            "point": points[index].copy(),
            "distance": value,
            "angle": angle,
            "turning_ratio": ratio,
            "thickness_ratio": local_thickness,
            "source": "contour",
        })
    return candidates


def _gaps_between_tips(
    contour: np.ndarray,
    tips: list[dict[str, Any]],
    palm_center: np.ndarray,
    palm_radius: float,
) -> list[dict[str, Any]]:
    """在每一对相邻指尖之间寻找真实指缝。

    输入：ROI 坐标中的轮廓、已经按极角排序且含 ``point``/``local`` 字段的
    指尖、掌心中心和手掌尺度。
    输出：每对确实存在凹谷的相邻指尖生成一条记录，包含 ``point``、
    ``local``、``depth_ratio`` 和 ``angle_deg``；没有凹谷则不生成记录。
    """
    if len(tips) < 2 or contour is None or int(contour.shape[0]) < 3:
        return []
    radius = _pulp_safe(palm_radius)
    points = contour.reshape(-1, 2).astype(np.float64)
    palm_distances = np.linalg.norm(points - palm_center.reshape(1, 2), axis=1)
    count = int(points.shape[0])
    gaps: list[dict[str, Any]] = []
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
        valley_distance = float(np.min(arc_distances[inner]))
        best = int(np.argmin(arc_distances[inner])) + 1
        valley = arc[best]
        # 深度表示凹谷相对两指尖连线向内退入的距离，对应凸缺陷深度。
        chord = second_point - first_point
        span = float(np.linalg.norm(chord))
        if span < 1e-6:
            continue
        depth = float(abs(np.cross(chord / span, valley - first_point)))
        local = _to_local(valley.reshape(1, 2), palm_center,
                          np.array([0.0, -1.0]), radius)[0]
        start_local = np.asarray(first.get("local", (0.0, 0.0)), dtype=np.float64).reshape(2)
        end_local = np.asarray(second.get("local", (0.0, 0.0)), dtype=np.float64).reshape(2)
        angle = _point_angle_deg(np.array([local[0], local[1]]),
                                 start_local - local, end_local - local)
        if angle > DEFECT_MAX_ANGLE_DEG:
            continue
        if depth < DEFECT_MIN_DEPTH_FACTOR * radius:
            continue
        gaps.append({
            "point": valley.copy(),
            "local": (float(local[0]), float(local[1])),
            "depth_ratio": depth / radius,
            "angle_deg": angle,
            "distance": valley_distance,
        })
    return _deduplicate_points(gaps, TIP_CLUSTER_DISTANCE_FACTOR * radius)


def _neighbourhood_count(binary: np.ndarray) -> np.ndarray:
    """返回每个像素周围八邻域中非零像素的数量。"""
    total = np.zeros(binary.shape, dtype=np.uint8)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            total += np.roll(np.roll(binary, dy, axis=0), dx, axis=1)
    return total


def _skeletonize(binary: np.ndarray, max_iterations: int = SKELETON_MAX_ITERATIONS) -> np.ndarray:
    """使用 Zhang-Suen 细化算法把二值掩膜压缩为单像素宽骨架。

    输入：单通道 0/255 掩膜，非零值表示前景。
    输出：0/255 骨架掩膜；输入为空时返回空骨架。
    """
    image = _as_uint8_mask(binary)
    image = (image > 0).astype(np.uint8)
    if int(image.sum()) == 0:
        return image.copy()
    padded = np.pad(image, 1, mode="constant")

    for _ in range(int(max_iterations)):
        removed_any = False
        for step in (0, 1):
            p2 = padded[0:-2, 1:-1]
            p3 = padded[0:-2, 2:]
            p4 = padded[1:-1, 2:]
            p5 = padded[2:, 2:]
            p6 = padded[2:, 1:-1]
            p7 = padded[2:, 0:-2]
            p8 = padded[1:-1, 0:-2]
            p9 = padded[0:-2, 0:-2]
            centre = padded[1:-1, 1:-1]

            neighbours = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
            transitions = ((p2 == 0) & (p3 == 1)).astype(np.uint8)
            transitions += ((p3 == 0) & (p4 == 1)).astype(np.uint8)
            transitions += ((p4 == 0) & (p5 == 1)).astype(np.uint8)
            transitions += ((p5 == 0) & (p6 == 1)).astype(np.uint8)
            transitions += ((p6 == 0) & (p7 == 1)).astype(np.uint8)
            transitions += ((p7 == 0) & (p8 == 1)).astype(np.uint8)
            transitions += ((p8 == 0) & (p9 == 1)).astype(np.uint8)
            transitions += ((p9 == 0) & (p2 == 1)).astype(np.uint8)

            if step == 0:
                extra = (p2 * p4 * p6 == 0) & (p4 * p6 * p8 == 0)
            else:
                extra = (p2 * p4 * p8 == 0) & (p2 * p6 * p8 == 0)

            doomed = (centre == 1) & (neighbours >= 2) & (neighbours <= 6) & \
                     (transitions == 1) & extra
            if not bool(doomed.any()):
                continue
            removed_any = True
            padded[1:-1, 1:-1] = np.where(doomed, 0, padded[1:-1, 1:-1])
        if not removed_any:
            break
    return (padded[1:-1, 1:-1] * 255).astype(np.uint8)


def _skeleton_endpoints(skeleton: np.ndarray) -> list[np.ndarray]:
    """以浮点坐标返回骨架端点；端点在八邻域中只有一个相邻骨架像素。"""
    binary = (skeleton > 0).astype(np.uint8)
    if int(binary.sum()) == 0:
        return []
    endpoints = (binary == 1) & (_neighbourhood_count(binary) == 1)
    ys, xs = np.nonzero(endpoints)
    return [np.array([float(x), float(y)], dtype=np.float64) for x, y in zip(xs, ys)]


def _thin_part_candidates(
    mask: np.ndarray,
    palm_center: np.ndarray,
    palm_radius: float,
    skeleton_endpoints: list[np.ndarray],
) -> list[dict[str, Any]]:
    """从掩膜的每个细长分支中生成指尖候选。

    张开手的中轴骨架通常由掌部主干和每根手指的分支组成，因此骨架端点可
    作为指尖候选。靠近骨架端点的厚度图峰值也会加入候选，因为它更接近
    同一手指的真实圆帽中心。掌部内部端点会被相对掌心距离条件剔除，
    手腕截断后留下的骨架残端也通过同一条件排除。
    """
    hand = _as_uint8_mask(mask)
    if int(np.count_nonzero(hand)) == 0:
        return []
    interior = _interior_thickness(hand)
    match_radius = max(4.0, THIN_PART_ENDPOINT_FACTOR * float(palm_radius))
    candidates: list[dict[str, Any]] = []

    for endpoint in skeleton_endpoints:
        point = np.asarray(endpoint, dtype=np.float64).reshape(2)
        distance = float(np.linalg.norm(point - palm_center))
        if distance <= TIP_MIN_DISTANCE_FACTOR * float(palm_radius):
            continue
        candidates.append({
            "point": point,
            "distance": distance,
            "angle": 180.0,
            "turning_ratio": 0.0,
            "thickness_ratio": _thickness_at(interior, point) / float(palm_radius),
            "source": "skeleton",
        })

    peak_limit = max(INTERIOR_PEAK_MIN_PIXELS, 0.01 * float(palm_radius))
    if float(interior.max()) < peak_limit or not skeleton_endpoints:
        return candidates
    suppression = max(3, int(round(THIN_PART_SUPPRESSION * float(palm_radius))))
    working = interior.copy()
    for _ in range(INTERIOR_PEAK_LIMIT):
        _, best_value, _, best_location = cv2.minMaxLoc(working)
        if not np.isfinite(best_value) or best_value < peak_limit:
            break
        x, y = int(best_location[0]), int(best_location[1])
        working[max(0, y - suppression):y + suppression + 1,
                max(0, x - suppression):x + suppression + 1] = 0.0
        point = np.array([float(x), float(y)], dtype=np.float64)
        nearest = min(float(np.linalg.norm(point - end)) for end in skeleton_endpoints)
        if nearest > match_radius:
            continue
        distance = float(np.linalg.norm(point - palm_center))
        if distance <= TIP_MIN_DISTANCE_FACTOR * float(palm_radius):
            continue
        candidates.append({
            "point": point,
            "distance": distance,
            "angle": 180.0,
            "turning_ratio": 0.0,
            "thickness_ratio": float(best_value) / float(palm_radius),
            "source": "thin_part",
        })
    return candidates


def _thickness_at(thickness: np.ndarray, point: np.ndarray) -> float:
    """读取指定点的内部厚度；坐标超界时先限制到图像范围内。"""
    if thickness is None or thickness.size == 0:
        return 0.0
    x = int(np.clip(int(round(float(point[0]))), 0, thickness.shape[1] - 1))
    y = int(np.clip(int(round(float(point[1]))), 0, thickness.shape[0] - 1))
    return float(thickness[y, x])


def _deduplicate_points(items: list[dict[str, Any]], minimum_distance: float) -> list[dict[str, Any]]:
    """贪心去重：每个空间簇只保留距离掌心最远的点。"""
    ordered = sorted(items, key=lambda item: -float(item["distance"]))
    kept: list[dict[str, Any]] = []
    for item in ordered:
        if all(float(np.linalg.norm(item["point"] - other["point"])) >= minimum_distance
               for other in kept):
            kept.append(item)
    return kept


def _detect_finger_gaps(
    contour: np.ndarray,
    palm_center: np.ndarray,
    palm_radius: float,
) -> list[dict[str, Any]]:
    """返回通过角度、深度验证且已经去重的凸缺陷，也就是指缝候选。"""
    count = int(contour.shape[0])
    if count < 4:
        return []
    hull_indices = cv2.convexHull(contour, returnPoints=False)
    if hull_indices is None or len(hull_indices) < 4:
        return []
    try:
        defects = cv2.convexityDefects(contour, hull_indices)
    except cv2.error:
        return []
    if defects is None:
        return []

    # OpenCV 4 返回形状 (N, 1, 4)，OpenCV 5 返回 (N, 4)，统一展开处理。
    defect_rows = np.asarray(defects).reshape(-1, 4)
    radius = _pulp_safe(palm_radius)
    gaps: list[dict[str, Any]] = []
    for index in range(int(defect_rows.shape[0])):
        start_index, end_index, far_index, depth_fixed = defect_rows[index]
        start = _contour_point(contour, int(start_index))
        end = _contour_point(contour, int(end_index))
        far = _contour_point(contour, int(far_index))
        depth = float(depth_fixed) / 256.0
        if depth < DEFECT_MIN_DEPTH_FACTOR * radius:
            continue
        angle = _point_angle_deg(far, start - far, end - far)
        if angle > DEFECT_MAX_ANGLE_DEG:
            continue
        if float(np.linalg.norm(far - palm_center)) < PALM_CENTER_FACTOR * radius:
            continue
        gaps.append({
            "point": far.copy(),
            "start": start,
            "end": end,
            "distance": float(np.linalg.norm(far - palm_center)),
            "depth": depth,
            "angle": angle,
        })
    gaps.sort(key=lambda item: -float(item["depth"]))
    return _deduplicate_points(gaps, TIP_CLUSTER_DISTANCE_FACTOR * radius)


def _merge_neighbouring_tips(
    tips: list[dict[str, Any]],
    palm_center: np.ndarray,
    palm_radius: float,
) -> list[dict[str, Any]]:
    """按相对掌心的极角排序指尖，并合并距离过近的重复候选。"""
    if not tips:
        return []
    radius = _pulp_safe(palm_radius)
    local = _to_local(np.array([item["point"] for item in tips]),
                      palm_center, np.array([0.0, -1.0]), radius)
    for item, (lx, ly) in zip(tips, local):
        item["angle"] = float(np.arctan2(ly, lx))
        item["local_x"] = float(lx)
        item["local_y"] = float(ly)
    ordered = sorted(tips, key=lambda item: item["angle"])
    merged: list[dict[str, Any]] = []
    for item in ordered:
        if not merged:
            merged.append(item)
            continue
        previous = merged[-1]
        spatial = float(np.linalg.norm(item["point"] - previous["point"]))
        # 距离小于 TIP_MERGE_DISTANCE_FACTOR×R 的候选通常描述同一个物理凸起，
        # 例如同一手指的骨架端点与圆帽峰值，因此只保留离掌心更远者。
        # 张开手的不同指尖（包括拇指与食指）通常会超过该距离。
        if spatial < TIP_MERGE_DISTANCE_FACTOR * radius:
            if float(item["distance"]) > float(previous["distance"]):
                merged[-1] = item
        else:
            merged.append(item)
    return merged


def extract_hand_features(mask: np.ndarray) -> Optional[dict[str, object]]:
    """提取掌心、手腕、局部坐标系、轮廓、指缝和指尖特征。

    输入：ROI 坐标中的单通道 uint8 掩膜，背景为 0，手部为 255。
    输出：特征字典，包含 ``palm_center``、``palm_radius``、``up``/``right``、
    截断后的 ``contour``/``hull``、``hull_local``、``solidity``、0～5 个
    ``fingertips`` 以及 ``finger_gaps``；其中几何量已按手掌尺度 R 归一化。
    失败：帧质量不足时返回 ``None``；空掩膜或噪声掩膜不会引发异常。
    """
    if mask is None or getattr(mask, "size", 0) == 0:
        return None
    hand = _as_uint8_mask(mask)
    if int(np.count_nonzero(hand)) == 0:
        return None

    contours, _ = cv2.findContours(hand, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    raw = max(contours, key=cv2.contourArea)
    if raw is None or int(raw.shape[0]) < int(CONTOUR_MIN_POINTS):
        return None
    if float(cv2.contourArea(raw)) <= 0.0:
        return None

    palm = _estimate_palm_center(hand)
    if palm is None:
        return None
    palm_center, palm_radius = palm
    radius = _pulp_safe(palm_radius)
    if _touch_edge(raw.reshape(-1, 2), hand.shape[:2]):
        return None

    wrist = _estimate_wrist_center(hand, palm_center, radius)
    if wrist is None:
        return None
    up = _unit(palm_center - wrist, np.array([0.0, -1.0]))
    right = np.array([-float(up[1]), float(up[0])], dtype=np.float64)

    truncated = _truncate_forearm(hand, palm_center, up, radius)
    analysis_contours, _ = cv2.findContours(truncated, cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_NONE)
    if not analysis_contours:
        return None
    contour = max(analysis_contours, key=cv2.contourArea)
    contour_area = float(cv2.contourArea(contour))
    if contour_area < CONTOUR_TRUNCATED_MIN_AREA_RATIO * float(hand.shape[0] * hand.shape[1]):
        return None
    if _touch_edge(contour.reshape(-1, 2), hand.shape[:2]):
        return None

    perimeter = float(cv2.arcLength(contour, True))
    epsilon = max(1.0, POLY_EPSILON_RATIO * perimeter)
    simplified = cv2.approxPolyDP(contour, epsilon, True)
    working = contour
    if simplified is not None and int(simplified.shape[0]) >= CONTOUR_POLY_MIN_POINTS:
        working = np.asarray(simplified, dtype=np.int32).reshape(-1, 1, 2)

    # 在原始全分辨率轮廓上寻找指尖候选，因为多边形近似可能削掉最外侧指尖。
    thickness = _interior_thickness(hand)
    skeleton_endpoints = _skeleton_endpoints(_skeletonize(truncated))
    tip_candidates = _fingertip_candidates_from_contour(contour, hand,
                                                        palm_center, radius)
    tip_candidates.extend(_thin_part_candidates(hand, palm_center, radius,
                                                skeleton_endpoints))

    hull = cv2.convexHull(working)
    if hull is None or int(hull.shape[0]) < 4:
        return None
    hull_area = float(cv2.contourArea(hull))
    if hull_area <= 0.0:
        return None
    solidity = float(np.clip(contour_area / hull_area, 0.0, 1.0))
    hull_points = hull.reshape(-1, 2).astype(np.float64)
    hull_local = _to_local(hull_points, palm_center, up, radius)

    defect_gaps = _detect_finger_gaps(working, palm_center, radius)

    tips: list[dict[str, Any]] = []
    for gap in defect_gaps:
        for endpoint in (gap["start"], gap["end"]):
            distance = float(np.linalg.norm(endpoint - palm_center))
            if distance < TIP_MIN_DISTANCE_FACTOR * radius:
                continue
            local = _to_local(endpoint.reshape(1, 2), palm_center, up, radius)[0]
            if not (TIP_MIN_PALM_Y <= local[1] <= 0.0):
                continue
            tips.append({
                "point": np.asarray(endpoint, dtype=np.float64).reshape(2),
                "distance": distance,
                "angle_deg": 180.0,
                "source": "defect",
            })

    for candidate in tip_candidates:
        point = candidate["point"]
        local = _to_local(point.reshape(1, 2), palm_center, up, radius)[0]
        # 局部坐标系中指尖方向为 y 负、手腕方向为 y 正（见 _to_local），
        # 因此指尖应落在掌心上方 TIP_MIN_PALM_Y～0 区间；这也能排除前臂
        # 截断线下方留下的两个尖角。
        if not (TIP_MIN_PALM_Y <= local[1] <= 0.0):
            continue
        tips.append({
            "point": point,
            "distance": float(candidate["distance"]),
            "angle_deg": float(candidate["angle"]),
            "thickness_ratio": float(candidate.get("thickness_ratio", 0.0)),
            "source": str(candidate.get("source", "contour")),
        })

    tips = _deduplicate_points(tips, TIP_CLUSTER_DISTANCE_FACTOR * radius)
    tips = _merge_neighbouring_tips(tips, palm_center, radius)
    # 先保留离掌心最远的五个候选，再按掌心周围的角度排序，避免截断候选时
    # 因排序位置而错误丢弃真实指尖。
    tips = sorted(tips, key=lambda item: -float(item["distance"]))[:MAX_TIP_COUNT]
    tips = sorted(tips, key=lambda item: float(item.get("angle", 0.0)))

    fingertips: list[dict[str, Any]] = []
    for item in tips:
        point = np.asarray(item["point"], dtype=np.float64).reshape(2)
        local = _to_local(point.reshape(1, 2), palm_center, up, radius)[0]
        fingertips.append({
            "point": point,
            "local": (float(local[0]), float(local[1])),
            "distance_ratio": float(item["distance"]) / radius,
            "angle_deg": float(item.get("angle_deg", 180.0)),
            "thickness_ratio": float(item.get("thickness_ratio", 0.0)),
            "source": str(item.get("source", "contour")),
        })

    # 只在最终接受的指尖之间测量凹谷，避免孤立噪声凸缺陷被计为指缝。
    gaps = _gaps_between_tips(contour, fingertips, palm_center, radius)

    gap_records: list[dict[str, Any]] = []
    for gap in gaps:
        gap_records.append({
            "point": np.asarray(gap["point"], dtype=np.float64).reshape(2),
            "local": (float(gap["local"][0]), float(gap["local"][1])),
            "depth_ratio": float(gap["depth_ratio"]),
            "angle_deg": float(gap["angle_deg"]),
        })

    gap_count = len(gap_records)
    if gap_count > MAX_GAP_COUNT:
        gap_records = gap_records[:MAX_GAP_COUNT]

    tip_points = np.array([item["point"] for item in fingertips], dtype=np.float64) \
        if fingertips else np.zeros((0, 2), dtype=np.float64)
    tip_locals = np.array([item["local"] for item in fingertips], dtype=np.float64) \
        if fingertips else np.zeros((0, 2), dtype=np.float64)

    return {
        "roi_shape": (int(hand.shape[0]), int(hand.shape[1])),
        "palm_center": palm_center,
        "palm_center_local": (0.0, 0.0),
        "palm_radius": float(palm_radius),
        "wrist_center": np.asarray(wrist, dtype=np.float64).reshape(2),
        "up": up,
        "right": right,
        "contour": np.asarray(working, dtype=np.int32).reshape(-1, 1, 2),
        "contour_area": contour_area,
        "hull": np.asarray(hull, dtype=np.int32).reshape(-1, 1, 2),
        "hull_local": hull_local,
        "solidity": solidity,
        "fingertips": fingertips,
        "tip_points": tip_points,
        "tip_locals": tip_locals,
        "finger_gaps": gap_records,
        "valid_gap_count": gap_count,
        "raw_gap_count": len(gaps),
        "quality_ok": True,
    }


# ===========================================================================
# 阶段 4c：单帧分类
# ===========================================================================
def _failed_classification(reason: str) -> tuple[str, float, str]:
    """返回统一格式的分类失败结果。"""
    return ("UNKNOWN", 0.0, reason)


def _valid_points_from_features(features: Optional[dict[str, object]]) -> np.ndarray:
    """在数据合法时返回形状为 (N, 2) 的指尖局部坐标数组。"""
    if not isinstance(features, dict):
        return np.zeros((0, 2), dtype=np.float64)
    points = features.get("tip_locals")
    array = np.asarray(points, dtype=np.float64) if points is not None \
        else np.zeros((0, 2), dtype=np.float64)
    array = array.reshape(-1, 2)
    if array.shape[0] == 0:
        return array
    if not np.all(np.isfinite(array)):
        return np.zeros((0, 2), dtype=np.float64)
    return array


def _gap_points_from_features(features: dict[str, object]) -> np.ndarray:
    """返回形状为 (N, 2) 的指缝局部坐标数组。"""
    gaps = features.get("finger_gaps")
    if not isinstance(gaps, (list, tuple)) or len(gaps) == 0:
        return np.zeros((0, 2), dtype=np.float64)
    collected: list[tuple[float, float]] = []
    for gap in gaps:
        if isinstance(gap, dict) and "local" in gap:
            local = gap["local"]
            collected.append((float(local[0]), float(local[1])))
    if not collected:
        return np.zeros((0, 2), dtype=np.float64)
    return np.asarray(collected, dtype=np.float64).reshape(-1, 2)


def _solidity_score(solidity: float) -> float:
    """把实心度映射到 0～1；合法区间中点得分为 1。"""
    if solidity <= 0.0:
        return 0.0
    if MIN_SOLIDITY <= solidity <= MAX_SOLIDITY:
        half = max(1e-6, 0.5 * (MAX_SOLIDITY - MIN_SOLIDITY))
        return float(1.0 - min(1.0, abs(solidity - 0.5 * (MIN_SOLIDITY + MAX_SOLIDITY)) / half))
    if solidity < MIN_SOLIDITY:
        return float(np.clip(solidity / MIN_SOLIDITY, 0.0, 1.0) * 0.5)
    return float(np.clip((1.0 - solidity) / max(1e-6, 1.0 - MAX_SOLIDITY), 0.0, 1.0) * 0.5)


def _mask_quality_score(features: dict[str, object], tips: np.ndarray) -> float:
    """计算具有较强局部轮廓证据的指尖候选比例。"""
    if tips.shape[0] == 0:
        return 0.0
    sources = [str(item.get("source", "")) for item in features.get("fingertips", []) or []]
    if len(sources) != tips.shape[0] or not sources:
        return 0.0
    # 骨架端点和轮廓峰值都属于直接局部证据；凸缺陷端点由凹谷间接推断，
    # 因此只把前两类候选计入强证据。
    strong = sum(1 for source in sources if source != "defect")
    return float(strong) / float(len(sources))


def classify_open_five(features: dict[str, object]) -> tuple[str, float, str]:
    """把单帧分类为 FIVE 或 UNKNOWN，并返回置信度与原因。

    输入：由 :func:`extract_hand_features` 生成的特征字典，或至少具有
    ``tip_locals`` 和 ``finger_gaps`` 的等价映射。
    输出：``(label, confidence, reason)``；标签为 FIVE 或 UNKNOWN，置信度
    位于 0～1，原因说明第一个未满足条件。
    行为：所有硬条件都必须成立，平均分不能补偿任何硬条件失败。
    """
    if not isinstance(features, dict):
        return _failed_classification("no hand features / no valid subject")
    if not bool(features.get("quality_ok", True)):
        return _failed_classification(
            str(features.get("quality_reason") or "frame quality gate failed"))

    tips = _valid_points_from_features(features)
    tip_count = int(tips.shape[0])
    gaps = _gap_points_from_features(features)
    gap_count = int(gaps.shape[0])
    solidity = float(features.get("solidity", 0.0) or 0.0)

    if tip_count < MAX_TIP_COUNT:
        return _failed_classification("only %d fingertips detected (need 5)" % tip_count)
    if tip_count > MAX_TIP_COUNT:
        return _failed_classification("too many fingertips (%d)" % tip_count)
    if gap_count < MAX_GAP_COUNT:
        return _failed_classification("only %d valid finger gaps (need 4)" % gap_count)
    if gap_count > MAX_GAP_COUNT:
        return _failed_classification("too many finger gaps (%d)" % gap_count)
    if solidity <= 0.0:
        return _failed_classification("solidity unavailable")
    if solidity < MIN_SOLIDITY:
        return _failed_classification("solidity %.3f below %.2f" % (solidity, MIN_SOLIDITY))
    if solidity > MAX_SOLIDITY:
        return _failed_classification("solidity %.3f above %.2f" % (solidity, MAX_SOLIDITY))

    distances = np.linalg.norm(tips, axis=1)
    if float(np.min(distances)) < TIP_MIN_DISTANCE_FACTOR:
        return _failed_classification("a fingertip is closer than %.2f R to the palm centre"
                                      % TIP_MIN_DISTANCE_FACTOR)
    if int(np.count_nonzero(tips[:, 1] <= -TIP_TOP_FACTOR)) < TIP_MIN_IN_UPPER:
        return _failed_classification("fewer than %d fingertips in the upper palm region"
                                      % TIP_MIN_IN_UPPER)
    lateral_span = float(np.max(tips[:, 0]) - np.min(tips[:, 0]))
    if lateral_span < TIP_LATERAL_SPAN_FACTOR:
        return _failed_classification("fingertips span only %.2f R across the palm (need %.2f R)"
                                      % (lateral_span, TIP_LATERAL_SPAN_FACTOR))
    if float(np.min(tips[:, 0])) > 0.0 or float(np.max(tips[:, 0])) < 0.0:
        return _failed_classification("fingertips do not cross both palm sides")

    if ENABLE_FINGERTIP_RESIDUAL_CHECK:
        hull_local = features.get("hull_local")
        if hull_local is not None:
            hull_array = np.asarray(hull_local, dtype=np.float64).reshape(-1, 2)
            if hull_array.shape[0] >= 3:
                residuals = [float(np.min(np.linalg.norm(
                    hull_array - tip.reshape(1, 2), axis=1))) for tip in tips]
                if max(residuals) > FINGERTIP_RESIDUAL_FACTOR:
                    return _failed_classification("a fingertip is not on the hull")

    tip_score = 1.0
    gap_score = 1.0 if gap_count == MAX_GAP_COUNT else float(gap_count) / float(MAX_GAP_COUNT)
    solidity_score = _solidity_score(solidity)
    spread_score = float(np.clip(lateral_span / 3.0, 0.0, 1.0))
    length_score = float(np.clip(
        (float(np.mean(distances)) - TIP_MIN_DISTANCE_FACTOR) /
        max(1e-6, 0.65), 0.0, 1.0))
    mask_score = _mask_quality_score(features, tips)
    upper_score = float(np.clip(
        int(np.count_nonzero(tips[:, 1] <= -TIP_TOP_FACTOR)) /
        float(MAX_TIP_COUNT), 0.0, 1.0))

    weights = (0.15, 0.15, 0.15, 0.15, 0.15, 0.125, 0.125)  # 权重总和为 1.0
    confidence = float(np.clip(
        weights[0] * tip_score +
        weights[1] * gap_score +
        weights[2] * solidity_score +
        weights[3] * spread_score +
        weights[4] * length_score +
        weights[5] * mask_score +
        weights[6] * upper_score, 0.0, 1.0))

    if confidence < CANDIDATE_MIN_CONFIDENCE:
        return _failed_classification("confidence %.2f below candidate threshold %.2f"
                                      % (confidence, CANDIDATE_MIN_CONFIDENCE))
    return ("FIVE", confidence, "five fingertips, four gaps, solidity %.2f" % solidity)


# ===========================================================================
# 阶段 4d：连续帧确认
# ===========================================================================
# 使用投票窗口的 id() 保存确认延迟。普通 ``collections.deque`` 不能附加
# 自定义属性，因此把这部分状态单独保存在字典中。
_LATENCY_STATE: dict[int, tuple[Optional[float], float]] = {}


def _history_latency(history: collections.deque) -> float:
    """返回指定投票窗口保存的手势确认延迟。"""
    return float(_LATENCY_STATE.get(id(history), (None, 0.0))[1])


def _set_history_latency(history: collections.deque,
                         start: Optional[float], value: float) -> None:
    """保存指定投票窗口的确认起点和延迟。"""
    _LATENCY_STATE[id(history)] = (start, float(value))


def update_stable_state(
    history: collections.deque,
    frame_label: str,
    confidence: float,
    timestamp: float,
) -> tuple[str, bool]:
    """更新 12 帧投票窗口，返回（稳定标签，状态是否改变）。

    输入：保存 ``(时间戳, 标签, 置信度, 稳定状态)`` 的双端队列、当前单帧
    标签与置信度，以及当前时间戳。
    输出：稳定标签 FIVE/UNKNOWN；只有稳定状态实际发生改变的那一帧才返回
    True。
    行为：进入 FIVE 需要至少 8 帧 FIVE 且平均置信度不低于 0.65；保持阶段
    最多容忍连续 5 帧 UNKNOWN，窗口中 FIVE 少于 4 帧时释放稳定状态。
    """
    if history is None:
        history = collections.deque(maxlen=HISTORY_LENGTH)
    previous_stable = str(history[-1][3]) if len(history) >= 1 else "UNKNOWN"
    if previous_stable not in ("FIVE", "UNKNOWN"):
        previous_stable = "UNKNOWN"

    history.append((float(timestamp), str(frame_label), float(confidence),
                    previous_stable))
    entries = list(history)[-HISTORY_LENGTH:]

    five_confidences = [float(item[2]) for item in entries if str(item[1]) == "FIVE"]
    five_count = len(five_confidences)
    mean_confidence = float(np.mean(five_confidences)) if five_confidences else 0.0

    consecutive_unknown = 0
    for item in reversed(entries):
        if str(item[1]) == "UNKNOWN":
            consecutive_unknown += 1
        else:
            break

    if previous_stable == "FIVE":
        if five_count < FIVE_FRAMES_TO_HOLD or \
                consecutive_unknown > MAX_CONSECUTIVE_UNKNOWN_TO_HOLD:
            stable = "UNKNOWN"
        else:
            stable = "FIVE"
    else:
        if five_count >= FIVE_FRAMES_TO_CONFIRM and \
                mean_confidence >= FIVE_MEAN_CONFIDENCE_TO_CONFIRM:
            stable = "FIVE"
        else:
            stable = "UNKNOWN"

    if stable == "FIVE" and previous_stable != "FIVE":
        start = entries[0][0] if len(entries) == HISTORY_LENGTH else float(timestamp)
        _set_history_latency(history, start, float(timestamp) - float(start))
    if stable == "UNKNOWN" and previous_stable != "UNKNOWN":
        _set_history_latency(history, None, 0.0)

    history[-1] = (float(timestamp), str(frame_label), float(confidence), stable)
    return stable, stable != previous_stable


# ===========================================================================
# 阶段 5：调试可视化
# ===========================================================================
def _draw_text_block(canvas: np.ndarray, lines: list[str],
                     origin: tuple[int, int], colour: tuple[int, int, int]) -> None:
    """从 ``origin`` 指定位置开始，逐行绘制文本列表。"""
    x, y = origin
    for index, text in enumerate(lines):
        cv2.putText(canvas, text, (x, y + index * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 1, cv2.LINE_AA)


def _draw_hint_banner(canvas: np.ndarray, phase: str, instruction: str,
                      colour: tuple[int, int, int] = COLOR_TEXT) -> None:
    """在画面底部绘制“当前阶段 + 操作提示”，让用户看清该做什么。

    标定阶段原本只在画面左上角显示一行小字，容易被忽略，导致看起来像卡住；
    这个提示条固定贴在底部，不会遮挡位于画面上部的 ROI。
    """
    if canvas is None or canvas.size == 0:
        return
    height, width = canvas.shape[:2]
    title = phase.encode("ascii", "ignore").decode("ascii") or "stage"
    text = instruction.encode("ascii", "ignore").decode("ascii") or "-"
    cv2.putText(canvas, title, (10, height - 44),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, colour, 2, cv2.LINE_AA)
    cv2.putText(canvas, text[:70], (10, height - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT, 1, cv2.LINE_AA)


def draw_debug_view(
    frame: np.ndarray,
    roi: tuple[int, int, int, int],
    mask: np.ndarray,
    features: Optional[dict[str, object]],
    frame_result: tuple[str, float, str],
    stable_label: str,
    fps: float,
) -> np.ndarray:
    """绘制完整调试叠加层并返回新画布，不修改输入画面。

    输入：BGR 帧、ROI 矩形、当前掩膜、特征字典或 None、单帧
    ``(标签, 置信度, 原因)``、稳定标签和实测 FPS。
    输出：新的 BGR 画布，其中绘制 ROI、掩膜缩略图、轮廓、凸包、掌心圆、
    指尖、指缝、识别结果、失败原因、置信度和 FPS。
    """
    if frame is None:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    canvas = frame.copy()
    left, top, roi_w, roi_h = (int(value) for value in roi)
    frame_height, frame_width = canvas.shape[:2]

    cv2.rectangle(canvas, (left, top), (left + roi_w - 1, top + roi_h - 1), COLOR_ROI, 2)

    if mask is not None and getattr(mask, "size", 0) > 0 and mask.shape[:2] == (roi_h, roi_w):
        thumbnail_height = max(1, int(round(MASK_THUMB_WIDTH * roi_h / float(roi_w))))
        thumbnail = cv2.resize(mask, (MASK_THUMB_WIDTH, thumbnail_height),
                               interpolation=cv2.INTER_NEAREST)
        thumbnail_bgr = cv2.cvtColor(thumbnail, cv2.COLOR_GRAY2BGR)
        y0 = max(0, frame_height - thumbnail_height - 8)
        x0 = max(0, frame_width - MASK_THUMB_WIDTH - 8)
        canvas[y0:y0 + thumbnail_height, x0:x0 + MASK_THUMB_WIDTH] = thumbnail_bgr
        cv2.putText(canvas, "mask", (x0 + 4, y0 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_ROI, 1, cv2.LINE_AA)

    if isinstance(features, dict):
        contour = features.get("contour")
        if contour is not None:
            shifted = contour.reshape(-1, 2).astype(np.int32) + np.array([left, top],
                                                                        dtype=np.int32)
            cv2.drawContours(canvas, [shifted.reshape(-1, 1, 2)], -1, COLOR_CONTOUR, 2)
        hull = features.get("hull")
        if hull is not None:
            hull_shifted = hull.reshape(-1, 2).astype(np.int32) + np.array([left, top],
                                                                          dtype=np.int32)
            cv2.drawContours(canvas, [hull_shifted.reshape(-1, 1, 2)], -1, COLOR_HULL, 1)
        palm_center = features.get("palm_center")
        palm_radius = float(features.get("palm_radius", 0.0) or 0.0)
        if palm_center is not None and palm_radius > 0:
            cv2.circle(canvas, (int(round(float(palm_center[0]))) + left,
                                int(round(float(palm_center[1]))) + top),
                       int(round(palm_radius)), COLOR_PALM, 2)
        wrist_center = features.get("wrist_center")
        if wrist_center is not None:
            cv2.circle(canvas, (int(round(float(wrist_center[0]))) + left,
                                int(round(float(wrist_center[1]))) + top),
                       4, COLOR_PALM, -1)
        for tip in features.get("fingertips", []) or []:
            point = np.asarray(tip["point"], dtype=np.float64).reshape(2)
            center = (int(round(float(point[0]))) + left, int(round(float(point[1]))) + top)
            cv2.circle(canvas, center, 5, COLOR_TIP, -1)
            cv2.putText(canvas, str(round(float(tip.get("distance_ratio", 0.0)), 2)),
                        (center[0] + 6, center[1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_TIP, 1, cv2.LINE_AA)
        for gap in features.get("finger_gaps", []) or []:
            point = np.asarray(gap["point"], dtype=np.float64).reshape(2)
            center = (int(round(float(point[0]))) + left, int(round(float(point[1]))) + top)
            cv2.drawMarker(canvas, center, COLOR_GAP, cv2.MARKER_TILTED_CROSS, 10, 2)

    frame_label = str(frame_result[0]) if frame_result else "UNKNOWN"
    frame_confidence = float(frame_result[1]) if frame_result and len(frame_result) > 1 else 0.0
    reason = str(frame_result[2]) if frame_result and len(frame_result) > 2 else ""
    tip_count = len(features.get("fingertips", []) or []) if isinstance(features, dict) else 0
    gap_count = int(features.get("valid_gap_count", 0) or 0) if isinstance(features, dict) else 0
    solidity = float(features.get("solidity", 0.0) or 0.0) if isinstance(features, dict) else 0.0

    lines = [
        "frame: %s  conf %.2f" % (frame_label, frame_confidence),
        "stable: %s" % str(stable_label),
        "tips %d  gaps %d" % (tip_count, gap_count),
        "solidity %.2f" % solidity,
        "fps %.1f" % float(fps),
        "reason: %s" % (reason[:48] if reason else "-"),
    ]
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (min(frame_width, 380), 22 * len(lines) + 12), (0, 0, 0), -1)
    canvas = cv2.addWeighted(overlay, 0.45, canvas, 0.55, 0.0)
    colour = COLOR_OK if frame_label == "FIVE" else COLOR_BAD
    _draw_text_block(canvas, lines, (10, 22), colour)
    cv2.putText(canvas, "q quit / r recalibrate", (10, frame_height - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT, 1, cv2.LINE_AA)
    return canvas


# ===========================================================================
# 阶段 6：主状态机
# ===========================================================================
def _safe_release(cap: Optional[cv2.VideoCapture]) -> None:
    """安全释放摄像头捕获对象，忽略底层后端在释放时产生的异常。"""
    if cap is None:
        return
    try:
        cap.release()
    except Exception:  # pragma: no cover - 防御性异常处理
        pass


def _safe_destroy_windows() -> None:
    """安全销毁 OpenCV 窗口，忽略图形后端异常。"""
    try:
        cv2.destroyAllWindows()
    except Exception:  # pragma: no cover - 防御性异常处理
        pass


def _create_window() -> None:
    """尽可能创建支持缩放的调试窗口；失败时由后续流程继续处理。"""
    try:
        cv2.namedWindow(DEBUG_WINDOW_NAME, cv2.WINDOW_NORMAL)
    except Exception:  # pragma: no cover - 防御性异常处理
        pass


def _handle_ui_event() -> str:
    """处理一次界面事件，把按键和窗口关闭统一成一个动作。

    输出："quit"、"recalibrate" 或 "none"。
    行为：除 q / r 按键外，也检测用户直接点击窗口右上角关闭按钮的情况；
    否则程序会继续在一个已经消失的窗口上循环，看起来像是“卡死”。
    """
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        return "quit"
    if key == ord("r"):
        return "recalibrate"
    if key in (32, 13):  # 空格或回车
        return "start"
    try:
        visible = cv2.getWindowProperty(DEBUG_WINDOW_NAME, cv2.WND_PROP_VISIBLE)
    except Exception:  # pragma: no cover - 某些后端不支持该属性
        return "none"
    return "none" if float(visible) >= 1.0 else "quit"


def _wait_for_start(cap: cv2.VideoCapture,
                    phase: str,
                    instruction: str,
                    colour: tuple[int, int, int] = COLOR_TEXT) -> Optional[str]:
    """显示实时预览并等待用户按空格开始，返回 None 表示应继续采集。

    采集只有几十帧，若不等待，阶段会在约一秒内自动跑完，用户根本来不及把
    手放进 ROI；同时返回的字符串用于告诉调用方发生了 q（退出）或 r（重标定）。
    """
    while True:
        ok, frame = cap.read()
        if ok and frame is not None and getattr(frame, "size", 0) > 0:
            preview = frame.copy()
            _draw_hint_banner(preview, "%s  -  ready" % phase,
                              "press SPACE to start, q to quit", colour)
            cv2.putText(preview, "READY - press SPACE", (10, 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, colour, 2, cv2.LINE_AA)
            cv2.imshow(DEBUG_WINDOW_NAME, preview)
        action = _handle_ui_event()
        if action == "start":
            return None
        if action in ("quit", "recalibrate"):
            return action


def _calibration_cycle(
    history: collections.deque,
    previous_centroid: Optional[tuple[int, int]],
    first_round: bool,
) -> tuple[str, Optional[np.ndarray], Optional[dict[str, np.ndarray | float]],
           Optional[np.ndarray], Optional[tuple[int, int, int, int]]]:
    """执行一轮摄像头预热、背景采集和肤色标定。

    返回 ``(status, cap, background, skin_model, roi)``。状态值可能是
    ok、quit、failed 或 error；状态为 error 时捕获对象已经释放，调用方
    必须以非零状态结束程序。
    """
    cap: Optional[cv2.VideoCapture] = None
    try:
        cap = open_camera(CAMERA_INDEX)
    except RuntimeError as error:
        print("[error] %s" % error)
        return ("error", None, None, None, None)

    try:
        print("[stage] camera opened (index %d)" % CAMERA_INDEX)
        if not warm_up_camera(cap):
            print("[stage] quit during warm-up")
            return ("quit", cap, None, None, None)

        first_frame_ok, first_frame = cap.read()
        if not first_frame_ok or first_frame is None or first_frame.size == 0:
            print("[error] camera stopped delivering frames after warm-up")
            return ("error", cap, None, None, None)
        roi = compute_roi(first_frame.shape)

        print("[stage] stage 2: press SPACE, then keep hands out of the ROI")
        gate = _wait_for_start(cap, "STAGE 2/3  background capture",
                               "keep hands out of the yellow ROI, then press SPACE",
                               COLOR_ROI)
        if gate is not None:
            print("[stage] aborted before background capture (%s)" % gate)
            return ("quit" if gate == "quit" else "failed", cap, None, None, roi)
        background = capture_background(cap, roi, BACKGROUND_FRAMES)
        if background is None:
            print("[stage] background capture aborted, no reference stored")
            return ("quit" if first_round else "failed", cap, None, None, roi)

        print("[stage] stage 3: press SPACE, then cover the 9 cells with the palm")
        gate = _wait_for_start(cap, "STAGE 3/3  skin calibration",
                               "cover all 9 green cells with your palm, then press SPACE",
                               COLOR_OK)
        if gate is not None:
            print("[stage] aborted before skin calibration (%s)" % gate)
            return ("quit" if gate == "quit" else "failed", cap, background, None, roi)
        skin_model = calibrate_skin(cap, roi, SKIN_SAMPLE_FRAMES)
        if skin_model is None:
            print("[stage] skin calibration failed - press r to retry")
            return ("failed", cap, background, None, roi)

        try:
            cap.release()
        finally:
            cap = None
        history.clear()
        _set_history_latency(history, None, 0.0)
        print("[stage] calibration finished, starting recognition")
        return ("ok", None, background, skin_model, roi)
    except Exception as error:  # pragma: no cover - 防御性异常处理
        print("[error] calibration aborted: %s" % error)
        return ("error", cap, None, None, None)


def _run_recognition_loop(
    cap: cv2.VideoCapture,
    roi: tuple[int, int, int, int],
    background: np.ndarray,
    skin_model: dict[str, np.ndarray | float],
    history: collections.deque,
) -> str:
    """运行实时识别循环，返回 ok、quit、error 或 recalibrate。"""
    rows, cols = _roi_slice(roi)
    previous_centroid: Optional[tuple[int, int]] = None
    read_failures = 0
    last_time = time.perf_counter()
    smoothed_fps = 0.0
    last_stable = "UNKNOWN"
    _set_history_latency(history, None, 0.0)

    while True:
        ok, frame = cap.read()
        if not ok or frame is None or frame.size == 0:
            read_failures += 1
            if read_failures >= MAX_CONSECUTIVE_READ_FAILURES:
                print("[error] camera read failed %d times in a row" % read_failures)
                return "error"
            continue
        read_failures = 0

        now = time.perf_counter()
        delta = max(1e-6, now - last_time)
        last_time = now
        instantaneous = 1.0 / delta
        smoothed_fps = instantaneous if smoothed_fps <= 0.0 else \
            0.9 * smoothed_fps + 0.1 * instantaneous

        patch = frame[rows, cols]
        mask = segment_hand(patch, background, skin_model, previous_centroid)
        features = extract_hand_features(mask)
        if isinstance(features, dict):
            palm_center = features.get("palm_center")
            if palm_center is not None:
                previous_centroid = (int(round(float(palm_center[0]))),
                                     int(round(float(palm_center[1]))))
        else:
            previous_centroid = None

        frame_result = classify_open_five(features)
        stable_label, changed = update_stable_state(history, frame_result[0],
                                                    frame_result[1], now)
        if changed and stable_label != last_stable:
            if stable_label == "FIVE":
                print("[state] stable FIVE (frame conf %.2f)" % frame_result[1])
            else:
                print("[state] stable %s" % stable_label)
            last_stable = stable_label
        latency = _history_latency(history)
        if stable_label == "FIVE" and latency > LATENCY_WARNING_SECONDS:
            print("[warn] confirmation took %.2f s (target < %.1f s)"
                  % (latency, LATENCY_TARGET_SECONDS))

        canvas = draw_debug_view(frame, roi, mask, features, frame_result,
                                 stable_label, smoothed_fps)
        if stable_label == "FIVE" and latency > LATENCY_WARNING_SECONDS:
            cv2.putText(canvas, "LATENCY WARNING %.1fs" % latency, (12, canvas.shape[0] - 34),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_BAD, 2)
        _draw_hint_banner(canvas, "STAGE 4/4  recognition  %s" % stable_label,
                          "open hand / palm facing camera / within ROI; "
                          "q quit, r recalibrate",
                          COLOR_OK if stable_label == "FIVE" else COLOR_TEXT)
        cv2.imshow(DEBUG_WINDOW_NAME, canvas)

        action = _handle_ui_event()
        if action == "quit":
            print("[stage] quit requested during recognition")
            return "quit"
        if action == "recalibrate":
            print("[stage] recalibration requested")
            return "recalibrate"


def main() -> int:
    """组织完整程序状态机，并保证所有退出路径都释放资源。

    正常退出返回 0；初始化失败或发生不可恢复的帧读取错误时返回非零值。
    """
    history: collections.deque = collections.deque(maxlen=HISTORY_LENGTH)
    _set_history_latency(history, None, 0.0)

    _create_window()
    # 窗口会在打开摄像头之前出现，此时没有任何帧可显示。若只留一个灰框，
    # 用户在摄像头初始化较慢时会以为程序卡死，因此先显示一张提示画面。
    startup = np.full((CAMERA_HEIGHT, CAMERA_WIDTH, 3), 40, dtype=np.uint8)
    cv2.putText(startup, "opening camera, please wait ...", (24, CAMERA_HEIGHT // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_TEXT, 2, cv2.LINE_AA)
    _draw_hint_banner(startup, "STAGE 0/3  initialising",
                      "this window is alive; waiting for the first camera frame")
    try:
        cv2.imshow(DEBUG_WINDOW_NAME, startup)
        cv2.waitKey(1)
    except Exception:  # pragma: no cover - 防御性异常处理
        pass

    cap: Optional[cv2.VideoCapture] = None
    try:
        status, cap, background, skin_model, roi = _calibration_cycle(history, None, True)
        if status != "ok" or background is None or skin_model is None or roi is None:
            return 0 if status == "quit" else 1

        while True:
            if cap is None:
                try:
                    cap = open_camera(CAMERA_INDEX)
                except RuntimeError as error:
                    print("[error] %s" % error)
                    return 1
            loop_status = _run_recognition_loop(cap, roi, background, skin_model, history)
            if loop_status == "recalibrate":
                _safe_release(cap)
                cap = None
                history.clear()
                _set_history_latency(history, None, 0.0)
                print("[stage] restarting calibration")
                status, cap, background, skin_model, roi = _calibration_cycle(
                    history, None, False)
                if status == "quit":
                    return 0
                if status != "ok" or background is None or skin_model is None or roi is None:
                    return 1
                continue
            if loop_status == "error":
                return 1
            return 0
    finally:
        _safe_release(cap)
        _safe_destroy_windows()


if __name__ == "__main__":
    raise SystemExit(main())
