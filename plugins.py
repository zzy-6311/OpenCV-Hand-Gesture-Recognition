# -*- coding: utf-8 -*-
"""外部插件模块：ROI 限制 + 灰度上下双阈值区间二值化。

main.py 会自动 import 本文件并调用 ``register(api)``，把这里注册的处理函数
追加到插件链末尾。所以主程序一行都不用改。

本模块现在有两个插件，按顺序执行：

    roi 限制 + 灰度双阈值  (gray_threshold)
       1. 取原图 -> 转灰度
       2. 上下双阈值「区间选择」二值化：GRAY_LOW <= 灰度 <= GRAY_HIGH -> 白
       3. 把 ROI 以外的像素一律置零（这是「真正生效」的限制）
    画 ROI 边界 + 触边检查  (roi_boundary)
       4. 把真正生效的 ROI 边界画出来（看到的就是生效的）
       5. 检查前景有没有贴到 ROI 边上，贴了就红色告警
"""

from __future__ import annotations

import cv2
import numpy as np

# ===========================================================================
#
#   ★★★★★  第一部分：ROI 配置（限制识别区域）  ★★★★★
#
# ===========================================================================
#
#  ROI_RELATIVE = (左, 上, 宽, 高)，都是**相对整幅图的比例**（0~1）。
#  默认值 (0.25, 0.10, 0.50, 0.80) 就是「居中的 50% x 80% 矩形」——
#  和 main.py 之前画的那个示例框位置一样，但现在它是**真的会裁掉外面**的。
#
#  按你现在的构图（2560x1440）换算成像素是： x 640~1920, y 144~1296
#  实测这个范围能排掉：纸箱 ✓  黄T恤 ✓  下巴/脸 ✓  深色窗帘 ✓
#  而手在 x 937~1513, y 312~1129，四边余量 168~407 像素，很安全。
#
#  想收得更紧（把背景布也少露一点），可以试着改成：
#      ROI_RELATIVE = (0.33, 0.17, 0.30, 0.70)     # x 845~1613, y 245~1253
#  代价是手移动的容错变小了。
#
ROI_RELATIVE = (0.25, 0.10, 0.50, 0.80)

# 是否启用 ROI 限制。改成 False 就退回「整幅图都参与识别」。
ROI_ENABLED = True

# 是否把 ROI 边界画在画面上。True = 你看到的框就是真正生效的框。
ROI_DRAW_BOUNDARY = True

# ---------------------------------------------------------------------------
#  触边检查：前景贴到 ROI 边界多少像素以内，就算「触边」。
#
#  ⚠️ 默认只检查 left / right / top 三条边，**故意不检查 bottom**。
#     因为手举在身前时，小臂是从下方伸进画面的；如果连下边一起检查，
#     小臂一出画面就会一直告警，反而没用。
#     需要时把 "bottom" 也加到元组里即可。
# ---------------------------------------------------------------------------
ROI_CHECK_EDGES = ("left", "right", "top")
ROI_TOUCH_MARGIN = 6
# ===========================================================================


# ===========================================================================
#
#   ★★★★★  第二部分：灰度双阈值的两个阈值就在这里改  ★★★★★
#
# ===========================================================================
#
#  GRAY_LOW  = 灰度下限（含）。灰度小于它的像素 -> 黑（背景）
#  GRAY_HIGH = 灰度上限（含）。灰度大于它的像素 -> 黑（背景）
#
#  只有   GRAY_LOW <= 灰度 <= GRAY_HIGH   的像素才会变成白（前景）。
#
#  ⚠️ 下限必须小于等于上限。写反了（LOW > HIGH）会自动交换并打印提示，
#     所以不会出现「整个画面全黑」还找不到原因的情况。
#
# ---------------------------------------------------------------------------
#  调参参考：shou1.png 里各区域的实测灰度均值
# ---------------------------------------------------------------------------
#      掌心(偏红最亮)  148        手掌中部  133        拇指  181
#      米白布-亮处     205        米白布-中  194        米白布-暗处  143
#      黄T恤           104        纸箱       122        下巴/脸       62
#      键盘            218
#
#  ⚠️ 手（约 133~181）和米白布（约 143~205）在灰度上是**重叠**的，
#     所以单靠灰度区间无法把手从米白布上分出来。不过 ROI 已经排掉了
#     纸箱/衣服/脸/窗帘这几个大干扰源，剩下的主要对手就是背景布了。
# ---------------------------------------------------------------------------
GRAY_LOW = 100
GRAY_HIGH = 180
# ===========================================================================


# ===========================================================================
#
#   ★★★★★  第三部分：颜色阈值（YCbCr 的 Cr / Cb）就在这里改  ★★★★★
#
# ===========================================================================
#
#  判据（三个条件必须同时满足才算前景=白）：
#      CR_LOW <= Cr <= CR_HIGH
#      CB_LOW <= Cb <= CB_HIGH
#      Y_LOW  <= Y  <= Y_HIGH
#
#  为什么用 YCbCr 而不是灰度/RGB：
#    YCbCr 把「亮度 Y」和「颜色 Cr/Cb」拆开了，所以判「是什么颜色」不受明暗影响。
#    阴影主要改 Y、基本不改 Cr/Cb —— 这正是它能救「手上自阴影导致掩膜出洞」的原因。
#    实测（shou1.png）：灰度对「手 vs 布」的可分性 d'=0.74（基本没用），
#    Cr/Cb 色度是 d'=1.45（静态图像里最好的判据）。
#
#  ⚠️ 通道顺序的坑：
#    cv2.COLOR_BGR2YCrCb 返回的三通道是 **Y, Cr, Cb**
#    （函数名写的是 YCrCb，所以第 1 个通道是 Cr、第 2 个是 Cb），别取反了。
#
# ---------------------------------------------------------------------------
#  调参参考：shou1.png 里各区域实测的 Cr / Cb
# ---------------------------------------------------------------------------
#      区域           Cr       Cb
#      掌心(偏红)    147.9    122.0
#      手掌中部      144.9    118.4
#      拇指          136.2    122.3
#      米白布-左     125.0    130.5     <- 布很中性：Cr 偏低、Cb 偏高
#      米白布-中     124.0    129.5
#      黄T恤         150.0     72.6     <- Cb 极低，靠 Cb 下限排掉
#      纸箱          136.0    107.3     <- 和肤色重叠！颜色分不开，只能靠 ROI 排掉
#      键盘          127.0    129.5
#      下巴/脸       146.6    110.9
#
#  调参思路（关键）：
#    * 手 Cb≈118~123，布 Cb≈129~131  ->  **Cb 上限压到 126~127 就能把布挡掉**
#    * 手 Cr≈136~148，布 Cr≈124~126  ->  **Cr 下限抬到 131~133 也能把布挡掉**
#    两个边界各自都能挡布，所以这个判据相对灰度是「双重保险」。
#    * 想更严（少误检）就把 CR_LOW 往上抬、CB_HIGH 往下压
#    * 想更松（少漏检）就反过来
#    * 手的一部分（如偏白的手指）漏了 -> 说明 CR_LOW 抬太高 或 CB_HIGH 压太狠
#
#  ⚠️ 每一对同样要求 下限 <= 上限，写反了会自动交换并打印提示。
# ---------------------------------------------------------------------------
CR_LOW = 131
CR_HIGH = 173
CB_LOW = 77
CB_HIGH = 128
Y_LOW = 40
Y_HIGH = 245

# ---------------------------------------------------------------------------
#  ★ 上面这组值的实测依据（别再凭感觉调了）★
#
#  实测：**当前这组值已经把掌心完整包住了，放宽阈值买不到完整性。**
#    Cr 下限    133 -> 131 -> 129 -> 127 -> 125 -> 123
#    ROI 内前景  19.8%  21.7%  24.4%  32.6%  62.0%  82.8%
#    掌心覆盖    92.1%  92.6%  93.2%  94.1%  94.8%  98.4%
#
#  → 放宽到 125（布 Cr 的中位数就是 124~126）时前景暴涨到 62%、整幅糊白，
#    而掌心覆盖只多 2.7%。**拐点在 129，再往下布就灌进来了。**
#  → 那个"92.1%"里缺的 8% 是几何矩形框的角落（框是方的、手不是），
#    不是手掌真的缺了。
#
#  所以这里只留了很小的余量（Cr 下限 133→131、Cb 上限 127→128），
#  目的是给别的帧/别的光照一点宽容度，实测代价是噪声 +1.9%。
#  **真正让掩膜变干净要靠下面的清理插件，不是靠放宽阈值。**
# ---------------------------------------------------------------------------
# ===========================================================================


# ===========================================================================
#
#   ★★★★★  第四部分：掩膜清理（开运算 + 面积过滤 + 填洞）就在这里改  ★★★★★
#
# ===========================================================================
#
#  这一层只作用于"掩膜"（显示模式 3），用来把阈值留下的噪声清掉。
#  三步按这个顺序做：
#
#    ① 小核开运算        去掉孤立噪点、磨平毛边
#                        核必须小！大了会把细手指一起吃掉
#    ② 连通域面积过滤    两道门槛同时生效（见下）
#    ③ 填内部孔洞        从图像边界 floodFill 找外部背景，剩下的空洞补上
#                        ★ 只补内部空洞、完全不动外轮廓 —— 这正是"让手掌完整"
#                          想要的效果，而且比闭运算好（闭运算会同时把手指粘起来）
#
#  ⚠️ 面积过滤为什么要两道门槛：
#     实测最大的噪声块是 **ROI 面积的 1.7%**（左下角键鼠那片）。
#     只用"绝对百分比"门槛的话，要么清不掉它（0.5% 太小），
#     要么得调到 3% 才清得掉 —— 但那样手小一点就会被误杀。
#     所以加一道**相对门槛**："比最大连通域小得多就丢"，
#     天然随手的大小自适应，也不怕手指被分成两块时误杀。
# ---------------------------------------------------------------------------
CLEANUP_OPEN_KERNEL = 3              # 开运算核边长（奇数，>=3）。必须小
CLEANUP_MIN_AREA_RATIO = 0.002       # 门槛一（绝对）：小于 ROI 面积这个比例的连通域丢掉
CLEANUP_MIN_AREA_VS_LARGEST = 0.25   # 门槛二（相对）：小于【最大连通域面积】这个比例的也丢掉
CLEANUP_FILL_HOLES = True            # 是否填内部孔洞
# ===========================================================================


# ===========================================================================
#
#   ★★★★★  第五部分：掌心重建（模式 4）就在这里改  ★★★★★
#
# ===========================================================================
#
#  思路：**宽阈值保召回，再用"和掌心颜色相近"把不是手的东西筛掉。**
#
#  关键点：判据全部是**相对掌心核心自算**的，不是写死的绝对阈值！
#    每一帧都先从掌心核心统计出 (Cr0, Cb0, Y0)，再用它们定容差。
#    所以光照变了、肤色变了，基准跟着变 —— **视频里不会抖**。
#    （写死的绝对阈值才会抖：灯一变，同一个像素时进时出。）
#
#  步骤：
#    1. 输入 = 上一级（清理后）的掩膜（手 + 小臂 + 可能的干扰）
#    2. 距离变换取最大值 -> 掌心 C、尺度 R
#    3. 种子 = { DT >= RECON_SEED_DT_FACTOR * R }      <- 掌心核心，实测很干净
#    4. 从种子里统计 Cr0 / Cb0 / Y0（取中位数，抗离群）
#    5. 允许区 = 掩膜 ∩ |Cr-Cr0|<=TOL_CR ∩ |Cb-Cb0|<=TOL_CB ∩ Y >= Y0-TOL_Y_DOWN
#    6. 重建 = 允许区里【与种子相连】的连通域          <- 一次连通域标记即可
#    7. 填内部孔洞
#
#  ⚠️ 第 6 步"重建"必须有第 5 步的颜色约束，否则**完全没用** ——
#     实测：如果只按连通性重建（允许区 = 掩膜本身），影子是 100% 保留的，
#     因为它在掩膜里和手是连着的。颜色约束的作用是**把那根"连接"切断**，
#     于是重建就长不到影子上去了。
#
#  ⚠️ 关于容差取值（实测，shou1.png，同一个掌心核心）：
#      TOL_CR=8  -> 手指保留 75.2%、影子去掉 65%
#      TOL_CR=10 -> 手指保留 85.6%、影子去掉 60%
#      TOL_CR=12 -> 手指保留 96.1%、影子去掉 45%   <- 当前选这个，优先保手指
#     影子去不干净是**颜色信息本身的天花板**：手的浅色/高光部分和布在颜色上重叠，
#     任何纯颜色方法都会在"去影子"和"保手指"之间做权衡。
# ---------------------------------------------------------------------------
RECON_SEED_DT_FACTOR = 0.7      # 种子 = DT >= 该比例 × R（0.7 是实测的甜点）
RECON_TOL_CR = 12               # Cr 容差（相对掌心核心中位数 Cr0）
RECON_TOL_CB = 15               # Cb 容差（Cb 分不开影子，给宽松值当安全边界）
RECON_TOL_Y_DOWN = 75           # Y 允许比掌心核心中位数 Y0 暗多少
RECON_FILL_HOLES = True         # 重建后填内部孔洞
RECON_MIN_SEED_PIXELS = 40      # 种子小于这么多像素就放弃重建（返回原掩膜）
RECON_MIN_RADIUS_PX = 8         # 掌心半径小于此值也放弃

# ---------------------------------------------------------------------------
#  ★ 上面这组值的实测依据（shou1.png，TOL_CR=12 固定，只扫 TOL_Y_DOWN）★
#
#   TOL_Y_DOWN   手指保留   掌心核心   小臂保留   影子带去掉了
#      120        96.9%      100%      99.0%      45%
#       90        96.9%      100%      97.2%      57%
#       75        82.2%      100%      92.2%      79%   <-- 当前值，拐点
#       60        72.7%      100%      80.4%      83%
#       40        60.6%      100%      69.8%      90%
#       30        54.0%      100%      65.8%      93%
#
#  → **拐点在 90~75 之间**：影子从去掉 57% 跳到 79%，手指从 96.9% 掉到 82.2%。
#    再往下压（75 → 60 → 40）影子只多去 4~11 个百分点，手指却掉得很快。
#    所以 75 是"每牺牲一点手指能换多少影子"最划算的位置。
#
#  → 想要更干净：把 TOL_Y_DOWN 调到 60（影子去 83%，手指剩 72.7%）
#    想要更保手指：调到 90（手指 96.9%，但影子只去 57%）
#    注意：**掌心核心在任何设置下都是 100%**，所以调这个不影响掌心定位。
#
#  ⚠️ 剩下一部分影子去不掉是**颜色信息本身的天花板**：手的浅色/高光部分和布
#     在颜色上重叠，任何纯颜色方法都要在"去影子"和"保手指"之间做权衡。
#     要突破这个上限只能靠正交信息（背景差分 / 梯度 / 时序）。
# ---------------------------------------------------------------------------
# ===========================================================================


# ===========================================================================
#
#   ★★★★★  第六部分：识别骨架（模式 5 / 6）就在这里改  ★★★★★
#
# ===========================================================================
#
#  目标：**先不做分类**，只把手部各个部位定位出来并画给你看，
#        让你能亲眼确认"5 根手指找得对不对"。
#
#  步骤（都在 step_hand_recognize 里）：
#    1. 输入 = 模式 4 重建后的掩膜（手 + 小臂）
#    2. 掌心 C 与尺度 R 直接取模式 4 算好的（拿不到就自己算一遍）
#    3. 手腕 W = 手部**底端带状区域**的质心
#    4. **前臂截断**：把掌心下方 1.5R 以外全裁掉 -> 只剩手
#    5. 在"只剩手"的轮廓上找**指尖**：
#       轮廓点到掌心的距离剖面局部极大
#       + 突出度门槛 + 转折比/夹角门槛 + 内部厚度门槛
#    6. 在相邻指尖之间找**指缝**（凹谷）
#    7. 每根手指的**指根** = 它两侧指缝的中点 -> 指根到指尖连成一条"手指"
#
#  ⚠️ 为什么先截前臂：前臂会在轮廓上形成额外顶点，凸包/凹谷判断会被它污染。
#     截断之后轮廓才是"纯手"，指尖和指缝才准。
#     截断产生的两个直角在**掌心下方**，会被"指尖必须在掌上方"这条过滤掉。
#
#  ⚠️ 全部长度阈值都按 **R 归一化**，所以拍摄距离变了不用改参数。
# ---------------------------------------------------------------------------
RECOG_WRIST_BAND_RATIO = 0.06      # 手腕取手部底端这段比例的带状区域质心
RECOG_WRIST_MIN_PIXELS = 12        # 带状区域像素太少就不估手腕
RECOG_TRUNCATE_FACTOR = 1.5        # 在掌心下方这么多 R 处截断前臂
RECOG_TRUNCATE_LINE_HALF = 1.6     # ★只影响"截断线画多长"（× R）。
                                   #   截断本身是半平面（见 _truncate_forearm），
                                   #   那根线只是画出来给你看，画 8R 长会横穿整个画面。

RECOG_TIP_MIN_DISTANCE = 1.25      # 指尖到掌心至少这么多 R（低于此的不算指尖）
RECOG_TIP_PEAK_WINDOW = 0.45       # 找"距离剖面局部极大"的窗口（× R）
RECOG_TIP_MIN_PROMINENCE = 0.20    # 局部极大的突出度下限（× R）
RECOG_TIP_MIN_TURN_RATIO = 0.10    # 转折比（弯曲深度/弦长）下限，越小越尖
RECOG_TIP_MAX_ANGLE_DEG = 145.0    # 轮廓夹角上限，超过就是钝的、不像指尖
RECOG_TIP_THICKNESS_SAMPLE = 0.2   # 往轮廓内部取厚度样本的比例（× 到掌心的距离）
RECOG_TIP_THICKNESS_MAX = 0.45     # 内部厚度比上限，太厚说明是掌缘/前臂不是手指
# ★"针状毛刺"判据：夹角和内部厚度**同时**极端小才拒绝。
#   实测轮廓上会出现 1~2 像素的针状凸起，它的夹角只有 1°、内部厚度 0.00，
#   而真指尖是"圆帽"——夹角 46~58°、厚度 0.08~0.19。
#   要求两个条件**同时**成立才拒，是为了不误伤真正尖锐的指尖。
RECOG_TIP_MIN_ANGLE_DEG = 20.0     # 夹角小于此值算"针状"
RECOG_TIP_MIN_THICKNESS = 0.05     # 内部厚度比小于此值算"没有肉"
RECOG_TIP_DEDUP = 0.35             # 距离小于这么多 R 的候选视为同一指尖
RECOG_TIP_MAX_COUNT = 6            # 最多留几个指尖候选
RECOG_TIP_PALM_SIDE_MAX = 0.15     # 局部系里 y 超过这个值就算"跑到掌下方了"，丢掉

RECOG_GAP_MIN_DEPTH = 0.25         # 指缝凹谷深度下限（× R）
RECOG_GAP_MAX_ANGLE_DEG = 105.0    # 凹谷夹角上限
RECOG_GAP_DEDUP = 0.35             # 指缝去重距离（× R）
RECOG_GAP_MAX_COUNT = 5            # 最多留几条指缝

RECOG_MASK_TRUNCATE = True         # 模式 5 是否显示"截断后（只剩手）"的掩膜
# ===========================================================================


# 插件在插件链里的名字（按 l 键会打印出来的那个名字）
PLUGIN_NAME = "gray_threshold"          # 灰度上下双阈值（已停用，代码保留）
COLOR_PLUGIN_NAME = "color_threshold"   # YCbCr 色度（Cr/Cb）双通道区间
CLEANUP_PLUGIN_NAME = "mask_cleanup"    # 开运算 + 面积过滤 + 填洞
RECON_PLUGIN_NAME = "palm_reconstruct"  # 从掌心核心做颜色约束重建
RECOG_PLUGIN_NAME = "hand_recognize"    # 识别骨架：截前臂 + 指尖 + 指缝（算）
DRAW_MASK_PLUGIN_NAME = "recognize_on_mask"      # 把识别结果画在掩膜上
DRAW_ORIG_PLUGIN_NAME = "recognize_on_original"  # 把识别结果画在原图上
ROI_PLUGIN_NAME = "roi_boundary"        # 画 ROI 边界 + 触边检查

# 默认启用状态。真正的开关在 main.py 的 VIEW_MODES（按 1/2/3/4/5/6 切换）。
# 灰度版按你的要求**停用**，只是注册着、代码保留。
GRAY_ENABLED_DEFAULT = False
COLOR_ENABLED_DEFAULT = True
CLEANUP_ENABLED_DEFAULT = True
RECON_ENABLED_DEFAULT = True
RECOG_ENABLED_DEFAULT = True
DRAW_ENABLED_DEFAULT = True

# 取哪张图做灰度：
#   False = 用传进来的 frame（插件链上游处理过的结果）—— 推荐，也是当前值
#   True  = 用 state.raw_frame（未经任何插件改动的原图）
#
# ★ 为什么这里是 False：
#   如果改成 True，本插件就完全无视上游的处理结果、自己从原图重算一遍，
#   于是**上游插件的效果会被全部丢掉** —— 表现就是按 m（镜像）/ d（信息叠加）
#   / r（ROI 参考框）画面毫无变化，像是"按键失效了"。实测过：差异 0 像素。
#
#   之前设成 True 是为了躲开 main.py 内置插件画的标注（黄色框灰度约 179，
#   会被阈值当成前景糊进掩膜）。那个问题现在从**结构上**解决了：
#   main.py 把绘制类插件挪到了插件链最后（add_display_plugins），
#   所以轮到本插件时画面上还没有任何标注，不需要再躲。
USE_RAW_FRAME = False

# 颜色（BGR）
COLOR_BOUNDARY = (0, 200, 255)      # ROI 边界：橙黄
COLOR_BOUNDARY_BAD = (0, 0, 255)    # 触边时的 ROI 边界：红
COLOR_TEXT = (255, 255, 255)


# ===========================================================================
# ROI 相关工具
# ===========================================================================
def roi_rect_pixels(shape_hw: tuple) -> tuple:
    """把 ROI_RELATIVE 换算成像素矩形 ``(x0, y0, x1, y1)``（右/下为开区间）。"""
    height, width = int(shape_hw[0]), int(shape_hw[1])
    left, top, rect_w, rect_h = ROI_RELATIVE
    x0 = int(round(float(left) * width))
    y0 = int(round(float(top) * height))
    x1 = int(round((float(left) + float(rect_w)) * width))
    y1 = int(round((float(top) + float(rect_h)) * height))
    # 夹回画面内，并保证至少有 1 个像素，避免手写错比例导致空掩膜
    x0 = max(0, min(x0, width - 1))
    y0 = max(0, min(y0, height - 1))
    x1 = max(x0 + 1, min(x1, width))
    y1 = max(y0 + 1, min(y1, height))
    return x0, y0, x1, y1


# ROI 掩膜缓存。它只取决于画面尺寸和 ROI 配置，每帧重建纯属浪费，
# 所以键里带上这几个量，变了才重建。
_ROI_CACHE: dict = {}


def roi_enabled_now(state) -> bool:
    """当前 ROI 是否启用。

    ``state.flags["roi_enabled"]`` 优先 —— 这样 main.py 的 r 键可以**实时**
    开关真正生效的 ROI（按一下就能对比"限制/不限制"的效果）。
    没设过就退回模块常量 ROI_ENABLED。
    """
    if state is not None and "roi_enabled" in getattr(state, "flags", {}):
        return bool(state.flags["roi_enabled"])
    return bool(ROI_ENABLED)


def _roi_cache_key(shape_hw: tuple, channels: int, enabled: bool) -> tuple:
    return (int(shape_hw[0]), int(shape_hw[1]), int(channels),
            bool(enabled), tuple(ROI_RELATIVE))


def build_roi_mask(shape_hw: tuple, enabled: bool = None) -> np.ndarray:
    """生成 0/255 的**单通道** ROI 掩膜（带缓存）。

    ``enabled`` 传 None 就用模块常量 ROI_ENABLED。
    ⚠️ 返回的是缓存对象，调用方**只读**，不要就地修改它。
    """
    if enabled is None:
        enabled = ROI_ENABLED
    key = _roi_cache_key(shape_hw, 1, enabled)
    cached = _ROI_CACHE.get(key)
    if cached is not None:
        return cached

    height, width = int(shape_hw[0]), int(shape_hw[1])
    mask = np.zeros((height, width), dtype=np.uint8)
    if not enabled:
        mask[:] = 255
    else:
        x0, y0, x1, y1 = roi_rect_pixels((height, width))
        mask[y0:y1, x0:x1] = 255
    _ROI_CACHE[key] = mask
    return mask


def build_roi_mask_bgr(shape_hw: tuple, enabled: bool = None) -> np.ndarray:
    """三通道版 ROI 掩膜，配合 ``cv2.bitwise_and`` 用（同样带缓存，只读）。"""
    if enabled is None:
        enabled = ROI_ENABLED
    key = _roi_cache_key(shape_hw, 3, enabled)
    cached = _ROI_CACHE.get(key)
    if cached is not None:
        return cached
    mask3 = cv2.cvtColor(build_roi_mask(shape_hw, enabled), cv2.COLOR_GRAY2BGR)
    _ROI_CACHE[key] = mask3
    return mask3


def find_touched_edges(frame_or_mask: np.ndarray, shape_hw: tuple,
                       enabled: bool = None) -> list:
    """检查前景是否贴到 ROI 需要检查的边上，返回触到的边名列表。

    只扫描 ROI 边缘那条几个像素宽的窄带，**不做全图布尔运算** ——
    在 2560x1440 上建一个全图 bool 数组要几十毫秒，而只看窄带只要几微秒。
    单通道 / 三通道都能传。
    """
    if enabled is None:
        enabled = ROI_ENABLED
    if not enabled:
        return []
    x0, y0, x1, y1 = roi_rect_pixels(shape_hw)
    margin = max(1, int(ROI_TOUCH_MARGIN))
    view = frame_or_mask[:, :, 0] if frame_or_mask.ndim == 3 else frame_or_mask
    touched = []
    if "left" in ROI_CHECK_EDGES and np.any(view[y0:y1, x0:min(x0 + margin, x1)]):
        touched.append("left")
    if "right" in ROI_CHECK_EDGES and np.any(view[y0:y1, max(x1 - margin, x0):x1]):
        touched.append("right")
    if "top" in ROI_CHECK_EDGES and np.any(view[y0:min(y0 + margin, y1), x0:x1]):
        touched.append("top")
    if "bottom" in ROI_CHECK_EDGES and np.any(view[max(y1 - margin, y0):y1, x0:x1]):
        touched.append("bottom")
    return touched


# ===========================================================================
# 两个阈值插件共用的辅助
# ===========================================================================
def _normalized_pair(low, high, label: str) -> tuple:
    """把一对上下限规整好；写反了自动交换并提示，省得对着全黑画面找原因。"""
    low, high = int(low), int(high)
    if low > high:
        print("[plugins] 警告：%s 的下限(%d) > 上限(%d)，已自动交换"
              % (label, low, high))
        low, high = high, low
    return low, high


def _finish_mask(mask: np.ndarray, state, method: str) -> np.ndarray:
    """两个阈值插件共用的收尾：应用 ROI 限制 + 记录统计 + 转三通道。

    阈值 -> 应用 ROI 的顺序不能反：先阈值再置零，才能保证不管阈值怎么设
    （哪怕下限设成 0），ROI 外的像素都一定是黑的。
    """
    height, width = mask.shape[:2]
    stats = {}
    enabled = roi_enabled_now(state)
    if enabled:
        # 用 bitwise_and 而不是 mask[roi == 0] = 0：后者要先建全图 bool 数组再做
        # 花式索引，2560x1440 上要几十毫秒；bitwise_and 是 SIMD，快一个数量级。
        cv2.bitwise_and(mask, build_roi_mask((height, width), enabled), dst=mask)
        x0, y0, x1, y1 = roi_rect_pixels((height, width))
        stats["roi_rect"] = (x0, y0, x1, y1)
        stats["roi_area_ratio"] = float((x1 - x0) * (y1 - y0)) / float(height * width)
    stats["roi_enabled"] = enabled
    stats["method"] = method
    stats["foreground_ratio"] = float(np.count_nonzero(mask)) / float(mask.size)
    state.flags["segment_stats"] = stats
    return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)


def _threshold_source(frame: np.ndarray, state) -> np.ndarray:
    """按 USE_RAW_FRAME 决定用传进来的帧还是原图。"""
    if USE_RAW_FRAME and getattr(state, "raw_frame", None) is not None:
        return state.raw_frame
    return frame


# ===========================================================================
# 处理方法 A：灰度 + 上下双阈值区间二值化
# ===========================================================================
def step_gray_threshold(frame: np.ndarray, state) -> np.ndarray:
    """灰度化 -> 上下双阈值区间二值化 -> 把 ROI 外置零。

    输入：``frame`` 是插件链传到本级的图像（BGR uint8）。
    输出：三通道二值图，前景为白 (255,255,255)，其余为黑 (0,0,0)。
    """
    gray = cv2.cvtColor(_threshold_source(frame, state), cv2.COLOR_BGR2GRAY)
    low, high = _normalized_pair(GRAY_LOW, GRAY_HIGH, "GRAY_LOW/GRAY_HIGH")
    mask = cv2.inRange(gray, low, high)
    return _finish_mask(mask, state, "gray")


# ===========================================================================
# 处理方法 B：YCbCr 色度（Cr / Cb）双通道区间二值化
# ===========================================================================
def step_color_threshold(frame: np.ndarray, state) -> np.ndarray:
    """YCbCr 色度阈值：Cr、Cb、Y 三段区间同时满足才算前景。

    和灰度版的区别只在判据：这里用的是**颜色**而不是亮度，
    所以对阴影不敏感（阴影改 Y、基本不改 Cr/Cb）。
    """
    ycrcb = cv2.cvtColor(_threshold_source(frame, state), cv2.COLOR_BGR2YCrCb)
    y_channel = ycrcb[:, :, 0]
    cr_channel = ycrcb[:, :, 1]     # 注意：BGR2YCrCb 的第 1 通道是 Cr
    cb_channel = ycrcb[:, :, 2]     # 第 2 通道是 Cb

    cr_lo, cr_hi = _normalized_pair(CR_LOW, CR_HIGH, "CR_LOW/CR_HIGH")
    cb_lo, cb_hi = _normalized_pair(CB_LOW, CB_HIGH, "CB_LOW/CB_HIGH")
    y_lo, y_hi = _normalized_pair(Y_LOW, Y_HIGH, "Y_LOW/Y_HIGH")

    # 三个区间取交集。用 bitwise_and 而不是 numpy 的 & 再 astype，
    # 少两次全图临时数组，而且和后面的 ROI 处理风格一致。
    mask = cv2.inRange(cr_channel, cr_lo, cr_hi)
    cv2.bitwise_and(mask, cv2.inRange(cb_channel, cb_lo, cb_hi), dst=mask)
    cv2.bitwise_and(mask, cv2.inRange(y_channel, y_lo, y_hi), dst=mask)

    # ★把 YCrCb 图挂到 state 上，供下游（掌心重建）取颜色用。
    #   必须是**这一张**（和掩膜同源、同一次镜像），不能改用 state.raw_frame ——
    #   镜像开着的时候 raw_frame 没被翻，坐标会和掩膜错位。
    #   这里只存引用、不拷贝，零成本；下一帧会被覆盖。
    state.flags["segment_ycrcb"] = ycrcb
    # ★同样把"这一帧的彩色原图"存下来，供模式 6（把识别结果画在原图上）用。
    #   理由同上：必须和掩膜同源，否则镜像时标注会错位。
    state.flags["segment_source_bgr"] = _threshold_source(frame, state)
    return _finish_mask(mask, state, "color")


# ===========================================================================
# 清理：小核开运算 -> 连通域面积过滤 -> 填内部孔洞
# ===========================================================================
def _fill_internal_holes(binary: np.ndarray) -> np.ndarray:
    """填掉前景**内部**的孔洞，完全不动外轮廓。

    做法：往外补一圈背景，再从角上 floodFill 把「外部背景」涂成 255。
    补的那一圈保证「所有和图像边界相连的背景」都是连通的，
    于是 floodFill 之后仍然是 0 的像素，就是被前景完全包住的内部空洞。

    ⚠️ 为什么要补一圈：如果只在原图上从 (0,0) floodFill，而前景恰好把
    画面上下切断（手贴到 ROI 左右边时就会这样），下半部分的背景就到不了，
    会被误判成"空洞"整片填白。补一圈就彻底避免了这个坑。
    """
    height, width = binary.shape[:2]
    padded = cv2.copyMakeBorder(binary, 1, 1, 1, 1,
                                cv2.BORDER_CONSTANT, value=0)
    flood_mask = np.zeros((height + 4, width + 4), np.uint8)
    cv2.floodFill(padded, flood_mask, (0, 0), 255)
    holes = cv2.bitwise_not(padded)[1:-1, 1:-1]
    return cv2.bitwise_or(binary, holes)


def roi_area_pixels(shape_hw: tuple, enabled: bool = None) -> int:
    """ROI 的像素面积（用来算绝对面积门槛）。ROI 关掉时就是整幅图。"""
    if enabled is None:
        enabled = ROI_ENABLED
    height, width = int(shape_hw[0]), int(shape_hw[1])
    if not enabled:
        return height * width
    x0, y0, x1, y1 = roi_rect_pixels((height, width))
    return int((x1 - x0) * (y1 - y0))


def step_mask_cleanup(frame: np.ndarray, state) -> np.ndarray:
    """把阈值留下的噪声清掉：开运算 -> 面积过滤 -> 填洞。

    输入是上一级阈值插件产出的二值掩膜（三通道黑/白）。
    输出同样格式，但更干净。
    """
    single = frame[:, :, 0] if frame.ndim == 3 else frame
    mask = single.copy()
    height, width = mask.shape[:2]
    enabled = roi_enabled_now(state)

    # ---- ① 小核开运算：去孤立噪点 ----
    kernel_size = int(CLEANUP_OPEN_KERNEL)
    if kernel_size >= 3:
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                           (kernel_size, kernel_size))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # ---- ② 连通域面积过滤（两道门槛取较大者）----
    removed_components = 0
    kept_components = 0
    min_area_used = 0.0
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if count > 1:
        areas = stats[1:, cv2.CC_STAT_AREA].astype(np.float64)
        largest = float(areas.max())
        absolute_min = float(CLEANUP_MIN_AREA_RATIO) * float(roi_area_pixels((height, width), enabled))
        relative_min = float(CLEANUP_MIN_AREA_VS_LARGEST) * largest
        min_area_used = max(1.0, absolute_min, relative_min)

        # 用查找表做筛选，比 np.isin 快得多
        lut = np.zeros(count, np.uint8)
        for label in range(1, int(count)):
            if float(stats[label, cv2.CC_STAT_AREA]) >= min_area_used:
                lut[label] = 255
                kept_components += 1
            else:
                removed_components += 1
        mask = lut[labels] if kept_components else np.zeros_like(mask)

    # ---- ③ 填内部孔洞 ----
    filled_pixels = 0
    if CLEANUP_FILL_HOLES and kept_components:
        before = int(np.count_nonzero(mask))
        mask = _fill_internal_holes(mask)
        filled_pixels = int(np.count_nonzero(mask)) - before

    # 更新统计（保留阈值那级写下的 method / roi 信息，补上清理的痕迹）
    segment_stats = dict(state.flags.get("segment_stats") or {})
    segment_stats["cleaned"] = True
    segment_stats["foreground_ratio"] = float(np.count_nonzero(mask)) / float(mask.size)
    segment_stats["removed_components"] = removed_components
    segment_stats["kept_components"] = kept_components
    segment_stats["min_area_used"] = min_area_used
    segment_stats["filled_pixels"] = filled_pixels
    state.flags["segment_stats"] = segment_stats

    return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)


# ===========================================================================
# 掌心重建：宽阈值保召回，再用"和掌心颜色相近"把不是手的筛掉
# ===========================================================================
def step_palm_reconstruct(frame: np.ndarray, state) -> np.ndarray:
    """从掌心核心做「颜色约束重建」，得到干净的手部掩膜。

    输入是上一级清理过的掩膜（三通道黑/白）。
    输出同样格式；判据全部相对掌心核心自算，所以不会随光照抖动。
    """
    single = frame[:, :, 0] if frame.ndim == 3 else frame
    mask = single.copy()
    height, width = mask.shape[:2]

    ycrcb = state.flags.get("segment_ycrcb")
    if ycrcb is None:
        # 没拿到颜色图（比如没开颜色阈值那一层）就原样返回，不要崩
        print("[%s] 没有可用的 YCrCb 图（上游没跑颜色阈值？），跳过重建"
              % RECON_PLUGIN_NAME)
        return frame

    # ---- 1) 距离变换 -> 掌心 C 与尺度 R ----
    distance = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    _, radius, _, location = cv2.minMaxLoc(distance)
    if radius < float(RECON_MIN_RADIUS_PX):
        print("[%s] 掌心半径只有 %.1f，太小，跳过重建" % (RECON_PLUGIN_NAME, radius))
        return frame
    cx, cy = int(location[0]), int(location[1])

    # ---- 2) 种子 = 掌心核心 ----
    seed = (distance >= float(RECON_SEED_DT_FACTOR) * radius)
    seed_pixels = int(np.count_nonzero(seed))
    if seed_pixels < int(RECON_MIN_SEED_PIXELS):
        print("[%s] 种子只有 %d px，太小，跳过重建" % (RECON_PLUGIN_NAME, seed_pixels))
        return frame

    # ---- 3) 从种子统计颜色基准（中位数，抗离群）----
    def median_of(plane):
        return float(np.median(plane[seed]))

    cr_plane = ycrcb[:, :, 1]
    cb_plane = ycrcb[:, :, 2]
    y_plane = ycrcb[:, :, 0]
    cr0 = median_of(cr_plane)
    cb0 = median_of(cb_plane)
    y0 = median_of(y_plane)

    # ---- 4) 允许区 = 掩膜 ∩ 三个"相对掌心核心"的区间 ----
    #     注意：上下限都是 (Cr0 ± 容差) 算出来的，每帧自适应。
    cr_lo = int(round(max(0.0, cr0 - float(RECON_TOL_CR))))
    cr_hi = int(round(min(255.0, cr0 + float(RECON_TOL_CR))))
    cb_lo = int(round(max(0.0, cb0 - float(RECON_TOL_CB))))
    cb_hi = int(round(min(255.0, cb0 + float(RECON_TOL_CB))))
    y_lo = int(round(max(0.0, y0 - float(RECON_TOL_Y_DOWN))))

    allowed = cv2.inRange(cr_plane, cr_lo, cr_hi)
    cv2.bitwise_and(allowed, cv2.inRange(cb_plane, cb_lo, cb_hi), dst=allowed)
    cv2.bitwise_and(allowed, cv2.inRange(y_plane, y_lo, 255), dst=allowed)
    cv2.bitwise_and(allowed, mask, dst=allowed)

    # ---- 5) 重建 = 允许区里【与种子相连】的连通域 ----
    #     数学上「形态学重建」就等于这一步，一次连通域标记即可，
    #     不用真去做迭代膨胀（那样几百轮，慢几百倍）。
    count, labels, stats, _ = cv2.connectedComponentsWithStats(allowed, 8)
    seed_labels = set(int(v) for v in np.unique(labels[seed])) - {0}
    if not seed_labels:
        print("[%s] 允许区和种子不相连（容差是不是调太小了？），跳过重建"
              % RECON_PLUGIN_NAME)
        return frame
    lookup = np.zeros(count, np.uint8)
    for label in seed_labels:
        lookup[label] = 255
    result = lookup[labels]

    # ---- 6) 填内部孔洞 ----
    if RECON_FILL_HOLES:
        result = _fill_internal_holes(result)

    # ---- 7) 统计 ----
    segment_stats = dict(state.flags.get("segment_stats") or {})
    segment_stats["reconstructed"] = True
    segment_stats["palm_center"] = (cx, cy)
    segment_stats["palm_radius"] = float(radius)
    segment_stats["seed_pixels"] = seed_pixels
    segment_stats["color_base"] = (cr0, cb0, y0)
    segment_stats["foreground_ratio"] = float(np.count_nonzero(result)) / float(result.size)
    state.flags["segment_stats"] = segment_stats

    return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)


# ===========================================================================
# 识别骨架：截前臂 -> 找指尖 -> 找指缝 -> 指根
# ===========================================================================
def _to_local_points(points, origin, up, scale) -> np.ndarray:
    """把点转到手部局部坐标系并按 R 归一化。

    x 为手掌左右方向，**y 为指向手腕为正**（所以指尖的 y 为负）。
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if pts.size == 0:
        return np.zeros((0, 2), dtype=np.float64)
    right = np.array([-float(up[1]), float(up[0])], dtype=np.float64)
    delta = pts - np.asarray(origin, dtype=np.float64).reshape(1, 2)
    safe = float(scale) if abs(float(scale)) > 1e-6 else 1.0
    return np.stack([delta @ right / safe, -(delta @ up) / safe], axis=1)


def _contour_point(contour, index) -> np.ndarray:
    """按环形索引取轮廓点（float64）。"""
    count = int(contour.shape[0])
    return contour[int(index) % count].reshape(2).astype(np.float64)


def _turning_ratio(contour, index: int, window: int) -> float:
    """轮廓在该点的「转折比」= 弯曲深度 / 弦长。

    直线段接近 0，指尖圆帽明显更大。**不依赖像素尺度**，所以不用绝对阈值。
    """
    here = _contour_point(contour, index)
    before = _contour_point(contour, index - window)
    after = _contour_point(contour, index + window)
    chord = after - before
    chord_length = float(np.linalg.norm(chord))
    if chord_length < 1e-6:
        return 0.0
    offset = here - before
    perpendicular = abs(float(offset[0] * chord[1] - offset[1] * chord[0])) / chord_length
    return float(np.clip(perpendicular / chord_length, 0.0, 1.0))


def _contour_angle(contour, index: int, window: int) -> float:
    """轮廓在该点的夹角（度）。越接近 180 越平，越小越尖。"""
    here = _contour_point(contour, index)
    before = _contour_point(contour, index - window) - here
    after = _contour_point(contour, index + window) - here
    na = float(np.linalg.norm(before))
    nb = float(np.linalg.norm(after))
    if na < 1e-6 or nb < 1e-6:
        return 180.0
    cos_value = float(np.clip(np.dot(before, after) / (na * nb), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_value)))


def _tip_sharpness(contour, index: int, window: int) -> tuple:
    """多尺度地量"这里尖不尖"，返回（转折比, 夹角）。"""
    windows = [max(2, int(round(window * factor)))
               for factor in (0.6, 1.0, 1.5)]
    ratios = [_turning_ratio(contour, index, w) for w in windows]
    angles = [_contour_angle(contour, index, w) for w in windows]
    # 小窗口看得到指尖圆帽的明显弯曲，大窗口会把它平均掉：
    # 所以转折比取最尖的读数，夹角取中位数（更稳）
    return float(np.max(ratios)), float(np.median(angles))


def _sample_plane(plane: np.ndarray, point: np.ndarray) -> float:
    """取某个浮点坐标处的像素值（就近取整，越界夹回）。"""
    height, width = plane.shape[:2]
    x = int(np.clip(round(float(point[0])), 0, width - 1))
    y = int(np.clip(round(float(point[1])), 0, height - 1))
    return float(plane[y, x])


def _estimate_wrist(mask: np.ndarray, palm_center, radius: float):
    """手腕 W = 手掌下方**横截面最窄**的那一行。

    ⚠️ 之前用的是"手部底端带状区域的质心"（solution.py 的做法）。但那个做法
    默认掩膜在手腕处就结束了 —— 而我们的掩膜里含**整条小臂**，所以底端质心
    取到的是小臂尖，实测跑到 (1522,1265)，离真正的手腕差了 400 多像素。

    改成找**宽度剖面的局部极小**：手掌往下、宽度会先收到最窄（手腕束腰），
    再进小臂又变宽。取那个"离掌心最近的、明显比最宽处窄的"局部极小即可。

    ``mask`` 必须是**含小臂**的（截断之前），否则找不到束腰。
    """
    height, width = mask.shape[:2]
    row_widths = np.count_nonzero(mask, axis=1)
    rows = np.nonzero(row_widths)[0]
    if rows.size == 0:
        return None
    bottom_row = int(rows[-1])
    cy = int(round(float(palm_center[1])))

    # 只从掌心下方约 0.8R 处开始找（再往上就是手掌本身，不是束腰）
    start = min(bottom_row, cy + int(round(0.8 * float(radius))))
    if bottom_row - start < 3:
        return None
    segment = row_widths[start:bottom_row + 1].astype(np.float32)
    if segment.size < 3:
        return None

    # 先平滑，免得单行的椒盐噪声当选
    kernel = max(3, int(round(0.20 * float(radius))) | 1)
    smoothed = cv2.blur(segment.reshape(-1, 1), (1, kernel)).ravel()
    peak = float(smoothed.max())
    if peak <= 0.0:
        return None

    left = np.concatenate([[smoothed[0]], smoothed[:-1]])
    right = np.concatenate([smoothed[1:], [smoothed[-1]]])
    is_min = (smoothed <= left) & (smoothed <= right) & (smoothed < 0.88 * peak)
    candidates = np.nonzero(is_min)[0]
    # 有束腰就取离掌心最近的那个；没有就退回"整段最窄的一行"
    best = int(candidates[0]) if candidates.size else int(np.argmin(smoothed))
    wrist_row = start + best

    xs = np.nonzero(mask[wrist_row])[0]
    if xs.size < int(RECOG_WRIST_MIN_PIXELS):
        return None
    center = np.array([0.5 * (float(xs.min()) + float(xs.max())), float(wrist_row)],
                      dtype=np.float64)
    # 手腕必须确实在掌心下方，否则说明定位有问题
    if center[1] <= float(palm_center[1]):
        return None
    return center


def _truncate_forearm(mask: np.ndarray, palm_center, up, radius: float) -> np.ndarray:
    """前臂截断：只保留"掌心 + 手指那一侧"的**半平面**。

    ⚠️ 这里踩过一个严重的坑，务必保留这个写法：
       最初把多边形边距写成 `4R`（约 594 px）。结果多边形在**手指方向**也只
       延伸了 594 px，而指尖离截断线有 601 px —— **整片指尖被削平了**。
       表现在掩膜上就是出现一条水平直边，那条直边上的毛刺被当成了 4 个"指尖"，
       于是指尖数从 5 变成 6，而且指根/长度全乱。
       判据：手指区每行游程数在 y=330~390 全是 0，y=400 才突然冒出 3~5 个。

       所以沿**两个方向**都要延伸到远大于图像尺寸 —— 等价于一个半平面，
       这样无论手举多高、指尖离掌心多远都不会被误切。
    """
    height, width = mask.shape[:2]
    far = float(max(height, width)) * 2.0     # 远大于图像，保证覆盖整个手指侧
    origin = np.asarray(palm_center, dtype=np.float64) - \
        np.asarray(up, dtype=np.float64) * (float(RECOG_TRUNCATE_FACTOR) * float(radius))
    along = np.array([-float(up[1]), float(up[0])], dtype=np.float64)
    polygon = np.array([
        origin + along * far,
        origin - along * far,
        origin - along * far + np.asarray(up) * far,
        origin + along * far + np.asarray(up) * far,
    ], dtype=np.int32)
    allowed = np.zeros((height, width), dtype=np.uint8)
    cv2.fillConvexPoly(allowed, polygon, 255)
    return cv2.bitwise_and(mask, allowed)


def _detect_fingertips(contour, mask: np.ndarray, palm_center, radius: float, up):
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
    idx = np.arange(count)
    floor_max = distances.copy()      # 环形邻域最大值
    floor_min = distances.copy()      # 环形邻域最小值（算突出度用）
    for offset in range(1, window + 1):
        np.maximum(floor_max, distances[(idx + offset) % count], out=floor_max)
        np.maximum(floor_max, distances[(idx - offset) % count], out=floor_max)
        np.minimum(floor_min, distances[(idx + offset) % count], out=floor_min)
        np.minimum(floor_min, distances[(idx - offset) % count], out=floor_min)

    # 先用纯向量化的三条门槛筛掉绝大多数轮廓点（快）
    keep = distances >= RECOG_TIP_MIN_DISTANCE * float(radius)
    keep &= floor_max <= distances + 1e-9
    keep &= (distances - floor_min) >= RECOG_TIP_MIN_PROMINENCE * float(radius)
    candidate_indices = np.nonzero(keep)[0]
    if candidate_indices.size == 0:
        return [], distances

    # 剩下少数候选才算形态和厚度（贵的那部分）
    thickness = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    candidates = []
    for raw_index in candidate_indices:
        index = int(raw_index)
        ratio, angle = _tip_sharpness(contour, index, window)
        inner = points[index] + (np.asarray(palm_center) - points[index]) * \
            float(RECOG_TIP_THICKNESS_SAMPLE)
        thickness_ratio = _sample_plane(thickness, inner) / float(radius)
        if ratio < RECOG_TIP_MIN_TURN_RATIO and \
                thickness_ratio > RECOG_TIP_THICKNESS_MAX:
            continue
        if angle > RECOG_TIP_MAX_ANGLE_DEG:
            continue
        # 针状毛刺：夹角和厚度同时极端小 -> 不是指尖（真指尖是圆帽，两者都不会这么极端）
        if angle < RECOG_TIP_MIN_ANGLE_DEG and \
                thickness_ratio < RECOG_TIP_MIN_THICKNESS:
            continue
        candidates.append({
            "point": points[index].copy(),
            "distance": float(distances[index]),
            "turn_ratio": float(ratio),
            "angle_deg": float(angle),
            "thickness_ratio": float(thickness_ratio),
        })

    # 去重：距离太近的是同一个指尖，留离掌心更远的那个
    candidates.sort(key=lambda item: -item["distance"])
    merged = []
    for item in candidates:
        if any(float(np.linalg.norm(item["point"] - kept["point"]))
               < RECOG_TIP_DEDUP * float(radius) for kept in merged):
            continue
        merged.append(item)
        if len(merged) >= int(RECOG_TIP_MAX_COUNT):
            break

    # 过滤：指尖必须落在掌上方（局部系 y <= 阈值）。
    # 这一步同时把"前臂截断留下的两个直角"清掉 —— 它们在掌心下方。
    result = []
    for item in merged:
        local = _to_local_points(item["point"].reshape(1, 2),
                                 palm_center, up, radius)[0]
        if float(local[1]) > float(RECOG_TIP_PALM_SIDE_MAX):
            continue
        item["local"] = (float(local[0]), float(local[1]))
        result.append(item)

    # 按相对掌心的极角排序，显示时就是 1,2,3...
    result.sort(key=lambda item: float(np.arctan2(item["local"][1], item["local"][0])))
    return result, distances


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
        if span < 1e-6:
            continue
        depth = abs(float(np.cross(chord / span, valley - first_point)))
        if depth < RECOG_GAP_MIN_DEPTH * float(radius):
            continue
        local = _to_local_points(valley.reshape(1, 2), palm_center, up, radius)[0]
        local_first = _to_local_points(first_point.reshape(1, 2), palm_center, up, radius)[0]
        local_second = _to_local_points(second_point.reshape(1, 2), palm_center, up, radius)[0]
        # 凹谷处"两指尖"张开的夹角。局部坐标系只是旋转+缩放，角度和图像里一致。
        v1 = np.asarray(local_first, dtype=np.float64) - np.asarray(local, dtype=np.float64)
        v2 = np.asarray(local_second, dtype=np.float64) - np.asarray(local, dtype=np.float64)
        n1 = float(np.linalg.norm(v1))
        n2 = float(np.linalg.norm(v2))
        if n1 < 1e-6 or n2 < 1e-6:
            continue
        angle = float(np.degrees(np.arccos(
            float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)))))
        if angle > RECOG_GAP_MAX_ANGLE_DEG:
            continue
        gaps.append({
            "point": valley.copy(),
            "local": (float(local[0]), float(local[1])),
            "depth_ratio": float(depth) / float(radius),
            "angle_deg": float(angle),
            "pair": (tuple(np.round(first_point).astype(int)),
                     tuple(np.round(second_point).astype(int))),
        })
    # ⚠️ 这里**不做按位置去重**。
    #    每条指缝都是"某一对相邻指尖"之间的谷，用的是轮廓上互不重叠的弧段，
    #    结构上就不可能重复。之前按 0.35R 做位置去重，实测把中间两条相邻的
    #    指缝（只差 20.8 px）误判成同一条给合并了 —— 5 指只剩 3 条缝。
    #    要去重也应该是"每对相邻指尖最多一条"，而这天然成立。
    gaps.sort(key=lambda item: float(np.arctan2(item["local"][1], item["local"][0])))
    return gaps[:int(RECOG_GAP_MAX_COUNT)]


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
        # ⚠️ 两侧都要做上下界检查：指尖数通常比指缝数多一个（5 指 4 缝），
        #    直接写 gaps[position - 1] 会越界。
        left = gaps[position - 1]["point"] if 0 <= position - 1 < len(gaps) else None
        right = gaps[position]["point"] if 0 <= position < len(gaps) else None
        if left is not None and right is not None:
            base = (np.asarray(left, dtype=np.float64) +
                    np.asarray(right, dtype=np.float64)) / 2.0
        elif left is not None:
            base = np.asarray(left, dtype=np.float64)
        elif right is not None:
            base = np.asarray(right, dtype=np.float64)
        else:
            direction = np.asarray(tip["point"], dtype=np.float64) - np.asarray(palm_center)
            norm = float(np.linalg.norm(direction))
            direction = direction / norm if norm > 1e-6 else np.asarray(up, dtype=np.float64)
            base = np.asarray(palm_center, dtype=np.float64) + direction * float(radius)
        length = float(np.linalg.norm(np.asarray(tip["point"], dtype=np.float64) - base))
        fingers.append({
            "index": position + 1,
            "tip": np.asarray(tip["point"], dtype=np.float64).copy(),
            "base": base.copy(),
            "tip_local": (float(tip_local[0]), float(tip_local[1])),
            "length_ratio": length / float(radius),
            "angle_deg": float(np.degrees(np.arctan2(tip_local[1], tip_local[0]))),
        })
    return fingers


def _recognize(mask: np.ndarray, state) -> Optional[dict]:
    """跑一遍识别骨架，返回结果字典；失败返回 None。"""
    if mask is None or int(np.count_nonzero(mask)) == 0:
        return None

    # 掌心 C 与尺度 R：优先用模式 4 已经算好的；没有就自己算一遍（兜底）
    segment_stats = state.flags.get("segment_stats") or {}
    palm_center = segment_stats.get("palm_center")
    radius = segment_stats.get("palm_radius")
    if palm_center is None or not radius:
        distance = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        _, radius, _, location = cv2.minMaxLoc(distance)
        if float(radius) < float(RECON_MIN_RADIUS_PX):
            return None
        palm_center = (int(location[0]), int(location[1]))
    palm_center = np.array([float(palm_center[0]), float(palm_center[1])], dtype=np.float64)
    radius = float(radius)

    # 手腕 -> 掌轴 -> 前臂截断
    wrist = _estimate_wrist(mask, palm_center, radius)
    if wrist is None:
        up = np.array([0.0, -1.0], dtype=np.float64)
    else:
        direction = palm_center - wrist
        norm = float(np.linalg.norm(direction))
        up = direction / norm if norm > 1e-6 else np.array([0.0, -1.0], dtype=np.float64)
    hand = _truncate_forearm(mask, palm_center, up, radius) if RECOG_MASK_TRUNCATE else mask
    if int(np.count_nonzero(hand)) == 0:
        hand = mask

    contours, _ = cv2.findContours(hand, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if int(contour.shape[0]) < 8:
        return None

    tips, _profile = _detect_fingertips(contour, hand, palm_center, radius, up)
    gaps = _detect_gaps(contour, tips, palm_center, radius, up)
    fingers = _build_fingers(tips, gaps, palm_center, radius, up)

    # ---- 附加的"零成本"特征（都是现有数据的简单再加工）----

    # 1) 实心度 = 轮廓面积 / 凸包面积。
    #    手指之间的凹陷越多，实心度越低。
    #
    #    ★ 下面这组是**在本工程实测出来的**，不是从文献抄的
    #      （我一开始抄了文献的"张开 0.5~0.65"，实测对不上）：
    #        五指张开   0.69 ~ 0.70
    #        五指并拢   0.78        （用闭运算合成验证）
    #        握拳       0.88        （用掌核合成验证）
    #      合成姿态验证过它是**单调**的，方向正确。
    #      换环境/换人建议重新量一遍（见文件末尾的说明）。
    hull = cv2.convexHull(contour)
    contour_area = float(cv2.contourArea(contour))
    hull_area = float(cv2.contourArea(hull)) if hull is not None else 0.0
    solidity = float(np.clip(contour_area / hull_area, 0.0, 1.0)) if hull_area > 0 else 0.0

    # 2) 极角跨度 = 所有指尖极角所张的最小圆弧。
    #    五指并拢时各指尖极角接近（跨度小）；张开时跨度大（实测 86~92°）。
    #    用"环形跨度"算法，这样手转到大角度、角度跨过 ±180° 时也不会算错。
    #
    #    ⚠️ **手指数 <= 2 时这个值不可信，不要用**：
    #       实测"仿握拳"（只剩掌核）时检出的是轮廓凸点而不是真指尖，
    #       跨度反而算出 106.8°，比五指张开还大。
    #       它只在"确实有多根真手指"时才有意义。
    angle_span_deg = 0.0
    if len(tips) >= 2:
        tip_angles = np.array([float(np.arctan2(t["local"][1], t["local"][0]))
                               for t in tips], dtype=np.float64)
        ordered = np.sort(tip_angles)
        circular_gaps = np.diff(np.concatenate([ordered, [ordered[0] + 2.0 * np.pi]]))
        angle_span_deg = float(np.degrees(2.0 * np.pi - float(np.max(circular_gaps))))

    # 3) 掌心在 ROI 里的相对位置（0~1）。零成本，但能让"同一手势+不同位置"变成两条指令。
    if roi_enabled_now(state):
        rx0, ry0, rx1, ry1 = roi_rect_pixels((hand.shape[0], hand.shape[1]))
    else:
        rx0, ry0, rx1, ry1 = 0, 0, hand.shape[1], hand.shape[0]
    span_x = float(max(1, rx1 - rx0))
    span_y = float(max(1, ry1 - ry0))
    palm_rel = (float((palm_center[0] - rx0) / span_x),
                float((palm_center[1] - ry0) / span_y))

    # 4) 手指长度比的均值（单指的长度比在 fingers 里，这里顺手给个总览）
    mean_length = float(np.mean([f["length_ratio"] for f in fingers])) if fingers else 0.0

    # 5) 尺度 R 就是 radius，直接返回，不用另算。

    return {
        "palm_center": palm_center,
        "palm_radius": radius,
        "wrist": wrist,
        "up": up,
        "hand_mask": hand,
        "contour": contour,
        "hull": hull,
        "solidity": solidity,
        "angle_span_deg": angle_span_deg,
        "palm_rel": palm_rel,
        "mean_length_ratio": mean_length,
        "tips": tips,
        "gaps": gaps,
        "fingers": fingers,
        "finger_count": len(tips),
    }


def step_hand_recognize(frame: np.ndarray, state) -> np.ndarray:
    """算识别骨架，把结果挂到 state.flags["recognize"]，输出"只剩手"的掩膜。

    本插件**只算不画**，画由后面两个绘制插件负责 ——
    这样同一份结果既能画在掩膜上（模式 5）也能画在原图上（模式 6）。
    """
    single = frame[:, :, 0] if frame.ndim == 3 else frame
    result = _recognize(single.copy(), state)
    state.flags["recognize"] = result
    if result is None:
        return frame
    return cv2.cvtColor(result["hand_mask"], cv2.COLOR_GRAY2BGR)


# ===========================================================================
# 把识别结果画出来（供模式 5 / 6 共用）
# ===========================================================================
COLOR_PALM = (0, 165, 255)       # 掌心：橙
COLOR_WRIST = (0, 255, 255)      # 手腕：黄
COLOR_TIP = (0, 0, 255)          # 指尖：红
COLOR_BASE = (255, 128, 0)       # 指根：蓝
COLOR_GAP = (255, 0, 255)        # 指缝：品红
COLOR_LABEL = (0, 255, 0)        # 文字：绿
COLOR_OUTLINE = (0, 0, 0)        # 文字描边：黑


def _put_text_outlined(image, text, origin, scale=0.7, color=COLOR_LABEL, thickness=2):
    """带黑描边的文字，保证在黑白掩膜和彩色原图上都看得清。"""
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale,
                COLOR_OUTLINE, thickness + 3, cv2.LINE_AA)
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale,
                color, thickness, cv2.LINE_AA)


def _draw_recognition(base: np.ndarray, result: Optional[dict]) -> np.ndarray:
    """把识别结果画到给定底图上。"""
    height, width = base.shape[:2]
    if result is None:
        _put_text_outlined(base, "no hand", (20, 40), 0.9, (0, 0, 255), 2)
        return base

    palm_center = result["palm_center"]
    radius = result["palm_radius"]
    up = result["up"]
    wrist = result["wrist"]
    center_px = (int(round(palm_center[0])), int(round(palm_center[1])))

    # 凸包（细线）：实心度 = 轮廓面积 / 凸包面积，把它画出来你就能看到
    # "手指之间的凹陷"是怎么让实心度变小的。先画，后面的标注盖在上面。
    hull = result.get("hull")
    if hull is not None and len(hull) >= 3:
        cv2.drawContours(base, [hull], -1, (255, 255, 0), 1, cv2.LINE_AA)

    # 掌心 + 手掌尺度圆
    cv2.circle(base, center_px, int(round(radius)), COLOR_PALM, 2)
    cv2.drawMarker(base, center_px, COLOR_PALM, cv2.MARKER_CROSS, 40, 3)

    # 掌轴（指向手指方向）和它的垂直方向
    axis_length = radius * 1.4
    tip_up = (int(round(palm_center[0] + up[0] * axis_length)),
              int(round(palm_center[1] + up[1] * axis_length)))
    cv2.arrowedLine(base, center_px, tip_up, COLOR_WRIST, 2, tipLength=0.15)
    right = np.array([-up[1], up[0]])
    tip_right = (int(round(palm_center[0] + right[0] * radius * 0.7)),
                 int(round(palm_center[1] + right[1] * radius * 0.7)))
    cv2.arrowedLine(base, center_px, tip_right, COLOR_WRIST, 2, tipLength=0.15)

    # 手腕 + 截断线
    if wrist is not None:
        wrist_px = (int(round(wrist[0])), int(round(wrist[1])))
        cv2.circle(base, wrist_px, 6, COLOR_WRIST, -1)
        cv2.line(base, wrist_px, center_px, COLOR_WRIST, 1)
    # 截断线：位置按 RECOG_TRUNCATE_FACTOR（是真的切在那），
    # 长度用 RECOG_TRUNCATE_LINE_HALF（只是画出来给你看），并夹在画面内。
    truncate_point = palm_center - up * (float(RECOG_TRUNCATE_FACTOR) * radius)
    half = float(RECOG_TRUNCATE_LINE_HALF) * radius
    raw1 = truncate_point + right * half
    raw2 = truncate_point - right * half
    p1 = (int(np.clip(round(raw1[0]), 0, width - 1)),
          int(np.clip(round(raw1[1]), 0, height - 1)))
    p2 = (int(np.clip(round(raw2[0]), 0, width - 1)),
          int(np.clip(round(raw2[1]), 0, height - 1)))
    cv2.line(base, p1, p2, (128, 128, 128), 2)

    # 指缝
    for gap in result["gaps"]:
        point = (int(round(gap["point"][0])), int(round(gap["point"][1])))
        cv2.circle(base, point, 10, COLOR_GAP, 2)

    # 每根手指：指根 -> 指尖 一条线 + 两端点 + 编号和长度
    for finger in result["fingers"]:
        tip_px = (int(round(finger["tip"][0])), int(round(finger["tip"][1])))
        base_px = (int(round(finger["base"][0])), int(round(finger["base"][1])))
        cv2.line(base, base_px, tip_px, COLOR_TIP, 3)
        cv2.circle(base, base_px, 7, COLOR_BASE, -1)
        cv2.circle(base, tip_px, 9, COLOR_TIP, -1)
        cv2.circle(base, tip_px, 11, (255, 255, 255), 2)
        _put_text_outlined(base, str(finger["index"]),
                           (tip_px[0] + 14, tip_px[1] - 10), 0.8, COLOR_TIP, 2)
        mid = ((tip_px[0] + base_px[0]) // 2, (tip_px[1] + base_px[1]) // 2)
        _put_text_outlined(base, "%.2fR" % finger["length_ratio"],
                           (mid[0] + 12, mid[1]), 0.55, COLOR_LABEL, 1)

    # 左上角信息
    palm_rel = result.get("palm_rel", (0.0, 0.0))
    lines = [
        "fingers: %d   gaps: %d" % (result["finger_count"], len(result["gaps"])),
        "solidity: %.2f   span: %.0f deg" % (result.get("solidity", 0.0),
                                             result.get("angle_span_deg", 0.0)),
        "R = %.0f px   mean len: %.2fR" % (radius, result.get("mean_length_ratio", 0.0)),
        "palm rel: (%.2f, %.2f)" % palm_rel,
    ]
    if wrist is not None:
        lines.append("wrist (%d,%d)" % (int(wrist[0]), int(wrist[1])))
    else:
        lines.append("wrist: not found -> up = image up")
    for order, text in enumerate(lines):
        _put_text_outlined(base, text, (20, 40 + order * 32), 0.8, COLOR_LABEL, 2)
    return base


def step_recognize_on_mask(frame: np.ndarray, state) -> np.ndarray:
    """模式 5：把识别结果画在当前（掩膜）画面上。"""
    base = frame.copy()
    return _draw_recognition(base, state.flags.get("recognize"))


def step_recognize_on_original(frame: np.ndarray, state) -> np.ndarray:
    """模式 6：把识别结果画在**原图**上。

    原图从 state.flags["segment_source_bgr"] 取（颜色阈值那一层存下来的），
    **不能用 state.raw_frame** —— 镜像开着的时候 raw_frame 没被翻，
    坐标会和掩膜错位，画出来的标注就偏了。
    """
    source = state.flags.get("segment_source_bgr")
    if source is None:
        source = state.raw_frame if getattr(state, "raw_frame", None) is not None else frame
    base = source.copy()
    return _draw_recognition(base, state.flags.get("recognize"))


# ===========================================================================
# 插件 2：把真正生效的 ROI 边界画出来 + 触边检查
# ===========================================================================
def step_roi_boundary(frame: np.ndarray, state) -> np.ndarray:
    """在当前结果上画出真正生效的 ROI 边界，并做触边检查。

    因为本插件排在二值化之后，画上去的边界不会被二值化吃掉，
    所以「你看到的框」就是「真正生效的框」。用 r 键可以实时开关。
    """
    enabled = roi_enabled_now(state)
    state.flags["roi_enabled"] = enabled          # 回写，让 r 键有个确定的初始值

    if not enabled:
        state.flags["roi_touched"] = []
        return frame

    height, width = frame.shape[:2]
    x0, y0, x1, y1 = roi_rect_pixels((height, width))

    # 触边检查：只在"当前显示的确实是掩膜"时才做。
    # 显示模式 1 是原图，任何像素都非零，做触边检查毫无意义（会把三条边全点亮）。
    has_mask = bool(state.flags.get("view_mode_has_mask", True))
    touched = find_touched_edges(frame, (height, width), enabled) if has_mask else []
    state.flags["roi_touched"] = touched

    if not ROI_DRAW_BOUNDARY:
        return frame

    # 画边界。触边时用红色，正常用橙黄。
    #
    # ⚠️ 注意线宽是**以坐标为中心向两侧扩展**的：在 (x0,y0) 画厚度 3 的矩形，
    #    线会溢出到 x0-1 / y0-1，也就是 ROI 之外。测试时正是这一点让
    #    "ROI 外必须全黑"被破坏。所以这里把矩形**往里缩** thickness+1 像素画。
    color = COLOR_BOUNDARY_BAD if touched else COLOR_BOUNDARY
    thickness = 3
    inset = thickness + 1
    cv2.rectangle(frame,
                  (x0 + inset, y0 + inset),
                  (x1 - 1 - inset, y1 - 1 - inset),
                  color, thickness)

    # 文字也要全部画在 ROI 内部（之前把前景占比画在 y0-12，跑到 ROI 外面去了）
    text_y = y0 + inset + 30
    if touched:
        cv2.putText(frame, "TOUCHING ROI: %s" % ",".join(touched),
                    (x0 + inset + 6, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
        text_y += 32

    # 左上角标出「当前显示模式」和前景占比，一眼知道自己在看什么
    stats = state.flags.get("segment_stats") or {}
    lines = []
    view_mode = state.flags.get("view_mode")
    if view_mode:
        lines.append("view: %s" % view_mode)
    if has_mask:
        piece = []
        method = stats.get("method")
        if method:
            piece.append("gray" if method == "gray" else "color(Cr/Cb)")
        if stats.get("cleaned"):
            piece.append("cleaned")
        if stats.get("reconstructed"):
            piece.append("recon")
        if piece:
            lines.append("method: %s" % "+".join(piece))
        ratio = stats.get("foreground_ratio")
        if ratio is not None:
            lines.append("foreground %.1f%%" % (100.0 * ratio))
        # 重建那一层额外把掌心和基准颜色显示出来，方便判断容差合不合适
        if stats.get("reconstructed"):
            palm = stats.get("palm_center")
            if palm is not None:
                lines.append("palm C=(%d,%d) R=%.0f" % (palm[0], palm[1], stats.get("palm_radius", 0.0)))
            base = stats.get("color_base")
            if base is not None:
                lines.append("base Cr=%.0f Cb=%.0f Y=%.0f" % base)
    for text in lines:
        cv2.putText(frame, text, (x0 + inset + 6, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_TEXT, 2, cv2.LINE_AA)
        text_y += 28

    # ★兜底保证：**掩膜模式下**，ROI 之外一律擦回黑色。
    #   这样"ROI 外全黑"就是结构性成立的，以后再加标注也不会破坏它。
    #   ⚠️ 必须只对掩膜做：显示模式 1 是原图，把外面擦黑它就不是原图了。
    #   （单测时踩到过这个坑：原图模式下 ROI 外被擦黑，6.4M 像素不对。）
    if has_mask:
        cv2.bitwise_and(frame, build_roi_mask_bgr((height, width), enabled), dst=frame)
    return frame


def register(api) -> None:
    """main.py 会调用这个函数，把本模块的处理函数注册进插件链。

    注册顺序就是执行顺序。main.py 的 VIEW_MODES 会按 1/2/3 键
    启用不同的组合（靠 PluginRegistry.enable 实现）：
        1 -> 全都关掉，画面就是原图
        2 -> 只开 color_threshold，看到的是原始掩膜
        3 -> 再开 mask_cleanup，看到的是清理后的掩膜
    """
    # 灰度版按你的要求停用（注册着、代码保留，不进任何模式）
    api.add(PLUGIN_NAME, step_gray_threshold, enabled=GRAY_ENABLED_DEFAULT)
    # 颜色阈值（模式 2 / 3 / 4 用）
    api.add(COLOR_PLUGIN_NAME, step_color_threshold, enabled=COLOR_ENABLED_DEFAULT)
    # 清理：开运算 + 面积过滤 + 填洞（模式 3 / 4 用），必须排在颜色阈值之后
    api.add(CLEANUP_PLUGIN_NAME, step_mask_cleanup, enabled=CLEANUP_ENABLED_DEFAULT)
    # 掌心重建（模式 4 / 5 / 6 用），必须排在清理之后 —— 它要靠清理后的单连通域算掌心
    api.add(RECON_PLUGIN_NAME, step_palm_reconstruct, enabled=RECON_ENABLED_DEFAULT)
    # 识别骨架：截前臂 + 指尖 + 指缝（模式 5 / 6 用），只算不画
    api.add(RECOG_PLUGIN_NAME, step_hand_recognize, enabled=RECOG_ENABLED_DEFAULT)
    # 两种画法：画在掩膜上（模式 5）/ 画在原图上（模式 6）
    api.add(DRAW_MASK_PLUGIN_NAME, step_recognize_on_mask, enabled=DRAW_ENABLED_DEFAULT)
    api.add(DRAW_ORIG_PLUGIN_NAME, step_recognize_on_original, enabled=DRAW_ENABLED_DEFAULT)
    # 画边界 + 触边检查（排在最后，画在结果之上）
    api.add(ROI_PLUGIN_NAME, step_roi_boundary, enabled=True)
