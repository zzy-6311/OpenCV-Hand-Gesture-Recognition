# 右手五指状态识别（OpenCV）

从手部图像中识别拇指、食指、中指、无名指、小指各自的伸开 / 收起状态。当前优先在背景变化较小、右手掌心朝向镜头的条件下完成识别，为后续机械手控制和 FPGA 移植提供软件参考。

仓库包含两个可独立运行的版本：**查看照片推荐使用 `end/` 优化版**；需要视频、实时摄像头或切换中间处理画面时，使用根目录交互版。优化版是独立子目录，不是继承根目录代码的 Python 类。

使用传统图像处理，不依赖深度学习、MediaPipe 或 OpenCV contrib。当前实现视觉识别与显示，尚未实现机械手通信或 FPGA 硬件逻辑。

## 快速开始：优化版

以下命令在**仓库根目录**执行。已验证环境为 Windows / Python 3.11.9；窗口显示需要带 GUI 支持的 `opencv-python`，不能只安装 headless 版本。

```powershell
# 只安装优化版需要的 NumPy 和 OpenCV
python -m pip install -r end/requirements.txt

# 使用本机默认照片：data/new_camera/shot_01.png
python -B end/main.py

# 或指定自己的照片、照片文件夹
python -B end/main.py "D:\photos\hand.png"
python -B end/main.py "D:\photos"
```

窗口中按 **N / 右方向键 / 空格** 查看下一张，按 **P / 左方向键** 查看上一张，按 **Q / Esc** 或关闭窗口退出。首尾循环，窗口标题显示文件名和当前序号。指定照片时，从该照片开始浏览同目录照片；指定文件夹时，从自然排序后的第一张开始。

**素材不随 Git 仓库提供。** 新克隆的仓库不能直接假定存在默认照片；请传入自己的路径，或准备下面说明的 `data/` 目录。

## 仓库结构与版本选择

```text
.
├── README.md               两个版本的统一使用说明
├── requirements.txt        交互版依赖，也覆盖优化版依赖
├── main.py                 交互版：图片 / 视频 / 摄像头、窗口和按键
├── plugins.py              交互版：处理、状态保持及绘制
├── finger_geometry.py      交互版：局部几何和手指身份分配
├── end/
│   ├── main.py             优化版：照片浏览、JSON 输出、可选保存
│   ├── recognizer.py       优化版：固定的无历史状态单帧识别流程
│   ├── finger_geometry.py  优化版自带的几何模块
│   └── requirements.txt    优化版最小依赖
└── data/                   本地素材，Git 忽略
```

| 对比项 | `end/` 优化版 | 根目录交互版 |
| --- | --- | --- |
| 启动入口 | `python -B end/main.py` | `python -B main.py` |
| 输入 | 单张照片、同目录照片浏览 | 预设图片组、视频、摄像头 |
| 识别执行 | 进入或切换照片时计算一次 | 随交互运行循环处理 |
| 处理方式 | 固定顺序，无跨帧状态保持 | 可切换显示阶段，含稳定性门控 |
| 结果 | `RecognitionResult` 和终端 JSON | `state.flags["recognize"]` |
| 运行代码依赖 | 仅 `end/` 内三个 Python 文件 | 根目录三个 Python 文件 |
| 第三方依赖 | NumPy、OpenCV | NumPy、OpenCV、Pillow |

`end/` 不导入根目录的 `plugins.py` 或开发项目中的辅助代码。两个版本各自保留 `finger_geometry.py`，以便独立复制运行，不在运行时互相引用。`end/` 内不放 README，使用说明统一维护在本文件。

依赖固定为 NumPy 2.4.6、opencv-python 5.0.0.93；交互版另外使用 Pillow 12.3.0 绘制中文。需要两个版本时，安装根目录依赖即可：

```powershell
python -m pip install -r requirements.txt
```

## 素材与路径

本机的 `data/` 是指向开发项目 `fpga-shoushi/data` 的 Windows 目录联接，未复制照片，也不进入 Git。换电脑或重新克隆后，需要自行准备素材或向优化版传入图片路径。

```text
data/
├── new_camera/
│   ├── shot_01.png ... shot_29.png
│   └── _真值.tsv
├── gesture_*.jpg
├── nonstd_01.png ... nonstd_10.png
├── _不标准_真值.tsv
├── shou1.png / shou2.png / badmat_01.png
└── gesture_video_1.mp4 / gesture_video_2.mp4
```

真值 TSV 仅用于开发评估，运行程序不会读取真值决定结果。缺少这些文件不影响优化版处理自定义照片，也不影响交互版使用实体摄像头。

优化版的默认照片路径相对于脚本位置解析，即仓库的 `data/new_camera/shot_01.png`；用户传入的相对路径则相对于当前工作目录。以下示例均以仓库根目录为工作目录。

## 优化版的完整用法

```powershell
# 无窗口：识别一张照片、输出 JSON，然后退出
python -B end/main.py "data/new_camera/shot_01.png" --no-show

# 浏览一个文件夹，按文件名自然排序
python -B end/main.py "data/new_camera"

# 启用旧手机照片对照使用的居中 ROI
python -B end/main.py "data/gesture_05.jpg" --roi

# 保存启动时第一张的标注图；输出目录必须已经存在
python -B end/main.py "data/new_camera/shot_01.png" --no-show --output "D:\photos\result.png"

# 查看参数
python -B end/main.py --help
```

| 参数 / 行为 | 说明 |
| --- | --- |
| 不传路径 | 从默认 `shot_01.png` 开始浏览它所在的文件夹 |
| 传图片路径 | 显示该图，并可浏览同目录其他照片 |
| 传文件夹路径 | 自然排序，如 `shot_2` 排在 `shot_10` 前 |
| `--no-show` | 只计算一次，不创建窗口；传文件夹时只处理第一张，不是批量导出 |
| `--show` | 显式打开浏览窗口，与默认行为一致 |
| `--roi` | 整次运行固定启用居中 ROI，不根据文件名自动选择参数 |
| `--output PATH` | 只保存启动时首张照片的标注图，翻页不覆盖此文件 |

支持 PNG、JPEG、BMP、TIFF、WebP 等列入程序后缀表的照片；中文路径通过 `imdecode` 读取。标准摄像头照片和非标准照片默认不启用 ROI；旧照片的历史对照使用 ROI，复测时需使用相同配置。

停留在当前照片时仅等待窗口事件，不重复识别。正常完成（包括未发现手）返回退出码 `0`；启动阶段输入、解码、处理或保存失败返回 `2`。翻页中单张照片失败会提示错误并允许继续浏览，正常退出仍返回 `0`。

程序不自动创建结果目录或日志，只有指定 `--output` 才保存图片。优化版入口禁用字节码缓存；示例中的 `-B` 同样用于避免产生 `__pycache__`。

### 输出顺序与无效结果

**接口顺序固定为 `TIMRP`**：`[拇指, 食指, 中指, 无名指, 小指]`。照片状态栏按 **`PRMIT`** 显示：从左到右为小指、无名指、中指、食指、拇指，对应右手掌心朝向镜头的视觉顺序。倒序仅作用于显示，不镜像图片、不改变输出数组。

| 输出 | 含义 |
| --- | --- |
| `finger_states = [False, True, True, False, False]` | 仅食指、中指伸开 |
| `finger_states = [False, False, False, False, False]` | 已识别手部，五指均收起 |
| `finger_states = None`，JSON 中为 `null` | 未取得有效手部识别，不能当作握拳 |
| `prediction = "IM"` | 伸开手指的字母组合 |
| `prediction = ""` | 已识别为五指均收起 |
| `prediction = None` | 未取得有效识别 |

终端 JSON 还包含 `file`、`status`、`finger_order`、`finger_text`、`method`、`fallback_reason`、`failure_reason`、`rotation_deg`、`processed_size` 和 `timings_ms`。耗时是本次 CPU 执行结果，不是 FPGA 性能。

### 从 Python 调用

下面代码在 `end/` 目录内执行，或放在与 `recognizer.py` 同目录的调用程序中：

```python
import cv2
from recognizer import recognize

image = cv2.imread("hand.png")
if image is None:
    raise ValueError("无法读取照片")

result = recognize(image, roi_enabled=False)
print(result.finger_states)
print(result.prediction)
print(result.to_dict())
```

输入为非空 `H×W×3`、`uint8`、BGR 图像。识别函数不修改输入、不使用前一张照片的结果，也不读取文件名或真值。

`result.image` 是缩放、旋转后的图，`result.mask` 是重建掩膜，`result.recognition` 保留轮廓、指尖和几何测量等信息；坐标属于**处理后的图像**。整幅掩膜无背景、无法得到有效掌心半径时，返回无效识别及失败原因，不将其伪装成握拳。

### 固定流程与优化内容

```text
读图 → 原规格缩放 → 自动转正 → YCrCb 分割 → 掩膜清理
     → 掌心重建 → 手腕和轮廓 → 指尖与指缝 → 五指状态
```

优化版删除插件注册、处理模式切换、摄像头 / 视频循环和跨帧稳定门控；照片浏览只负责选择输入，不向识别算法引入历史状态。空图、无手等条件检查，以及几何特征不可用时的旧算法回退仍保留。

主要优化保持原识别参数和图像处理效果：

- 中间掩膜保持单通道，省去显示用途的三通道扩展和多余复制。
- 分割与重建合并颜色阈值操作，减少中间数组和全图遍历。
- 场景色度直接按原网格取样，避免创建整幅 ROI 布尔数组。
- 自动转正去掉两次未参与角度计算的距离变换。
- 轮廓局部极值改用环形补边的窗口运算，保持原精度和判据。
- 掌心闭运算裁剪至带足够边距的前景范围，再贴回全图；后续距离变换保留原尺寸。
- 连通域不再计算未使用的统计量，绘制放在识别完成之后。

## 根目录交互版

需要视频或实时摄像头时使用此版本。以下命令在仓库根目录执行；其参数与优化版不同，不能混用。旧版的 `image` 第二参数用于选择预设照片组，不接受任意照片路径，也不提供 `--help`。

| 命令 | 输入 |
| --- | --- |
| `python -B main.py` | 默认 29 张标准摄像头照片 |
| `python -B main.py image camera` | `data/new_camera/shot_01.png` 至 `shot_29.png` |
| `python -B main.py image old` | `data/` 下的旧照片与补充图 |
| `python -B main.py video 1` | `data/gesture_video_1.mp4` |
| `python -B main.py video 2` | `data/gesture_video_2.mp4` |
| `python -B main.py video "D:\videos\hand.mp4"` | 自定义视频 |
| `python -B main.py camera` | 实时摄像头，默认索引 `0` |

摄像头索引在根目录 `main.py` 的 `CAMERA_INDEX` 配置。相对视频文件名从 `VIDEO_DATA_DIR` 读取，默认是 `data/`。图片不自动翻页，视频默认循环；标准照片默认关闭 ROI，旧照片和视频默认开启，可按 `r` 切换。

| 按键 | 交互版功能 |
| --- | --- |
| `q` / `Esc` | 退出 |
| 空格 | 暂停 / 继续；注意优化版的空格是下一张 |
| `1` / `2` / `3` | 原图 / 初始掩膜 / 清理后掩膜 |
| `4` / `5` / `6` | 掌心重建 / 掩膜上识别 / 原图上识别（默认） |
| `r` / `m` | ROI 限制 / 手动镜像开关 |
| `d` / `f` / `l` | 信息叠加 / 全屏 / 列出插件 |
| `n` / `p` / `R` | 图片模式：下一张 / 上一张 / 重新读取 |

交互版结果位于 `state.flags["recognize"]["finger_states"]`；`recognize` 不存在或为 `None` 表示无有效结果。五个方块按 `PRMIT` 显示：实心绿色表示伸开，空心灰色表示收起。过渡帧可能保持旧结果；`held=True` 表示保持，`settled` 表示稳定性门控状态。优化版没有这种跨帧保持行为。

交互版默认 `plugins.FINGER_IDENTITY_MODE = "geometry"`。局部指轴投影用于身份判定，并非检测到的真实关节点；拇指单独判断，其余四指按掌部位置分配。几何测量不可用时回退到旧方法。主要配置位于根目录 `main.py`、`plugins.py` 及 `finger_geometry.py` 的 `GeometryConfig`。

## 历史验证与已知效果

以下为 2026-10-10 优化时留下的开发集记录，**不是本次同步重新完成的全量测试**。静态评估关闭绘制和时序保持，五位状态全部正确才计为正确；旧照片使用 ROI。

| 素材 | 五位状态完全正确 | 原版单张中位耗时 | 优化版单张中位耗时 |
| --- | --- | --- | --- |
| 标准摄像头照片 | 28 / 29 | 187.25 ms | 121.95 ms |
| 旧照片，包含手背照片 | 25 / 28 | 312.47 ms | 214.51 ms |
| 非标准照片 | 6 / 10 | 193.50 ms | 116.91 ms |

另有 3 张无真值诊断图，不计准确率。70 张现有图像的优化前后对照覆盖处理图像、掩膜、轮廓、完整识别字典、几何浮点值及最终状态，结果一致；输入未被修改，逆序运行也一致。另完成了 30 个黑白 / 随机边界输入和 996 个形态学边界用例的对照，以及无效半径保护检查。

CPU 基准使用 Intel Core i9-13900HX、Python 3.11.9、NumPy 2.4.6、OpenCV 5.0.0，固定 OpenCV 单线程、关闭 OpenCL，预加载和预热后交错测量 3 轮。每版共 210 次调用：总时间 **55.198 秒 → 36.514 秒**，约 **1.512 倍速度、33.85% 耗时下降**。计时包含识别流程，排除读盘、JSON 输出和绘图，不保证其他机器或实际交互时有相同耗时。

这些是参与过调参的开发素材，尚不能代表独立新样本的泛化准确率。验证脚本和原始记录保留在开发项目的 `codex/`，不随本仓库提供，也不参与运行。静态回归不等于已验证实体摄像头、真实窗口交互或实时视频逐帧准确率。

已知问题仍包括：

- 标准照片 `shot_08` 将 `IMR` 识别为 `IMP`，无名指与小指仍有混淆。
- `gesture_33` 系列三张手背照片错误；当前主要适用右手掌心朝向镜头。
- `nonstd_03/06/08/09` 涉及遮挡、多余候选和几何位置偏移，仍识别错误。
- 严重并拢、重叠和掩膜损坏，不能靠末端身份分配单独修复；当前未加入内部边缘互补算法。
- 视频没有逐帧真值，目前不承诺实时准确率或固定帧率。

## FPGA 移植边界

两个版本都是 CPU 软件参考实现，`end/` 的固定顺序有利于梳理硬件步骤，**并不表示它已变成 FPGA 单像素流水线，或可以直接烧录到板子上**。

`assign_slots_q10` 最多处理 4 个非拇指候选，按候选数量枚举对应的有序槽位组合，最多 6 种；使用 Q10 坐标和 Q20 平方差代价，限定输入范围后的代价可放入有符号 32 位整数。这个末端分配规模较小，但产生输入特征的过程仍有插值、长度和归一化等运算。

上游任意角旋转、连通域、填洞、距离变换和轮廓追踪仍需要整帧数据、多遍扫描或浮点计算。`AUTO_ROTATE_DOWNSCALE=320` 仅用于方向估计；1280×720 图像并不会因此变成 320×180 的全流程处理，旋转还可能扩大画布。

后续应先建立定点参考模型和中间结果对照，再设计 DDR 工作区、顺序状态机和共享运算单元。缩图、固定 ROI、取消旋转或删除回退路径都需要重新验证效果。CPU 加速比例不能直接换算成 FPGA 资源或帧率；新算法尚未完成 RTL 综合、布局布线及实板验证，不能保证资源足够。

## 维护约定

- 根目录交互版与 `end/` 优化版分别维护入口和运行依赖，不通过开发目录中的脚本绕接。
- 使用方式和版本区别集中维护在根目录 README，`end/` 保持三个 Python 文件与一份依赖清单。
- 评估、计时、导出脚本、历史备份及生成图片留在开发项目的 `codex/`，不放入运行目录。
- 本地 `data/` 继续由 Git 忽略；发布代码不等于发布素材。
