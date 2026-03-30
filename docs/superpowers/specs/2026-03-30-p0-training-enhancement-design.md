# P0 训练增强工具设计文档

## 概述

为 YOLO 工具链新增两个 P0 级别的训练增强工具：
1. **最佳模型选择工具** (`yolo-best-model`) — 自动对比 best.pt 和 last.pt，按指定指标选择最佳模型
2. **TTA 推理工具** (`yolo-tta`) — 测试时增强，提升推理精度

---

## 一、最佳模型选择工具 (best-model-selector)

### 1.1 功能定位

对比 `best.pt` 和 `last.pt`，在验证集上按指定指标评估，自动选择更好的模型。

### 1.2 CLI 接口

```bash
yolo-best-model --model <model_dir_or_path> --data <dataset.yaml> [--metric <metric>] [--output <path>]
```

**参数说明：**

| 参数 | 类型 | 默认值 | 说明 |
|-----|------|--------|------|
| `--model` | str | 必填 | 模型目录（含 weights/ 文件夹）或直接指定 .pt 文件 |
| `--data` | str | 必填 | 数据集 YAML 配置文件 |
| `--metric` | str | `fitness` | 选择指标：`mAP50`, `mAP50-95`, `recall`, `precision`, `fitness` |
| `--output` | str | None | 将结果路径写入指定文件（Pipeline params 用 `output`） |
| `--device` | str | `0` | 评估设备 |

**输出示例：**
```
=== 模型选择结果 ===
模型目录: ./runs/train/exp/weights

best.pt:
  - mAP50: 0.8523
  - mAP50-95: 0.6234
  - Recall: 0.7892
  - Precision: 0.8156

last.pt:
  - mAP50: 0.8491
  - mAP50-95: 0.6198
  - Recall: 0.7921
  - Precision: 0.8189

按 [fitness] 指标，最佳模型: best.pt
输出路径: ./runs/train/exp/weights/best.pt
```

### 1.3 Pipeline 节点配置

**节点名：** `best-model-select`

```yaml
# configs/nodes/best-model-select.yaml
name: "最佳模型选择"
tool: "best-model-select"
params:
  model: null                    # Pipeline 运行时注入
  data: null                     # Pipeline 运行时注入
  metric: "fitness"              # 可选：fitness/mAP50/mAP50-95/recall/precision
  output: null                   # 可选：输出路径（CLI 用 --output）
  device: "${device}"            # 从 global_params 继承
```

**注意：** CLI 参数为 `--output`，Pipeline params 参数为 `output`（统一命名）。

**Pipeline 使用示例：**
```yaml
stages:
  - name: "训练模型"
    tool: "train"
    params:
      data: ./configs/dataset.yaml
      epochs: 100
      # ...

  - name: "选择最佳模型"
    tool: "best-model-select"
    params:
      model: "${project}/train/weights"
      data: ./configs/dataset.yaml
      metric: "mAP50"
      output: "${project}/best_model.pt"
```

### 1.4 实现逻辑

```
1. 解析 model 参数，支持：
   - 目录路径：<project>/<name>/weights/
   - 直接 .pt 文件路径

2. 检查 best.pt 和 last.pt 是否存在

3. 使用 Ultralytics YOLO.val() 在验证集上评估两个模型

4. 按指定 metric 比较结果

5. 返回得分更高的模型路径，写入 output 文件（如果指定）
```

### 1.5 核心代码结构

```python
# src/tools/best_model_selector.py

@dataclass
class BestModelSelectorConfig:
    model: str
    data: str
    metric: str = "fitness"  # mAP50, mAP50-95, recall, precision, fitness
    output_path: Optional[str] = None
    device: str = "0"

class BestModelSelector:
    METRIC_MAP = {
        "mAP50": lambda r: r.box.map50,
        "mAP50-95": lambda r: r.box.map,
        "recall": lambda r: r.box.mr,
        "precision": lambda r: r.box.mp,
        "fitness": lambda r: r.box.fitness(),
    }

    def select(self) -> Dict[str, Any]:
        # 评估 best.pt
        # 评估 last.pt
        # 比较并返回结果
```

---

## 二、TTA 推理工具 (tta-inference)

### 2.1 功能定位

推理时对图片进行多尺度、翻转增强，使用 WBF（加权框融合）合并检测结果，提升检测精度。

### 2.2 TTA 策略

| 增强类型 | 实现 |
|---------|------|
| 多尺度 | 原始尺寸 × 0.8 × 1.2 |
| 水平翻转 | 原图 + 水平翻转 |
| 框融合 | WBF（加权框融合），IoU 阈值 0.5 |

**默认组合：** 3 尺度 × 2 翻转 = 6 张图片融合

### 2.3 CLI 接口

```bash
yolo-tta --model <model.pt> --images <path> [--output <dir>] [--scales <list>] [--flip] [--wbf-iou <float>]
```

**参数说明：**

| 参数 | 类型 | 默认值 | 说明 |
|-----|------|--------|------|
| `--model` | str | 必填 | 模型路径 |
| `--images` | str | 必填 | 图片目录或文件路径 |
| `--output` | str | `./tta_results` | 输出目录 |
| `--scales` | list | `[0.8, 1.0, 1.2]` | 尺度列表（CLI 空格分隔，如 `--scales 0.8 1.0 1.2`；默认 0.8/1.0/1.2 为经验值，平衡召回率与速度） |
| `--flip` | bool | `True` | 是否启用水平翻转（默认启用，与多尺度组合） |
| `--conf` | float | `0.25` | 置信度阈值 |
| `--iou` | float | `0.7` | NMS IoU 阈值 |
| `--wbf-iou` | float | `0.5` | WBF 融合 IoU 阈值 |
| `--device` | str | `cpu` | 推理设备 |
| `--save-txt` | bool | `False` | 保存检测结果为 TXT |
| `--save-conf` | bool | `True` | TXT 中包含置信度 |
| `--save-vis` | bool | `True` | 保存标注可视化图片 |

**Pipeline 用法：**
```yaml
stages:
  - name: "TTA 推理"
    tool: "tta-inference"
    params:
      model: "${project}/best_model.pt"
      images: ./test_images
      output: "${project}/tta_results"   # 注意：Pipeline params 用 output（CLI 用 --output）
      scales: [0.8, 1.0, 1.2]
      flip: true
      conf: 0.25
      wbf_iou: 0.5
```

**参数命名说明：** CLI 参数 `--output` 在 Pipeline 中映射为 `output`（统一命名约定）。

### 2.4 WBF 融合算法

**实现决策：** 自行实现 WBF（不引入 `ensemble_boxes` 依赖），参考 https://github.com/ZFTurbo/Weighted-Boxes-Fusion 的融合逻辑。

**核心原理：**
1. 将所有检测框按置信度降序排列
2. 遍历检测框，尝试融合到已有簇（IoU > 阈值）或创建新簇
3. 融合时对框坐标和置信度进行加权平均

```python
def wbf_fusion(boxes_list, scores_list, labels_list, iou_threshold=0.5):
    """
    boxes_list: List of detection boxes per image [x1, y1, x2, y2] (normalized 0-1)
    scores_list: List of confidence scores per image
    labels_list: List of class labels per image
    """
    # 参考: https://github.com/ZFTurbo/Weighted-Boxes-Fusion
    # 实现步骤：
    # 1. 将所有 boxes/scores/labels 按 image 分组展平
    # 2. 按 labels 分组处理
    # 3. 对每组：初始化空簇列表，遍历检测框
    #    - 计算与已有簇的 IoU
    #    - 若 IoU > iou_threshold，融合到该簇（加权平均坐标和置信度）
    #    - 否则创建新簇
    # 4. 返回融合后的 boxes, scores, labels
```

### 2.5 集成到 verify-inference

**实现方式：** `verify-inference` 工具内部调用 `TTAInference` 类实现 TTA 逻辑，而非调用独立 CLI。修改 `tool_verify_inference` 函数，添加 `--tta` 参数分支。

```bash
# 启用 TTA
yolo-verify-inference --model best.pt --images ./test --tta

# TTA 详细配置
yolo-verify-inference --model best.pt --images ./test --tta --tta-scales 0.8 1.0 1.2 --tta-flip --tta-wbf-iou 0.5
```

**参数映射：**
| 参数 | 说明 |
|-----|------|
| `--tta` | 启用 TTA 增强 |
| `--tta-scales` | TTA 尺度列表 |
| `--tta-flip` | 启用水平翻转 |
| `--tta-wbf-iou` | WBF 融合 IoU 阈值 |

### 2.6 输出结果

```
tta_results/
├── images/                 # 标注可视化图片
│   ├── img1.jpg
│   └── img2.jpg
├── labels/                 # 检测结果 TXT（可选）
│   ├── img1.txt
│   └── img2.txt
└── tta_report.json         # TTA 结果汇总
```

**tta_report.json 格式：**
```json
{
  "model": "best.pt",
  "images": 100,
  "total_detections": 342,
  "avg_detections_per_image": 3.42,
  "tta_settings": {
    "scales": [0.8, 1.0, 1.2],
    "flip": true,
    "wbf_iou": 0.5
  }
}
```

### 2.7 核心代码结构

```python
# src/tools/tta_inference.py

@dataclass
class TTAConfig:
    model: str
    images: str
    output_dir: str = "./tta_results"
    scales: List[float] = field(default_factory=lambda: [0.8, 1.0, 1.2])
    flip: bool = True
    conf: float = 0.25
    iou: float = 0.7
    wbf_iou: float = 0.5
    device: str = "cpu"

class TTAInference:
    def run(self) -> Dict[str, Any]:
        # 1. 加载模型
        # 2. 获取图片列表
        # 3. 对每张图片进行 TTA 推理
        # 4. 融合检测结果
        # 5. 保存结果
```

---

## 三、文件结构

```
src/tools/
├── best_model_selector.py    # 新增：最佳模型选择工具
│   ├── BestModelSelectorConfig
│   ├── BestModelSelector
│   └── main()
└── tta_inference.py         # 新增：TTA 推理工具
    ├── TTAConfig
    ├── TTAInference
    └── main()

configs/nodes/
├── best-model-select.yaml   # 新增：Pipeline 节点配置
└── tta-inference.yaml       # 新增：Pipeline 节点配置
```

**pyproject.toml 新增脚本：**
```toml
yolo-best-model = "src.tools.best_model_selector:main"
yolo-tta = "src.tools.tta_inference:main"
```

---

## 四、实现顺序

1. **best_model_selector.py** — 最佳模型选择（较简单，先实现）
2. **tta_inference.py** — TTA 推理
3. **Pipeline 节点配置** — nodes/best-model-select.yaml, nodes/tta-inference.yaml
4. **verify-inference TTA 集成** — 修改 pipeline.py 中的 tool_verify_inference

---

## 五、依赖

- `ultralytics>=8.0.0` — 已有
- `opencv-python>=4.5.0` — 已有
- `numpy>=1.21.0` — 已有
- WBF 融合：自行实现（不引入 `ensemble_boxes` 依赖，避免额外包）

---

## 六、测试计划

1. **单元测试：**
   - `test_best_model_selector.py` — 测试指标计算、模型对比逻辑
   - `test_tta_inference.py` — 测试 WBF 融合、尺度变换

2. **集成测试：**
   - Pipeline 节点测试
   - CLI 端到端测试

3. **验证：**
   - 在木箱检测数据集上验证 TTA 精度提升效果
