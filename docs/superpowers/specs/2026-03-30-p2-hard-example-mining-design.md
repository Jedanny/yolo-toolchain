# P2 难例挖掘自动重训工具设计文档

## 概述

自动识别模型预测困难的样本（FP/FN/小目标），通过动态过采样和增强策略，用难例数据集重训模型，提升模型在困难场景的表现。

**核心流程**：
```
模型推理 → FP/FN/小目标分类 → 难例评分 → 动态过采样+增强 → 新数据集 → 重训
```

---

## 一、功能定位

### 1.1 难例定义

| 类型 | 定义 | 筛选条件 |
|------|------|----------|
| FP (False Positive) | 误检 - 预测框与真值 IoU=0 或类别错误 | 预测框无匹配真值 |
| FN (False Negative) | 漏检 - 真值未被匹配 | 真值无匹配预测框 |
| 小目标 (Small Object) | 面积过小的目标 | bbox面积/图像面积 < 阈值 |

### 1.2 难例评分

```python
def compute_hardness_score(error_type: str, iou: float, confidence: float) -> float:
    """
    计算难例分数 (0-1, 越高越难)

    FP: 分数 = 1.0 - confidence  (置信度越高越"自信"的误检越难)
    FN: 分数 = 1.0 - iou         (IoU越低越难)
    小目标: 分数 = 0.8           (固定高分)
    """
```

### 1.3 动态过采样比例

| 难例分数范围 | 过采样倍数 | 说明 |
|-------------|-----------|------|
| 0.5 - 0.7 | 2x | 中等难度 |
| 0.7 - 0.9 | 3x | 高难度 |
| > 0.9 | 5x | 极难样本 |

---

## 二、CLI 接口

### 2.1 命令行用法

```bash
yolo-hard-example-mining \
  --model <model.pt> \
  --data <dataset.yaml> \
  --output <output_dir> \
  --strategy <oversample|weighted|filter> \
  [--iou-threshold <float>] \
  [--conf-threshold <float>] \
  [--small-area-threshold <float>] \
  [--max-oversample <int>] \
  [--device <str>]
```

### 2.2 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--model` | str | 必填 | 模型路径 |
| `--data` | str | 必填 | 数据集 YAML |
| `--output` | str | `./hard_examples` | 输出目录 |
| `--strategy` | str | `oversample` | 策略: oversample/weighted/filter |
| `--iou-threshold` | float | `0.5` | IoU 阈值判断匹配 |
| `--conf-threshold` | float | `0.25` | 置信度阈值 |
| `--small-area-threshold` | float | `0.01` | 小目标面积比例阈值 |
| `--max-oversample` | int | `5` | 最大过采样倍数 |
| `--device` | str | `cpu` | 推理设备 |

---

## 三、增强策略

### 3.1 增强类型

| 增强类型 | 参数 | 说明 |
|---------|------|------|
| 模糊增强 | blur_kernel=5, blur_sigma=2, blur_threshold=100 | Laplace方差 < blur_threshold 时应用 |
| 尺度变换 | scale_range=(0.8, 1.2) | 随机缩放 |
| 亮度调整 | brightness_range=(0.7, 1.3) | 随机亮度 |
| 噪声增强 | noise_var=10 | 添加高斯噪声 |

### 3.2 增强应用规则

| 难例类型 | 增强策略 | 变体数量 |
|---------|---------|---------|
| FP (误检) | 模糊+亮度 | 2 + score_offset 个 (score 0.7-0.9 → 3个, >0.9 → 4个) |
| FN (漏检) | 尺度+亮度 | 2 + score_offset 个 |
| 小目标 | 尺度(放大)+模糊 | 3 + score_offset 个 |

**变体数量计算**：
```python
def get_variant_count(hardness_score: float) -> int:
    base = 2 if hardness_score < 0.7 else 3 if hardness_score < 0.9 else 4
    score_offset = int((hardness_score - 0.5) * 2)  # 0.5-0.7 → 0, 0.7-0.9 → 0-1, >0.9 → 1-2
    return min(base + score_offset, max_variants)  # 最多 5 个
```

### 3.3 策略说明

| 策略 | 说明 |
|------|------|
| `oversample` | 复制难例图片到merged目录，保持原始数据集结构 |
| `weighted` | 不生成新图片，在训练时通过 `class_weights` 和 `class_weights_rebalance` 参数加权 |
| `filter` | 只输出难例列表（不生成新数据集），用于人工审核 |

### 3.4 策略对比

| 策略 | 优点 | 缺点 | 适用场景 |
|------|------|------|----------|
| `oversample` | 效果最好，数据量增加 | 磁盘占用增加 | 小数据集 (<5000) |
| `weighted` | 不占用额外空间 | 效果有限 | 大数据集 (>10000) |
| `filter` | 可人工审核难例 | 需人工介入 | 质量控制 |

---

## 四、输出产物

```
hard_examples/
├── fp/                           # 误检难例
│   ├── images/                   # 原图 + 增强
│   │   ├── img001.jpg
│   │   ├── img001_aug1.jpg
│   │   └── img001_aug2.jpg
│   └── labels/                   # 对应标签
├── fn/                           # 漏检难例
├── small/                        # 小目标难例
├── merged/                       # 合并后的训练集
│   ├── images/
│   └── labels/
├── mining_report.json            # 难例分析报告
└── retrain_config.yaml           # 重训配置
```

### 4.1 mining_report.json 格式

```json
{
  "model": "best.pt",
  "total_images": 1000,
  "hard_examples": {
    "fp": {"count": 45, "avg_score": 0.72},
    "fn": {"count": 32, "avg_score": 0.65},
    "small": {"count": 28, "avg_score": 0.80}
  },
  "oversampling": {
    "fp_images": 90,
    "fn_images": 64,
    "small_images": 112,
    "total_augmented": 266
  },
  "merged_dataset": {
    "original_images": 1000,
    "augmented_images": 266,
    "total_images": 1266
  }
}
```

### 4.2 retrain_config.yaml 格式

```yaml
# 难例挖掘重训配置
# 使用方法: yolo-train --config retrain_config.yaml

model: best.pt
data: dataset_merged.yaml  # 合并后的数据集
epochs: 50
imgsz: 640

# 难例挖掘参数记录
hard_example_mining:
  strategy: oversample
  original_images: 1000
  augmented_images: 266
  total_images: 1266
```

---

## 五、Pipeline 节点配置

### 5.1 节点配置

```yaml
# configs/nodes/hard-example-mining.yaml
name: "难例挖掘"
tool: "hard-example-mining"
params:
  model: null                    # Pipeline 运行时注入
  data: null                     # Pipeline 运行时注入
  output: null                   # 输出目录
  strategy: "oversample"        # oversample / weighted / filter
  iou_threshold: 0.5
  conf_threshold: 0.25
  small_area_threshold: 0.01
  max_oversample: 5
  device: "${device}"
```

### 5.2 Pipeline 使用示例

```yaml
stages:
  - name: "训练初始模型"
    tool: "train"
    params:
      data: ./configs/dataset.yaml
      epochs: 100

  - name: "难例挖掘"
    tool: "hard-example-mining"
    params:
      model: "${train.best_model}"
      data: ./configs/dataset.yaml
      output: "${project}/hard_examples"
      strategy: "oversample"

  - name: "难例重训"
    tool: "train"
    params:
      data: "${hard_example_mining.merged_dataset_yaml}"
      model: "${train.best_model}"
      epochs: 50
      name: "hard_example_retrain"
```

---

## 六、核心代码结构

```python
# src/tools/hard_example_miner.py

@dataclass
class HardExampleMiningConfig:
    """难例挖掘配置"""
    model: str
    data: str
    output: str = "./hard_examples"
    strategy: str = "oversample"  # oversample / weighted / filter
    iou_threshold: float = 0.5
    conf_threshold: float = 0.25
    small_area_threshold: float = 0.01
    max_oversample: int = 5
    device: str = "cpu"


class HardExampleMiner:
    """难例挖掘器"""

    def mine(self) -> Dict[str, Any]:
        # 1. 加载模型和数据集
        # 2. 推理验证集，收集 FP/FN/小目标
        # 3. 计算难例分数
        # 4. 应用增强策略
        # 5. 生成新数据集
        # 6. 保存报告和重训配置

    def _classify_errors(self, predictions, ground_truths) -> tuple:
        """分类错误类型"""

    def _compute_hardness_score(self, error_type, iou, confidence) -> float:
        """计算难例分数"""

    def _augment_image(self, img_path, error_type) -> List[str]:
        """对难例图片进行增强"""

    def _oversample_and_merge(self, hard_examples) -> str:
        """过采样并合并数据集"""
```

---

## 七、与其他工具的关系

### 7.1 依赖已有工具

| 工具 | 用途 |
|------|------|
| `error-analyze` | 错误分析和归因（可复用其 FP/FN 分类逻辑） |
| `train` | 重训模型 |
| `augment` | 图片增强（可复用其增强逻辑） |

### 7.2 复用设计

**error-analyze 的错误分类逻辑** 可直接复用：
- `ErrorAnalyzer.analyze_errors()` 返回 FP/FN 列表
- `ErrorCase` 数据结构包含 image_path, error_type, predicted_box, gt_box

**增强逻辑** 可复用 `auto_annotator.py` 中的图片处理函数，或参考 `tta_inference.py` 中的图像变换。

---

## 八、实现顺序

1. **hard_example_miner.py** — 核心难例挖掘逻辑
2. **Pipeline 节点配置** — nodes/hard-example-mining.yaml
3. **集成到 pipeline.py** — 注册 tool_hard_example_mining
4. **单元测试** — tests/tools/test_hard_example_miner.py

---

## 九、依赖

- `ultralytics>=8.0.0` — 已有
- `opencv-python>=4.5.0` — 已有
- `numpy>=1.21.0` — 已有
- `Pillow` — 图片增强用（YOLO 已依赖）

---

## 十、测试计划

1. **单元测试**：
   - `test_hardness_score.py` — 测试难例评分计算
   - `test_oversample_ratio.py` — 测试动态过采样比例
   - `test_augmentation.py` — 测试增强生成

2. **集成测试**：
   - 在木箱数据集上验证难例挖掘效果
   - 对比重训前后的 mAP 提升

3. **验证**：
   - 难例数量和分类是否合理
   - 增强图片是否有效
   - 重训后模型精度是否有提升
