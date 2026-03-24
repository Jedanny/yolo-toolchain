# 抽烟检测模型优化方案

## 问题诊断

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| 误报 | 相似物体混淆（手指、笔等） | Hard Negative Mining + 数据清洗 |
| 漏报 | 遮挡/小目标/特殊角度 | 数据增强 + 难例挖掘 |
| 场景泛化 | 训练数据与实际场景差异大 | 域适应 + 增量训练 |

---

## 优化流程

```
┌─────────────────────────────────────────────────────────────┐
│                    1. 问题诊断                                │
│  yolo-diagnose --model best.pt --data data.yaml           │
│  --output diagnostics/                                     │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                 2. 收集难例数据                             │
│  收集误报/漏报的真实场景图片                                  │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│              3. AI 自动标注 + 人工核验                        │
│  yolo-auto-annotate --images hard_samples/                  │
│      --output annotated/ --conf 0.5                         │
│                                                              │
│  yolo-verify --images annotated/images                      │
│      --labels annotated/labels                               │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│              4. 合并数据集继续训练                           │
│  原数据集 + 新难例 → 增量训练                                │
└─────────────────────────────────────────────────────────────┘
```

---

## 具体命令

### Step 1: 诊断分析
```bash
yolo-diagnose --model best.pt --data data/smoking.yaml --output runs/diagnostics
```

### Step 2: 收集难例图片
```bash
mkdir -p hard_samples/false_positives  # 误报图片
mkdir -p hard_samples/false_negatives   # 漏报图片
```

### Step 3: 自动标注难例
```bash
# 对漏报图片进行自动标注（高置信度阈值避免误标）
yolo-auto-annotate --images hard_samples/false_negatives \
    --output annotated/fn --dataset data/smoking.yaml --conf 0.6

# 对误报图片进行标注
yolo-auto-annotate --images hard_samples/false_positives \
    --output annotated/fp --classes cigarette --conf 0.3
```

### Step 3.5: 核验标注结果
```bash
# 交互式核验
yolo-verify --images annotated/fn --labels annotated/fn \
    --classes person cigarette --mode interactive

# 自动过滤低置信度标注
yolo-verify --images annotated/fn --labels annotated/fn \
    --mode auto --conf 0.5 --output annotated/fn_filtered
```

快捷键：`0-9`选择、`d`删除、`c`改类、`s`保存、`a`接受所有、`q`退出

### Step 4: 合并数据集训练
```bash
# 方式A：增量训练
yolo-incremental-train --model best.pt --data data/smoking_v2.yaml \
    --epochs 30 --project runs/smoking_v2 --name incremental

# 方式B：断点续训
yolo-train --model best.pt --data data/smoking_v2.yaml \
    --epochs 50 --resume --project runs/smoking_v2 --name fine_tune
```

### Step 5: 调整推理参数
```bash
# 提高置信度阈值减少误报
yolo-diagnose --model best.pt --data data/smoking.yaml --conf 0.5

# 降低置信度阈值减少漏报
yolo-diagnose --model best.pt --data data/smoking.yaml --conf 0.2
```

---

## 高级优化技巧

### 1. 数据增强增强版
```bash
yolo-augment --input data/smoking/images/train \
    --output data/smoking_aug --num_augment 5 --use_albumentations
```

### 2. 难例挖掘训练
```python
# 用当前模型预测难例，高loss样本重新标注训练
from ultralytics import YOLO
model = YOLO('best.pt')
# 预测并筛选高loss样本
```

### 3. 测试时增强 (TTA)
```python
from ultralytics import YOLO
model = YOLO('best.pt')
results = model.predict('test.jpg', augment=True)
```

---

## 判断优先优化方向

| 如果你的问题是 | 优先处理 |
|---------------|---------|
| 误报多（把不是烟的检测成烟） | 提高置信度阈值 + 数据清洗 |
| 漏报多（烟在但检测不到） | 降低阈值 + 小目标增强 |
| 特定场景差（光线/角度） | 收集该场景数据增量训练 |
| 小目标检测差 | 降低图像尺寸 + 添加小目标数据 |

---

## TODO 清单

- [ ] 运行诊断分析，生成误报/漏报报告
- [ ] 收集至少 50 张难例图片（误报/漏报各 25 张）
- [ ] 使用 AI 自动标注难例图片
- [ ] 人工核验标注结果（交互式 verify_annotator）
- [ ] 合并数据集，创建 `smoking_v2.yaml`
- [ ] 增量训练或断点续训
- [ ] 评估优化后的模型
- [ ] 调整推理阈值达到预期效果

---

## 后续规划

### 1. 第三方检测 API
- [ ] 引入第三方目标检测 API（如萤石 API、阿里云 API）作为自动标注备选方案
  - 支持基于预训练模型的目标检测 API
  - 可作为 Kimi-K2.5 多模态模型的替代方案，提升标注速度
  - 参考架构：添加 `src/tools/annotation_backends.py` 抽象接口

### 2. 生成式数据增强
- [ ] 基于图像生成模型（Stable Diffusion + LoRA + ControlNet）生成训练数据
  - 参考架构：`src/tools/synthetic_augmentor.py`
  - 流程：少量真实图片 → 训练 LoRA（主体/风格） → ControlNet（姿态/构图） → 批量生成
  - 可选方案：
    - 本地部署：Stable Diffusion WebUI / ComfyUI
    - 云服务：OpenAI DALL-E、Midjourney API、Stable Diffusion API
