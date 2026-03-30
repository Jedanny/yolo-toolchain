# YOLOv11 Enhanced Toolchain

基于 Ultralytics YOLOv11 的目标检测工具链，支持数据准备、训练优化、评估诊断和模型导出。

## 安装

```bash
make install        # 核心依赖
make install-dev    # 开发依赖（含格式化、lint、测试）
```

## 20 个 CLI 命令

安装后直接使用（无需 `uv run python -m`）：

| 命令 | 功能 | 适用场景 |
|------|------|----------|
| `yolo-download` | 下载预训练模型 | 开始新项目前下载模型 |
| `yolo-preprocess` | 图片预处理 | 原始图片需要标准化时 |
| `yolo-convert` | VOC/COCO → YOLO | 已有 VOC/COCO 格式数据 |
| `yolo-validate` | 数据集校验 | 数据导入后验证完整性 |
| `yolo-label-qc` | 标签质量检查与修复 | 标注完成后质量核查 |
| `yolo-auto-annotate` | AI 自动标注 | 大量图片需要快速标注 |
| `yolo-verify` | 标注核验 | AI 标注后人工审核 |
| `yolo-verify-inference` | 推理验证 | 训练完成后测试集验证 |
| `yolo-augment` | 数据增强 | 小数据集扩充 |
| `yolo-anchors` | 生成最优 anchors | 训练前优化检测框 |
| `yolo-train` | 普通训练 | 通用训练场景 |
| `yolo-freeze-train` | 冻结训练 | 小数据集防过拟合 |
| `yolo-incremental-train` | 增量训练 | 已训练模型继续训练 |
| `yolo-tune` | 超参数调优 | 追求最优性能 |
| `yolo-pr-analyze` | PR/F1 曲线分析 | 找最优置信度阈值 |
| `yolo-prune` | 模型剪枝 | 模型压缩加速 |
| `yolo-error-analyze` | 错误分析 | 错误分类诊断 |
| `yolo-diagnose` | 诊断分析 | 全面评估模型 |
| `yolo-export` | 模型导出 | 部署前转换格式 |
| `yolo-pipeline` | 全流程串联 | 自动化完整流程 |

---

## 数据集配置格式

推荐在 `dataset.yaml` 中集中配置所有参数：

```yaml
# dataset.yaml 示例
path: /path/to/dataset          # 数据集根目录（绝对路径或相对路径）
train: images/train             # 训练集图片目录
val: images/test                # 验证集图片目录
test: images/test               # 测试集图片目录

nc: 2                           # 类别数量
names:
  0: person                     # 类别名称（支持中文）
  1: cigarette

# 类别详细描述（用于 AI 自动标注）
class_descriptions:
  0: 人物：检测人体，包括站姿和坐姿
  1: 香烟：检测手持香烟或烟雾，不包括烟盒
```

### class_descriptions 使用说明

在 `class_descriptions` 中定义每个类的详细描述，AI 自动标注时会自动读取：

```bash
# 自动从 dataset.yaml 读取 class_descriptions
yolo-auto-annotate --images ./images --output ./output --dataset ./dataset.yaml

# 也可通过命令行显式指定（优先级更高）
yolo-auto-annotate --images ./images --output ./output \
  --classes person cigarette \
  --class-desc person:人物：检测人体 \
  --class-desc cigarette:香烟：检测手持香烟
```

---

## 0. 下载预训练模型

```bash
# 列出可用模型
yolo-download --list

# 下载模型（支持 YOLO11/YOLOv8/YOLOv9/YOLOv10）
yolo-download --model yolo11n --output models/
yolo-download --model yolov8m --output models/
```

---

## 1. 数据准备

### 1.1 图片预处理

```bash
yolo-preprocess --input /path/images --resize 640 480 --enhance --denoise
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--input` | 输入图片目录 | 必填 |
| `--output` | 输出目录 | 同 input |
| `--resize` | 目标尺寸 (width height) | 不 resize |
| `--enhance` | 增强对比度 | 关闭 |
| `--brightness` | 亮度调整 (-1~1) | 0 |
| `--contrast` | 对比度调整 | 0 |
| `--denoise` | 去噪 | 关闭 |
| `--grayscale` | 转灰度 | 关闭 |
| `--format` | 输出格式 (jpg/png) | 原格式 |
| `--quality` | 输出质量 (1-100) | 95 |

### 1.2 格式转换

```bash
# VOC 格式转换
yolo-convert --mode voc --input /path/VOC --output /path/yolo

# COCO 格式转换
yolo-convert --mode coco --input /path/coco --output /path/yolo
```

### 1.3 数据集校验

检测常见数据问题：损坏图片、分辨率异常、bbox 越界、类别失衡、重复图片。

```bash
yolo-validate --input dataset/ --output report.json --fix
```

| 参数 | 说明 |
|------|------|
| `--input` | 数据集目录 |
| `--output` | 报告输出路径 (JSON) |
| `--fix` | 自动修复可修复的问题 |

### 1.4 标签质量检查

```bash
yolo-label-qc --data dataset/ --auto_fix
```

| 参数 | 说明 |
|------|------|
| `--data` | 数据集目录（包含 images/ 和 labels/） |
| `--auto_fix` | 自动修复重复框、过小 bbox |

### 1.5 AI 自动标注

需要先配置 API Key：

```bash
cp .env.example .env
# 编辑 .env 填入 SILICONFLOW_API_KEY
```

基本用法：

```bash
# 从 dataset.yaml 读取类别和描述
yolo-auto-annotate --images ./images --output ./output --dataset ./dataset.yaml

# 显式指定类别
yolo-auto-annotate --images ./images --output ./output --classes person cigarette

# 单图片模式
yolo-auto-annotate --images photo.jpg --output labels.txt --single

# 标注完成后用 labelImg 预览
yolo-auto-annotate --images ./images --output ./output --dataset ./dataset.yaml --labelimg

# 禁用可视化图片输出（节省磁盘空间）
yolo-auto-annotate --images ./images --output ./output --no-save-vis
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--images` | 图片目录或单图片路径 | 必填 |
| `--output` | 输出目录 | 必填 |
| `--dataset` | 数据集配置文件 | - |
| `--classes` | 类别列表 | 从 dataset 读取 |
| `--class-desc` | 类别详细描述 | 从 dataset 读取 |
| `--conf` | 置信度阈值 | 0.25 |
| `--single` | 单图片模式 | False |
| `--workers` | 并发线程数 | 10 |
| `--seed` | 随机种子 | 42 |
| `--save-vis` | 保存标注可视化图片 | True |
| `--no-save-vis` | 不保存可视化图片 | False |
| `--labelimg` | 完成后用 labelImg 预览 | False |

**数据集划分**：默认 `train:val:test = 7:2:1`

**输出结构**：
```
output/
├── images/
│   ├── train/...
│   ├── val/...
│   └── test/...
├── labels/
│   ├── train/... (YOLO 格式 .txt)
│   ├── val/...
│   └── test/...
├── vis/                    # 标注可视化图片（可省）
│   ├── train/...
│   ├── val/...
│   └── test/...
└── dataset.yaml
```

### 1.6 标注核验

AI 标注后人工审核修正：

```bash
yolo-verify --images ./images --labels ./labels --classes person cigarette
yolo-verify --images ./images --labels ./labels --mode auto --conf 0.6
```

| 参数 | 说明 |
|------|------|
| `--mode` | 模式：manual/auto |

**快捷键**：`0-9` 选类、`d` 删除、`c` 改类、`s` 保存、`a` 接受所有、`q` 退出

### 1.7 推理验证

用训练好的模型在测试集上推理，输出带标注的图片：

```bash
# 批量验证
yolo-verify-inference --model best.pt --images ./test_images --conf 0.25

# 单图片
yolo-verify-inference --model best.pt --images test.jpg --output ./results

# 保存检测结果为 TXT
yolo-verify-inference --model best.pt --images ./test --data dataset.yaml --save-txt
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model` | 模型路径 | 必填 |
| `--images` | 图片目录或路径 | 必填 |
| `--data` | 数据集配置（用于获取类别名） | - |
| `--classes` | 类别列表 | 从 model 读取 |
| `--conf` | 置信度阈值 | 0.25 |
| `--iou` | NMS IoU 阈值 | 0.7 |
| `--imgsz` | 图片尺寸 | 640 |
| `--device` | 设备 (0/mps/cpu) | 0 |
| `--output` | 输出目录 | ./inference_results |
| `--save-txt` | 保存 YOLO 格式结果 | False |
| `--save-conf` | TXT 中包含置信度 | False |
| `--line-width` | 框线宽度 | 2 |

### 1.8 数据增强

```bash
yolo-augment --input /path/images --output /path/augmented --num_augment 5
```

---

## 2. 模型优化

### 2.1 生成 Anchors

使用 K-means 聚类生成适合数据集的 anchor boxes：

```bash
yolo-anchors --data dataset/ --output anchors.yaml

# 生成后用已有模型验证效果
yolo-anchors --data dataset/ --validate --model best.pt
```

| 参数 | 说明 |
|------|------|
| `--data` | 数据集目录或 dataset.yaml |
| `--output` | 输出 YAML 路径 |
| `--validate` | 用模型验证 anchors 效果 |

### 2.2 超参数调优

使用遗传算法进化最优超参数：

```bash
# 快速调优（10 代）
yolo-tune --data dataset.yaml --model yolo11n.pt --generations 10

# 深度调优（20 代），每个子任务 5 epoch
yolo-tune --data dataset.yaml --model best.pt --generations 20 --epochs 5
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--data` | 数据集配置 | 必填 |
| `--model` | 基础模型 | yolo11n.pt |
| `--generations` | 进化代数 | 10 |
| `--epochs` | 每代训练 epoch | 10 |
| `--device` | 设备 | 0 |

---

## 3. 训练模型

### 训练策略选择

| 场景 | 推荐策略 | 说明 |
|------|----------|------|
| 小数据集 (< 5k 图) | 冻结 + 解冻两阶段 | 防止过拟合 |
| 中等数据集 (5k-50k) | 普通训练 + 早停 | 默认即可 |
| 大数据集 (> 50k) | 普通训练 | 数据足够多 |
| 已训练模型继续训练 | 增量训练 | 增量学习 |
| 追求最高精度 | 两阶段 + 超参调优 | 最完整流程 |

### 3.1 普通训练

```bash
yolo-train --data data.yaml --epochs 100 --device 0
yolo-train --data data.yaml --epochs 100 --resume   # 断点续训
```

### 3.2 冻结训练

冻结 backbone 训练，适合小数据集：

```bash
# 默认冻结前 10 层
yolo-freeze-train --data data.yaml --epochs 100

# 冻结更多层
yolo-freeze-train --data data.yaml --epochs 100 --freeze 0-15
```

### 3.3 增量训练

已训练模型继续训练新数据：

```bash
yolo-incremental-train --model best.pt --data new_data.yaml --epochs 50
```

**注意**：新数据集 YAML 必须包含所有旧类别。

### 训练通用参数

```bash
yolo-train --data data.yaml \
  --model yolo11n.pt \
  --epochs 100 \
  --batch 16 \
  --imgsz 640 \
  --device mps \
  --project runs/train \
  --name exp \
  --config custom_train.yaml
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--data` | 数据集 YAML | 必填 |
| `--model` | 模型路径 | yolo11n.pt |
| `--epochs` | 训练轮数 | 100 |
| `--batch` | 批次大小 | 16 |
| `--imgsz` | 图片尺寸 | 640 |
| `--device` | 设备 (0/1/mps/cpu) | 0 |
| `--project` | 输出项目目录 | runs/train |
| `--name` | 实验名称 | exp |
| `--config` | YAML 配置文件 | - |
| `--resume` | 断点续训 | False |

**训练优化参数**：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--optimizer` | 优化器 (SGD/Adam/AdamW) | auto |
| `--lr0` | 初始学习率 | 0.01 |
| `--lrf` | 最终学习率因子 | 0.01 |
| `--momentum` | SGD 动量 | 0.937 |
| `--weight_decay` | 权重衰减 | 0.0005 |
| `--warmup_epochs` | 预热轮数 | 3.0 |
| `--cos_lr` | 余弦学习率 | False |
| `--patience` | 早停耐心值 | 50 |

**数据增强参数**：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--hsv_h` | 色调增强 | 0.015 |
| `--hsv_s` | 饱和度增强 | 0.7 |
| `--hsv_v` | 亮度增强 | 0.4 |
| `--degrees` | 随机旋转 | 0.0 |
| `--translate` | 随机平移 | 0.1 |
| `--scale` | 随机缩放 | 0.5 |
| `--shear` | 随机剪切 | 0.0 |
| `--flipud` | 上下翻转概率 | 0.0 |
| `--fliplr` | 左右翻转概率 | 0.5 |
| `--mosaic` | Mosaic 概率 | 1.0 |
| `--mixup` | MixUp 概率 | 0.0 |
| `--copy_paste` | Copy-paste 概率 | 0.0 |
| `--label_smoothing` | 标签平滑 | 0.0 |

---

## 4. 评估诊断

### 4.1 PR/F1 曲线分析

分析不同置信度阈值下的 Precision-Recall 曲线，找最优阈值：

```bash
yolo-pr-analyze --model best.pt --data data.yaml --output-dir pr_analysis/
```

**输出**：
- `pr_analysis_results.json` - 包含最优阈值、F1 分数、AUC-PR
- `pr_curves.png` - PR 曲线和 F1 曲线图

### 4.2 错误分析

分析误报（FP）和漏报（FN）：

```bash
yolo-error-analyze --model best.pt --data data.yaml --images ./test_images --output error_analysis/
```

**输出**：
- `error_analysis_results.json` - 错误分类统计
- 可视化错误图片

### 4.3 模型剪枝

压缩模型加速推理：

```bash
# BN Gamma 剪枝（推荐）
yolo-prune --model best.pt --data data.yaml --method bn_gamma --amount 0.3 --fine-tune

# 剪枝后微调
yolo-prune --model best.pt --method l1 --amount 0.3 --fine-tune-epochs 10 --device mps
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--method` | 剪枝方法：l1/l2/bn_gamma | l1 |
| `--amount` | 剪枝比例 (0-1) | 0.3 |
| `--fine-tune` | 剪枝后微调 | False |
| `--fine-tune-epochs` | 微调轮数 | 10 |
| `--fine-tune-lr` | 微调学习率 | 0.0001 |

### 4.4 诊断分析

```bash
yolo-diagnose --model best.pt --data data.yaml --output diagnostics/
```

**输出**：
- `diagnostics_report.json`
- `confusion_matrix.png`
- `class_performance.png`

---

## 5. 模型导出

```bash
# ONNX（通用）
yolo-export --model best.pt --format onnx

# TensorRT FP16（NVIDIA GPU）
yolo-export --model best.pt --format engine --half

# OpenVINO（Intel CPU）
yolo-export --model best.pt --format openvino

# CoreML（macOS/iOS）
yolo-export --model best.pt --format coreml

# 指定输入尺寸
yolo-export --model best.pt --format onnx --imgsz 1280
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--format` | 导出格式 | onnx |
| `--half` | FP16 量化 | False |
| `--imgsz` | 输入尺寸 | 640 |

**支持格式**：onnx, torchscript, engine, openvino, coreml, tflite, ncnn, mnn

---

## 6. Pipeline 全流程串联

使用 YAML 配置串联多个工具，支持变量引用和全局参数。

### 6.1 基本用法

```bash
# 预览流程
yolo-pipeline --config pipeline_train.yaml --dry-run

# 执行流程
yolo-pipeline --config pipeline_train.yaml

# 保存执行报告
yolo-pipeline --config pipeline_train.yaml --report report.json
```

### 6.2 配置格式

```yaml
# pipeline_train.yaml
name: "完整训练流程"
description: "从数据校验到模型导出的完整流程"

# 全局参数，所有 stage 共享
global_params:
  device: mps
  imgsz: 640
  batch: 32
  project: ./pipeline_output
  conf: 0.25
  iou: 0.7

stages:
  # 数据校验
  - name: "1. 数据集校验"
    tool: validate
    params:
      dataset: ./configs/dataset.yaml
      model: yolo11n.pt

  # 标签质量检查
  - name: "2. 标签质量检查"
    tool: label-qc
    params:
      dataset: ./data/iia
      auto_fix: true

  # 生成 Anchors
  - name: "3. 生成 Anchors"
    tool: anchors
    params:
      data: ./configs/dataset.yaml
      output_dir: ${project}/anchors
      validate: true

  # 训练（可引用 ${project}）
  - name: "4. 训练模型"
    tool: train
    params:
      model: yolo11n.pt
      data: ./configs/dataset.yaml
      epochs: 100
      project: ${project}
      name: train
      exist_ok: true

  # 导出
  - name: "5. 导出模型"
    tool: export
    params:
      model: ${project}/train/weights/best.pt
      format: onnx
      project: ${project}
```

### 6.3 两阶段训练配置（推荐小数据集）

```yaml
stages:
  # Stage 1: 冻结 backbone
  - name: "4.1 冻结骨干网络训练"
    tool: train
    params:
      model: yolo11n.pt
      data: ./configs/dataset.yaml
      epochs: 30
      freeze: [0,1,2,3,4,5,6,7,8,9,10,11]  # 冻结 backbone
      lr0: 0.001
      optimizer: AdamW
      warmup_epochs: 3
      cos_lr: false
      project: ${project}
      name: stage1_freeze
      label_smoothing: 0.1
      hsv_h: 0.015
      hsv_s: 0.7
      hsv_v: 0.4
      degrees: 5.0
      translate: 0.1
      scale: 0.5
      shear: 2.0
      fliplr: 0.5
      mosaic: 1.0
      mixup: 0.1

  # Stage 2: 解冻微调
  - name: "4.2 解冻全量微调"
    tool: train
    params:
      model: ${project}/stage1_freeze/weights/best.pt
      data: ./configs/dataset.yaml
      epochs: 150
      freeze: []
      lr0: 0.0001
      optimizer: AdamW
      warmup_epochs: 3
      cos_lr: true
      project: ${project}
      name: stage2_finetune
      label_smoothing: 0.05
      hsv_h: 0.015
      hsv_s: 0.7
      hsv_v: 0.4
      degrees: 3.0
      translate: 0.1
      scale: 0.3
      shear: 1.0
      fliplr: 0.5
      mosaic: 1.0
      mixup: 0.15
```

### 6.4 训练参数说明

**Epochs 设置建议**：

| 数据集大小 | Stage 1 (冻结) | Stage 2 (解冻) | 总计 |
|------------|----------------|----------------|------|
| 小 (< 2k) | 20-30 | 150-200 | 180-230 |
| 中 (2k-10k) | 10-20 | 80-150 | 100-170 |
| 大 (> 10k) | 5-10 | 50-100 | 60-110 |

**数据增强策略**：

| 参数 | 小数据集 | 中等数据集 | 大数据集 |
|------|---------|-----------|---------|
| `degrees` | 5.0 | 3.0 | 0.0 |
| `scale` | 0.5 | 0.3 | 0.0 |
| `shear` | 2.0 | 1.0 | 0.0 |
| `mosaic` | 1.0 | 1.0 | 1.0 |
| `mixup` | 0.1-0.15 | 0.05-0.1 | 0.0 |
| `label_smoothing` | 0.1 | 0.05 | 0.0 |

### 6.5 预置 Pipeline 配置

```
configs/
├── pipeline_iia_best_practice.yaml  # IIA 最佳实践（180 epochs，两阶段）
├── pipeline_iia_quick.yaml         # IIA 快速验证（100 epochs）
├── pipeline_train.yaml             # 通用训练流程
└── ...
```

---

## 7. 快速开始（抽烟检测完整示例）

```bash
# 1. 下载模型
yolo-download --model yolo11n --output models/

# 2. 校验数据
yolo-validate --input data/smoking/ --fix

# 3. 标签质量检查
yolo-label-qc --data data/smoking/ --auto_fix

# 4. 生成 anchors
yolo-anchors --data data/smoking/ --output anchors.yaml

# 5. 配置数据集（编辑 dataset.yaml，添加 class_descriptions）
vim data/smoking/dataset.yaml

# 6. AI 自动标注
yolo-auto-annotate --images data/smoking/images --output data/smoking --dataset data/smoking/dataset.yaml

# 7. 人工核验
yolo-verify --images data/smoking/images --labels data/smoking/labels --classes person cigarette

# 8. 两阶段训练
yolo-pipeline --config configs/pipeline_iia_best_practice.yaml

# 9. PR 曲线分析（找最优阈值）
yolo-pr-analyze --model pipeline_output/iia_best_practice/stage2_finetune/weights/best.pt --data data/smoking/dataset.yaml

# 10. 推理验证
yolo-verify-inference --model pipeline_output/iia_best_practice/stage2_finetune/weights/best.pt --images data/smoking/images/test --conf 0.3

# 11. 导出模型
yolo-export --model pipeline_output/iia_best_practice/stage2_finetune/weights/best.pt --format onnx
```

---

## 8. 项目结构

```
src/
├── tools/                    # 数据处理工具
│   ├── anchor_generator.py   # Anchor 生成
│   ├── hyperparameter_tuner.py  # 超参数调优
│   ├── label_qc.py           # 标签质量检查
│   ├── pipeline.py            # 流程编排
│   ├── dataset_validator.py   # 数据集校验
│   ├── dataset_builder.py     # VOC/COCO 转换
│   ├── auto_annotator.py      # AI 自动标注
│   ├── verify_annotator.py    # 标注核验
│   ├── preprocess.py           # 预处理
│   ├── augmentor.py            # 数据增强
│   └── downloader.py            # 模型下载
├── train/                     # 训练策略
│   ├── trainer.py              # 普通训练
│   ├── freeze_trainer.py       # 冻结训练
│   ├── incremental_trainer.py  # 增量学习
│   └── pruner.py               # 模型剪枝
├── eval/                      # 评估诊断
│   ├── diagnostics.py          # 诊断分析
│   ├── error_analyzer.py       # 错误分析（FP/FN分类）
│   └── pr_curve_analyzer.py    # PR/F1 曲线分析
└── export/                    # 模型导出
    └── exporter.py             # ONNX/TensorRT 等

configs/                       # Pipeline YAML 配置
tests/                         # 单元测试
```

---

## 注意事项

| 阶段 | 注意 |
|------|------|
| 模型下载 | 需 Hugging Face 网络连接 |
| 自动标注 | API Key 保密；多模态响应较慢，默认 300s 超时 |
| 标注核验 | 交互模式需 GUI |
| 断点续训 | `last.pt` 与 `data.yaml` 需匹配 |
| 增量训练 | 新 YAML 需包含所有旧类别 |
| 超参数调优 | 耗时较长，建议先小规模验证 |
| 模型导出 | TensorRT 需 NVIDIA GPU；CoreML 需 macOS |
| 两阶段训练 | 小数据集推荐使用，防止过拟合 |

---

## License

基于 Ultralytics 开源协议。
