# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

基于 Ultralytics YOLOv11 的目标检测工具链，支持数据准备、训练优化、评估诊断和模型导出。

## 语言 
项目文档和代码注释使用中文

## 常用命令

```bash
# 开发环境
make install-dev    # 安装开发依赖
make format        # black + isort
make lint          # flake8 (max-line-length=100)
make test          # pytest tests/

# 运行单个测试
uv run pytest tests/tools/test_pipeline.py::test_name -v

# 20 个 CLI 命令（安装后直接使用）
yolo-download          # 下载预训练模型
yolo-preprocess        # 图片预处理
yolo-convert           # VOC/COCO → YOLO
yolo-validate          # 数据集校验
yolo-label-qc          # 标签质量检查与修复
yolo-auto-annotate     # AI 自动标注
yolo-verify            # 标注核验
yolo-verify-inference  # 推理验证（标注检测框+类别+置信度）
yolo-augment           # 数据增强
yolo-anchors           # 生成最优 anchors
yolo-train             # 普通训练
yolo-freeze-train      # 冻结训练
yolo-incremental-train # 增量训练
yolo-tune              # 超参数调优
yolo-pr-analyze        # PR/F1 曲线分析（最优阈值）
yolo-prune             # 模型剪枝（压缩）
yolo-error-analyze     # 错误分析（FP/FN分类）
yolo-diagnose          # 诊断分析
yolo-export            # 模型导出
yolo-pipeline          # 全流程串联

# Pipeline 编排
yolo-pipeline --config configs/pipeline_train.yaml --dry-run  # 预览
yolo-pipeline --config configs/pipeline_train.yaml            # 执行
```

## 架构设计

### 目录结构

```
src/
├── tools/          # 数据处理工具（preprocess, convert, validate, auto-annotate 等）
├── train/          # 训练策略（trainer, freeze_trainer, incremental_trainer, pruner）
├── eval/           # 评估诊断（diagnostics, error_analyzer, pr_curve_analyzer）
└── export/         # 模型导出（exporter）
configs/            # Pipeline YAML 配置
tests/              # 单元测试
```

### 核心设计模式

1. **Dataclass 配置** - 每个工具模块有对应的 `*Config` dataclass，定义参数默认值和验证
2. **CLI 双入口** - 每个模块可通过 `yolo-<tool>` 直接调用，或 `uv run python -m src.<module>` 调用
3. **Pipeline 编排器** (`pipeline.py`) - YAML 配置串联多工具，通过 `ToolRegistry` 动态发现和执行工具
4. **API Key 配置** - `SILICONFLOW_API_KEY` 支持环境变量或 `.env` 文件

### Pipeline 设计

Pipeline 使用 `ToolRegistry` 单例模式注册工具函数，通过 `@register_tool` 装饰器自动注册。

**注意：Pipeline 参数名称与 CLI 参数名称可能不同**，编写 YAML 配置时必须使用工具函数定义的参数名：

| 工具 | CLI 参数 | Pipeline params 参数 |
|------|----------|---------------------|
| validate | `--dataset` | `dataset` |
| label-qc | `--dataset` | `dataset` (目录路径) |
| anchors | `--data` | `data` |
| pr-analyze | `--data` | `data` |
| error-analyze | `--data` | `data` |
| verify-inference | `--data` | `data` |
| train | `--data` | `data` |

**全局参数合并**：`global_params` 作为默认值，会与各 stage 的 `params` 合并（stage params 优先）。

```yaml
# 全局参数定义，所有 stage 共享
global_params:
  device: "mps"
  imgsz: 640
  batch: 8
  project: "./pipeline_output"
  conf: 0.25    # 置信度阈值
  iou: 0.7      # NMS IoU 阈值

stages:
  - name: "train"
    tool: "train"
    params:
      data: "./dataset.yaml"
      epochs: 100
      # project, device 等会从 global_params 继承
```

**两阶段微调策略**：小数据集推荐使用冻结 + 解冻两阶段训练。

```yaml
stages:
  - name: "冻结训练"
    tool: "train"
    params:
      model: "yolo11n.pt"
      freeze: [0,1,2,3,4,5,6,7,8,9]  # 冻结 backbone
      lr0: 0.001

  - name: "解冻微调"
    tool: "train"
    params:
      model: "./pipeline_output/stage1/weights/best.pt"
      freeze: []                       # 完全解冻
      lr0: 0.0001
      cos_lr: true
```

**数据增强参数**：训练节点支持 YOLO 原生数据增强。

```yaml
# 数据增强参数（适用于小数据集）
hsv_h: 0.015    # 色调
hsv_s: 0.7      # 饱和度
hsv_v: 0.4      # 亮度
degrees: 0.0    # 随机旋转
translate: 0.1  # 随机平移
scale: 0.5      # 随机缩放
shear: 0.0      # 随机剪切
perspective: 0.0 # 随机透视
flipud: 0.0    # 上下翻转
fliplr: 0.5     # 左右翻转
mosaic: 1.0    # Mosaic 增强
mixup: 0.0      # MixUp 增强
copy_paste: 0.0 # Copy-paste 增强
```

**训练优化参数**：

```yaml
# 类别不平衡处理
class_weights: [1.0, 2.0]  # 各类别权重
cls_loss_gain: 0.0          # 分类损失增益
box_loss_gain: 0.0           # 边框损失增益

# 正则化
label_smoothing: 0.1        # Label smoothing (防过拟合)

# EMA 指数移动平均
ema: 0.9999                  # 提升模型稳定性

# 梯度累积 (显存不足时模拟大batch)
accumulate: 4                # batch=4 时 effectively batch=16
```

### 关键约束

- **断点续训** - `last.pt` 与 `data.yaml` 需匹配
- **增量训练** - 新 YAML 需包含所有旧类别
- **域偏移问题** - 训练数据与实际场景差异大时，模型泛化能力差，需收集实际场景数据
- **模型导出** - TensorRT 需 NVIDIA GPU；CoreML 需 macOS

### 域偏移问题与数据收集策略

**典型域偏移场景**：
- 训练数据：线上图片、公开数据集
- 实际场景：室内摄像头、工业环境

**数据配比建议**：
| 数据来源 | 比例 | 作用 |
|----------|------|------|
| 实际场景数据 | 60-70% | 主体，保证场景匹配 |
| 线上人工标注 | 20-30% | 补充多样性 |
| 公开数据集 | 10% | 扩充数据量 |

**室内摄像头数据增强建议**：
```yaml
degrees: 5.0          # 小角度旋转
translate: 0.15        # 平移变化
scale: 0.3            # 缩小scale范围（目标大小相对一致）
hsv_h: 0.01          # 降低色调变化
hsv_s: 0.3           # 降低饱和度变化
hsv_v: 0.2           # 降低亮度变化（室内较暗）
perspective: 0.0001   # 轻微透视
```

**两阶段训练策略**：
1. 阶段1：用线上+公开数据 pretrain（学到基本特征）
2. 阶段2：用实际场景数据微调（适配实际场景）
