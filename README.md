# YOLOv11 Enhanced Toolchain

基于 Ultralytics YOLOv11 的目标检测工具链，支持数据准备、训练优化、评估诊断和模型导出。

## 安装

```bash
make install        # 核心依赖
make install-dev   # 开发依赖 (pytest, black, isort, flake8)
make install-coreml  # CoreML 支持 (macOS)
```

## 快速开始

### 数据准备

```bash
# AI 自动标注（SiliconFlow Kimi-K2.5）
# 1. 复制 .env.example 为 .env，填入 API Key 和模型
cp .env.example .env

# 2. 运行标注（指定 YAML 配置自动加载类别）
uv run python -m src.tools.auto_annotator --images /path/to/images --output /path/to/output --dataset data.yaml

# 或手动指定类别，设置置信度阈值
uv run python -m src.tools.auto_annotator --images /path/to/images --output /path/to/output --classes person car dog --conf 0.3

# 单图片标注
uv run python -m src.tools.auto_annotator --images photo.jpg --output labels.txt --dataset data.yaml --conf 0.5 --single

# VOC → YOLO 格式转换
uv run python -m src.tools.dataset_builder --mode voc --input /path/to/VOC --output /path/to/yolo

# 数据增强
uv run python -m src.tools.augmentor --input /path/images --output /path/augmented --num_augment 5
```

### 训练模型

```bash
# 冻结骨干网络训练（默认冻 0-9 层）
uv run python -m src.train.freeze_trainer --data data.yaml --epochs 100

# 增量训练（添加新类别）
uv run python -m src.train.incremental_trainer --model best.pt --data new_data.yaml --epochs 50
```

### 评估诊断

```bash
uv run python -m src.eval.diagnostics --model best.pt --data data.yaml --output diagnostics/
```

### 模型导出

```bash
# ONNX
uv run python -m src.export.exporter --model best.pt --format onnx

# TensorRT FP16
uv run python -m src.export.exporter --model best.pt --format engine --half

# OpenVINO
uv run python -m src.export.exporter --model best.pt --format openvino
```

## 项目结构

```
yolo-toolchain/
├── src/
│   ├── tools/                  # 数据处理工具
│   │   ├── dataset_builder.py  # VOC/COCO → YOLO 格式转换
│   │   ├── augmentor.py        # Mosaic、Mixup、HSV 等增强
│   │   └── auto_annotator.py   # Kimi-K2.5 AI 自动标注
│   ├── train/                  # 训练策略
│   │   ├── freeze_trainer.py   # 冻结骨干网络微调
│   │   └── incremental_trainer.py  # 增量学习
│   ├── eval/                   # 评估诊断
│   │   └── diagnostics.py      # 误报/漏报分析
│   └── export/                 # 模型导出
│       └── exporter.py         # 多格式导出
├── configs/
│   ├── default_train.yaml      # 默认训练配置
│   └── augmentation.yaml       # 增强配置
└── Makefile
```

## Makefile 命令

```bash
make help                # 显示所有命令
make sync                # 同步依赖
make format              # 代码格式化
make lint                # 代码检查
make test                # 运行测试
make clean               # 清理缓存
make run-auto-annotate   # AI 自动标注帮助
make run-freeze-train    # 冻结训练帮助
make run-export          # 导出帮助
```

## 核心功能

| 模块 | 功能 |
|------|------|
| `tools/auto_annotator` | SiliconFlow Kimi-K2.5 AI 自动标注 |
| `tools/dataset_builder` | VOC/COCO → YOLO 格式转换，数据集统计 |
| `tools/augmentor` | Mosaic、Mixup、HSV、Albumentations 增强 |
| `train/freeze_trainer` | 冻结骨干网络微调（保留预训练知识） |
| `train/incremental_trainer` | 增量添加新类别 |
| `eval/diagnostics` | 误报/漏报分析，混淆矩阵，每类别 Precision/Recall |
| `export/exporter` | ONNX/TensorRT/OpenVINO/CoreML/TFLite 等导出 |

## License

基于 Ultralytics 开源协议。
