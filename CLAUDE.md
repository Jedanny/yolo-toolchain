# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

基于 Ultralytics YOLOv11 的目标检测工具链，支持数据准备、训练优化、评估诊断和模型导出。

## 常用命令

```bash
# 安装依赖
make install              # 核心依赖
make install-dev         # 开发依赖 (pytest, black, isort, flake8)
make sync                # 同步依赖

# 代码质量
make format              # 代码格式化 (black + isort)
make lint                # 代码检查 (flake8)
make test                # 运行测试

# 模块运行方式
uv run python -m src.train.freeze_trainer --help
uv run python -m src.train.incremental_trainer --help
uv run python -m src.tools.augmentor --help
uv run python -m src.export.exporter --help
uv run python -m src.eval.diagnostics --help
```

## 架构设计

### 模块结构

```
src/
├── tools/          # 数据处理工具
│   ├── dataset_builder.py   # VOC/COCO→YOLO 格式转换，数据集分析
│   ├── augmentor.py         # Mosaic、Mixup、HSV 等数据增强
│   └── auto_annotator.py    # SiliconFlow Kimi-K2.5 AI 自动标注
├── train/          # 训练策略
│   ├── freeze_trainer.py    # 冻结骨干网络微调（默认冻0-9层）
│   └── incremental_trainer.py  # 增量学习添加新类别
├── eval/           # 评估诊断
│   └── diagnostics.py       # 误报/漏报分析，混淆矩阵，每类别 Precision/Recall
└── export/         # 模型导出
    └── exporter.py          # ONNX/TensorRT/OpenVINO/CoreML 等多格式导出
```

### 配置模式

使用 `@dataclass` 定义配置类（如 `FreezeTrainConfig`、`ExportConfig`）：
- `to_dict()` 方法转换为 Ultralytics 训练参数
- 支持从 YAML 文件加载配置
- CLI 使用 `argparse`，可通过 `--config` 指定配置文件

### 核心设计模式

1. **Dataclass 配置** - 所有配置类继承 `@dataclass`，包含类型注解和默认值
2. **YOLO 封装** - 内部使用 `ultralytics.YOLO` 类，封装额外逻辑
3. **CLI 入口点** - 每个模块有 `main()` 函数，通过 `python -m src.<module>.<file>` 调用
4. **控制台脚本** - pyproject.toml 定义了快捷命令如 `yolo-freeze-train`、`yolo-export`

### 训练示例

```bash
# 冻结骨干网络训练
uv run python -m src.train.freeze_trainer --data data.yaml --epochs 100 --freeze 0 1 2 3 4 5 6 7 8 9

# 增量训练（添加新类别）
uv run python -m src.train.incremental_trainer --model best.pt --data new_data.yaml --epochs 50

# AI 自动标注（配置 .env 文件）
cp .env.example .env  # 填入 SILICONFLOW_API_KEY 和 SILICONFLOW_MODEL

# 使用 YAML 配置加载类别，设置置信度阈值
uv run python -m src.tools.auto_annotator --images /path/to/images --output /path/to/output --dataset data.yaml --conf 0.3

# 或手动指定类别
uv run python -m src.tools.auto_annotator --images /path/to/images --output /path/to/output --classes person car dog --conf 0.25

# 单图片标注
uv run python -m src.tools.auto_annotator --images photo.jpg --output labels/photo.txt --dataset data.yaml --conf 0.5 --single

# 数据格式转换
uv run python -m src.tools.dataset_builder --mode voc --input /path/to/voc --output /path/to/yolo

# 模型导出
uv run python -m src.export.exporter --model best.pt --format onnx
uv run python -m src.export.exporter --model best.pt --format engine --half

# 诊断分析
uv run python -m src.eval.diagnostics --model best.pt --data data.yaml --output diagnostics/
```
