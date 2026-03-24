# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

基于 Ultralytics YOLOv11 的目标检测工具链，支持数据准备、训练优化、评估诊断和模型导出。

## 常用命令

```bash
# 安装依赖
make install              # 核心依赖
make install-dev         # 开发依赖
make sync                # 同步依赖

# 代码质量
make format              # 代码格式化
make lint                # 代码检查
make test                # 运行测试

# 脚本入口（安装后直接使用）
yolo-download --model yolo11n     # 下载模型
yolo-preprocess --input /path     # 图片预处理
yolo-convert --mode voc           # 格式转换
yolo-auto-annotate --images ...   # AI 标注
yolo-verify --images ...          # 标注核验
yolo-augment --input ...          # 数据增强
yolo-train --data data.yaml       # 训练
yolo-freeze-train --data ...      # 冻结训练
yolo-incremental-train --model .. # 增量训练
yolo-diagnose --model ...         # 诊断分析
yolo-export --model ...           # 模型导出
```

## 架构设计

### 模块结构

```
src/
├── tools/          # 数据处理工具
│   ├── dataset_builder.py   # VOC/COCO→YOLO 格式转换
│   ├── augmentor.py         # Mosaic/Mixup/HSV 增强
│   ├── auto_annotator.py    # Kimi-K2.5 AI 自动标注（多线程）
│   ├── verify_annotator.py  # 标注核验
│   ├── preprocess.py         # 图片预处理
│   └── downloader.py         # Hugging Face 模型下载
├── train/          # 训练策略
│   ├── trainer.py          # 普通训练（断点续训）
│   ├── freeze_trainer.py    # 冻结骨干网络（默认冻0-9层）
│   └── incremental_trainer.py  # 增量学习
├── eval/           # 评估诊断
│   └── diagnostics.py       # 误报/漏报分析
└── export/         # 模型导出
    └── exporter.py          # ONNX/TensorRT/OpenVINO/CoreML 导出
```

### 配置模式

使用 `@dataclass` 定义配置类（如 `FreezeTrainConfig`、`ExportConfig`）：
- `to_dict()` 方法转换为 Ultralytics 训练参数
- 支持从 YAML 文件加载配置
- CLI 使用 `argparse`，可通过 `--config` 指定配置文件

### 核心设计模式

1. **Dataclass 配置** - 所有配置类继承 `@dataclass`，包含类型注解和默认值
2. **YOLO 封装** - 内部使用 `ultralytics.YOLO` 类，封装额外逻辑
3. **CLI 入口点** - 每个模块有 `main()` 函数，通过 `python -m src.<module>` 调用
4. **控制台脚本** - pyproject.toml 定义了快捷命令如 `yolo-train`、`yolo-export`

### 训练示例

```bash
# 普通训练
yolo-train --data data.yaml --epochs 100

# 断点续训
yolo-train --data data.yaml --epochs 100 --resume

# 冻结骨干网络训练
yolo-freeze-train --data data.yaml --epochs 100

# 增量训练
yolo-incremental-train --model best.pt --data new_data.yaml --epochs 50

# AI 自动标注
cp .env.example .env  # 填入 SILICONFLOW_API_KEY
yolo-auto-annotate --images /path/images --output /path/output --dataset data.yaml --conf 0.3

# 数据格式转换
yolo-convert --mode voc --input /path/voc --output /path/yolo

# 图片预处理
yolo-preprocess --input /path/images --resize 640 480 --enhance --denoise

# 模型导出
yolo-export --model best.pt --format onnx
yolo-export --model best.pt --format engine --half
```
