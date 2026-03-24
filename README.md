# YOLOv11 Enhanced Toolchain

基于 Ultralytics YOLOv11 的目标检测工具链，支持数据准备、训练优化、评估诊断和模型导出。

## 安装

```bash
make install        # 核心依赖
make install-dev    # 开发依赖
```

## 脚本入口

安装后可直接使用（无需 `uv run python -m`）：

| 脚本 | 功能 |
|------|------|
| `yolo-download` | 下载预训练模型 |
| `yolo-preprocess` | 图片预处理 |
| `yolo-convert` | VOC/COCO → YOLO |
| `yolo-auto-annotate` | AI 自动标注 |
| `yolo-verify` | 标注核验 |
| `yolo-augment` | 数据增强 |
| `yolo-train` | 普通训练 |
| `yolo-freeze-train` | 冻结训练 |
| `yolo-incremental-train` | 增量训练 |
| `yolo-diagnose` | 诊断分析 |
| `yolo-export` | 模型导出 |

或使用 `uv run python -m src.<模块>` 调用。

---

## 0. 下载预训练模型

```bash
yolo-download --list                    # 列出可用模型
yolo-download --model yolo11n           # 下载模型
```

**可用模型**：YOLO11/YOLOv8/YOLOv9/YOLOv10 (n/s/m/l/x)

---

## 1. 数据准备

### 1.1 图片预处理

```bash
yolo-preprocess --input /path/images --resize 640 480 --enhance --denoise
```

参数：`--input/--output/--resize/--enhance/--brightness/--contrast/--denoise/--grayscale/--format/--quality`

### 1.2 格式转换

```bash
yolo-convert --mode voc --input /path/VOC --output /path/yolo
yolo-convert --mode coco --input /path/coco --output /path/yolo
```

### 1.3 AI 自动标注

```bash
# 配置 API Key
cp .env.example .env && vim .env

yolo-auto-annotate --images /path/images --output /path/output --dataset data.yaml --conf 0.3
yolo-auto-annotate --images /path/images --output /path/output --classes person cigarette
yolo-auto-annotate --images photo.jpg --output labels.txt --single
yolo-auto-annotate --images /path/images --output /path/output --labelimg  # 完成后预览
```

参数：`--images/--output/--dataset/--classes/--conf/--single/--workers/--seed/--labelimg`

默认：`train:val:test = 7:2:1`，`seed=42`，`workers=10`

输出：`images/train/`、`labels/train/`、`dataset.yaml`、`labels/train/classes.txt`

### 1.4 标注核验

```bash
yolo-verify --images /path/images --labels /path/labels --classes person cigarette
yolo-verify --images /path/images --labels /path/labels --mode auto --conf 0.6
```

快捷键：`0-9`选择、`d`删除、`c`改类、`s`保存、`a`接受所有、`q`退出

### 1.5 数据增强

```bash
yolo-augment --input /path/images --output /path/augmented --num_augment 5
```

---

## 2. 训练模型

### 2.1 普通训练

```bash
yolo-train --data data.yaml --epochs 100
yolo-train --data data.yaml --epochs 100 --resume        # 断点续训
```

### 2.2 冻结训练

```bash
yolo-freeze-train --data data.yaml --epochs 100          # 冻结前10层
```

### 2.3 增量训练

```bash
yolo-incremental-train --model best.pt --data new_data.yaml --epochs 50
```

通用参数：`--data/--model/--epochs/--batch/--imgsz/--device/--config/--log-level`

---

## 3. 评估诊断

```bash
yolo-diagnose --model best.pt --data data.yaml --output diagnostics/
```

输出：`diagnostics_report.json`、`confusion_matrix.png`、`class_performance.png`

---

## 4. 模型导出

```bash
yolo-export --model best.pt --format onnx
yolo-export --model best.pt --format engine --half        # TensorRT FP16
yolo-export --model best.pt --format openvino
yolo-export --model best.pt --format coreml              # macOS
```

格式：onnx, torchscript, engine, openvino, coreml, tflite, ncnn, mnn

---

## 快速开始（抽烟检测）

```bash
# 1. 下载模型
yolo-download --model yolo11n --output models/

# 2. 自动标注
cp .env.example .env && vim .env
yolo-auto-annotate --images data/smoking/images --output data/smoking --dataset data/smoking.yaml

# 3. 人工核验
yolo-verify --images data/smoking/images --labels data/smoking/labels --classes person cigarette

# 4. 训练
yolo-freeze-train --model models/yolo11n.pt --data data/smoking.yaml --epochs 100

# 5. 导出
yolo-export --model runs/train/freeze_train/weights/best.pt --format onnx
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
| 模型导出 | TensorRT 需 NVIDIA GPU；CoreML 需 macOS |

---

## License

基于 Ultralytics 开源协议。
