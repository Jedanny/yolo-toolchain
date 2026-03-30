# 难例挖掘自动重训 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现难例挖掘自动重训工具，自动识别 FP/FN/小目标难例，通过动态过采样和增强策略生成新数据集

**Architecture:** 独立工具模块 + Pipeline 集成，数据流：模型推理 → 难例分类 → 评分 → 过采样+增强 → 新数据集

**Tech Stack:** ultralytics YOLO, opencv-python, numpy, yaml

---

## Task 1: Config + 难例评分

**Files:**
- Create: `src/tools/hard_example_miner.py` (部分)
- Test: `tests/tools/test_hard_example_miner.py`

### 子任务 1.1: 配置文件

- [ ] **Step 1: 写测试**

```python
# tests/tools/test_hard_example_miner.py
import pytest
from src.tools.hard_example_miner import HardExampleMiningConfig

def test_config_defaults():
    config = HardExampleMiningConfig(
        model="best.pt",
        data="dataset.yaml"
    )
    assert config.output == "./hard_examples"
    assert config.strategy == "oversample"
    assert config.iou_threshold == 0.5
    assert config.conf_threshold == 0.25
    assert config.small_area_threshold == 0.01
    assert config.max_oversample == 5
```

Run: `uv run pytest tests/tools/test_hard_example_miner.py::test_config_defaults -v`
Expected: FAIL - ModuleNotFoundError

### 子任务 1.2: 写配置类

```python
# src/tools/hard_example_miner.py
from dataclasses import dataclass, field
from typing import List, Optional

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
```

Run: `uv run pytest tests/tools/test_hard_example_miner.py::test_config_defaults -v`
Expected: PASS

### 子任务 1.3: 写难例评分测试

```python
def test_hardness_score():
    from src.tools.hard_example_miner import compute_hardness_score

    # FP: 置信度越高越难
    assert compute_hardness_score("FP", 0.0, 0.9) == pytest.approx(0.1)  # 1.0 - 0.9
    assert compute_hardness_score("FP", 0.0, 0.5) == pytest.approx(0.5)

    # FN: IoU 越低越难
    assert compute_hardness_score("FN", 0.3, 0.0) == pytest.approx(0.7)  # 1.0 - 0.3
    assert compute_hardness_score("FN", 0.1, 0.0) == pytest.approx(0.9)

    # 小目标: 固定 0.8
    assert compute_hardness_score("small", 0.0, 0.0) == pytest.approx(0.8)
```

Run: `uv run pytest tests/tools/test_hard_example_miner.py::test_hardness_score -v`
Expected: FAIL - function not defined

### 子任务 1.4: 写难例评分函数

```python
# src/tools/hard_example_miner.py (续)

def compute_hardness_score(error_type: str, iou: float, confidence: float) -> float:
    """
    计算难例分数 (0-1, 越高越难)

    FP: 分数 = 1.0 - confidence  (置信度越高越"自信"的误检越难)
    FN: 分数 = 1.0 - iou         (IoU越低越难)
    小目标: 分数 = 0.8           (固定高分)
    """
    if error_type == "FP":
        return 1.0 - confidence
    elif error_type == "FN":
        return 1.0 - iou if iou < 1.0 else 0.0
    elif error_type == "small":
        return 0.8
    else:
        return 0.5
```

Run: `uv run pytest tests/tools/test_hard_example_miner.py::test_hardness_score -v`
Expected: PASS

### 子任务 1.5: 写过采样比例测试

```python
def test_oversample_ratio():
    from src.tools.hard_example_miner import get_oversample_ratio

    assert get_oversample_ratio(0.3) == 1  # < 0.5, 不过采样
    assert get_oversample_ratio(0.5) == 1
    assert get_oversample_ratio(0.6) == 2  # 0.5 - 0.7 → 2x
    assert get_oversample_ratio(0.75) == 3  # 0.7 - 0.9 → 3x
    assert get_oversample_ratio(0.95) == 5  # > 0.9 → 5x
    assert get_oversample_ratio(1.0) == 5  # 不超过 max_oversample
```

Run: `uv run pytest tests/tools/test_hard_example_miner.py::test_oversample_ratio -v`
Expected: FAIL - function not defined

### 子任务 1.6: 写过采样比例函数

```python
def get_oversample_ratio(hardness_score: float, max_oversample: int = 5) -> int:
    """根据难例分数返回过采样倍数"""
    if hardness_score < 0.5:
        return 1
    elif hardness_score < 0.7:
        return 2
    elif hardness_score < 0.9:
        return 3
    else:
        return min(5, max_oversample)
```

Run: `uv run pytest tests/tools/test_hard_example_miner.py::test_oversample_ratio -v`
Expected: PASS

### 子任务 1.7: 提交

```bash
git add src/tools/hard_example_miner.py tests/tools/test_hard_example_miner.py
git commit -m "feat: add HardExampleMiningConfig and hardness scoring"
```

---

## Task 2: 错误分类 (FP/FN/小目标)

**Files:**
- Modify: `src/tools/hard_example_miner.py`
- Test: `tests/tools/test_hard_example_miner.py`

### 子任务 2.1: 写 IoU 计算测试

```python
def test_compute_iou_xyxy():
    from src.tools.hard_example_miner import compute_iou_xyxy
    import numpy as np

    # 完全不重叠
    box1 = [0, 0, 10, 10]
    box2 = [20, 20, 30, 30]
    assert compute_iou_xyxy(box1, box2) == pytest.approx(0.0)

    # 完全重叠
    box1 = [0, 0, 10, 10]
    box2 = [0, 0, 10, 10]
    assert compute_iou_xyxy(box1, box2) == pytest.approx(1.0)

    # 部分重叠
    box1 = [0, 0, 10, 10]
    box2 = [5, 5, 15, 15]
    iou = compute_iou_xyxy(box1, box2)
    assert 0.1 < iou < 0.3  # 约 0.14
```

Run: `uv run pytest tests/tools/test_hard_example_miner.py::test_compute_iou_xyxy -v`
Expected: FAIL - function not defined

### 子任务 2.2: 写 IoU 计算函数

```python
def compute_iou_xyxy(box1: List[float], box2: List[float]) -> float:
    """计算两个框的 IoU [x1, y1, x2, y2]"""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
        return 0.0

    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = area1 + area2 - inter_area

    return inter_area / union_area if union_area > 0 else 0.0
```

Run: `uv run pytest tests/tools/test_hard_example_miner.py::test_compute_iou_xyxy -v`
Expected: PASS

### 子任务 2.3: 写错误分类测试

```python
def test_classify_errors():
    from src.tools.hard_example_miner import classify_errors

    # 模拟预测和真值
    predictions = [
        {"image": "img1.jpg", "boxes": [[0, 0, 10, 10, 0.9, 0]]},  # 正确检测
        {"image": "img2.jpg", "boxes": [[100, 100, 110, 110, 0.8, 0]]},  # FP
    ]
    ground_truths = {
        "img1.jpg": [[0, 0, 10, 10, 0]],  # 匹配
        "img2.jpg": [],  # 无真值
    }
    image_paths = {"img1.jpg": "/path/img1.jpg", "img2.jpg": "/path/img2.jpg"}

    fp_cases, fn_cases, correct = classify_errors(
        predictions, ground_truths, image_paths,
        iou_threshold=0.5, conf_threshold=0.25
    )

    assert len(fp_cases) == 1
    assert fp_cases[0]["image"] == "img2.jpg"
```

Run: `uv run pytest tests/tools/test_hard_example_miner.py::test_classify_errors -v`
Expected: FAIL - function not defined

### 子任务 2.4: 写错误分类函数

```python
@dataclass
class HardExample:
    """难例"""
    image_path: str
    error_type: str  # "FP", "FN", "small"
    box: List[float]  # xyxy
    score: float = 0.0  # 难例分数
    confidence: float = 0.0


def classify_errors(
    predictions: List[Dict],
    ground_truths: Dict[str, List[List[float]]],
    image_paths: Dict[str, str],
    iou_threshold: float = 0.5,
    conf_threshold: float = 0.25
) -> Tuple[List[HardExample], List[HardExample], List[HardExample]]:
    """
    分类错误类型

    Returns:
        (fp_cases, fn_cases, correct_cases)
    """
    fp_cases = []
    fn_cases = []
    correct_cases = []

    for pred in predictions:
        image_name = pred["image"]
        pred_boxes = pred.get("boxes", [])
        gt_boxes = ground_truths.get(image_name, [])

        matched_gt = set()

        # 按置信度排序
        pred_boxes_sorted = sorted(pred_boxes, key=lambda x: x[4], reverse=True)

        for pred_idx, pred_box in enumerate(pred_boxes_sorted):
            pred_xyxy = pred_box[:4]
            pred_conf = pred_box[4]
            pred_cls = int(pred_box[5]) if len(pred_box) > 5 else 0

            if pred_conf < conf_threshold:
                continue

            best_iou = 0
            best_gt_idx = -1

            for gt_idx, gt_box in enumerate(gt_boxes):
                if gt_idx in matched_gt:
                    continue
                gt_xyxy = gt_box[:4]
                gt_cls = int(gt_box[4]) if len(gt_box) > 4 else 0

                if pred_cls != gt_cls:
                    continue

                iou = compute_iou_xyxy(pred_xyxy, gt_xyxy)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            if best_iou >= iou_threshold:
                matched_gt.add(best_gt_idx)
                correct_cases.append(HardExample(
                    image_path=image_paths.get(image_name, image_name),
                    error_type="correct",
                    box=pred_xyxy,
                    score=0.0,
                    confidence=pred_conf
                ))
            else:
                # FP
                score = compute_hardness_score("FP", 0.0, pred_conf)
                fp_cases.append(HardExample(
                    image_path=image_paths.get(image_name, image_name),
                    error_type="FP",
                    box=pred_xyxy,
                    score=score,
                    confidence=pred_conf
                ))

        # 找出 FN
        for gt_idx, gt_box in enumerate(gt_boxes):
            if gt_idx in matched_gt:
                continue
            gt_xyxy = gt_box[:4]
            score = compute_hardness_score("FN", 0.0, 0.0)
            fn_cases.append(HardExample(
                image_path=image_paths.get(image_name, image_name),
                error_type="FN",
                box=gt_xyxy,
                score=score,
                confidence=0.0
            ))

    return fp_cases, fn_cases, correct_cases
```

Run: `uv run pytest tests/tools/test_hard_example_miner.py::test_classify_errors -v`
Expected: PASS

### 子任务 2.5: 提交

```bash
git add src/tools/hard_example_miner.py tests/tools/test_hard_example_miner.py
git commit -m "feat: add error classification (FP/FN)"
```

---

## Task 3: 增强函数

**Files:**
- Modify: `src/tools/hard_example_miner.py`
- Test: `tests/tools/test_hard_example_miner.py`

### 子任务 3.1: 写增强函数测试

```python
def test_augment_image():
    import tempfile
    import os
    from src.tools.hard_example_miner import augment_image
    import cv2
    import numpy as np

    # 创建临时图片
    with tempfile.TemporaryDirectory() as tmpdir:
        img_path = os.path.join(tmpdir, "test.jpg")
        # 先创建测试图片
        test_img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.imwrite(img_path, test_img)

        # 验证增强函数
        variants = augment_image(img_path, "FP", output_dir=tmpdir)
        assert len(variants) >= 1
        assert all(os.path.exists(v) for v in variants)
```

Run: `uv run pytest tests/tools/test_hard_example_miner.py::test_augment_image -v`
Expected: FAIL - function not defined

### 子任务 3.2: 写增强函数

```python
def augment_image(
    image_path: str,
    error_type: str,
    output_dir: str,
    variant_count: int = 2,
    blur_threshold: float = 100.0
) -> List[str]:
    """
    对难例图片进行增强

    Args:
        image_path: 原图路径
        error_type: 错误类型 "FP"/"FN"/"small"
        output_dir: 输出目录
        variant_count: 生成的变体数量上限
        blur_threshold: 模糊阈值

    Returns:
        增强后的图片路径列表（最多 variant_count 个）
    """
    import cv2
    import numpy as np
    from pathlib import Path

    img = cv2.imread(image_path)
    if img is None:
        return []

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    stem = Path(image_path).stem
    variants = []

    if error_type == "FP":
        # FP: 模糊 + 亮度 + 噪声
        all_variants = []
        all_variants.extend(_apply_blur(img, image_path, output_path, stem, blur_threshold))
        all_variants.extend(_apply_brightness(img, image_path, output_path, stem))
        all_variants.extend(_apply_noise(img, image_path, output_path, stem))
        variants = all_variants[:variant_count]  # 限制数量

    elif error_type == "FN":
        # FN: 尺度 + 亮度 + 噪声
        all_variants = []
        all_variants.extend(_apply_scale(img, image_path, output_path, stem))
        all_variants.extend(_apply_brightness(img, image_path, output_path, stem))
        all_variants.extend(_apply_noise(img, image_path, output_path, stem))
        variants = all_variants[:variant_count]

    elif error_type == "small":
        # 小目标: 放大 + 模糊 + 噪声
        all_variants = []
        all_variants.extend(_apply_scale_up(img, image_path, output_path, stem))
        all_variants.extend(_apply_blur(img, image_path, output_path, stem, blur_threshold))
        all_variants.extend(_apply_noise(img, image_path, output_path, stem))
        variants = all_variants[:variant_count]

    return variants


def _apply_blur(img, original_path, output_path, stem, threshold=100.0):
    """应用模糊增强"""
    import cv2
    import numpy as np

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    # 只有当图片模糊时才增强
    if laplacian_var < threshold:
        blurred = cv2.GaussianBlur(img, (5, 5), 2)
        output_file = output_path / f"{stem}_aug_blur.jpg"
        cv2.imwrite(str(output_file), blurred)
        return [str(output_file)]
    return []


def _apply_brightness(img, original_path, output_path, stem):
    """应用亮度增强"""
    import cv2
    import numpy as np

    variants = []
    for alpha in [0.7, 1.3]:  # 变暗/变亮
        brightened = cv2.convertScaleAbs(img, alpha=alpha, beta=0)
        output_file = output_path / f"{stem}_aug_bright_{int(alpha*10)}.jpg"
        cv2.imwrite(str(output_file), brightened)
        variants.append(str(output_file))
    return variants


def _apply_noise(img, original_path, output_path, stem, noise_var=10):
    """应用噪声增强"""
    import cv2
    import numpy as np

    noise = np.random.normal(0, noise_var, img.shape).astype(np.float32)
    noisy = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    output_file = output_path / f"{stem}_aug_noise.jpg"
    cv2.imwrite(str(output_file), noisy)
    return [str(output_file)]


def _apply_scale(img, original_path, output_path, stem):
    """应用尺度变换"""
    import cv2

    variants = []
    for scale in [0.8, 1.2]:
        h, w = img.shape[:2]
        new_h, new_w = int(h * scale), int(w * scale)
        scaled = cv2.resize(img, (new_w, new_h))
        output_file = output_path / f"{stem}_aug_scale_{int(scale*10)}.jpg"
        cv2.imwrite(str(output_file), scaled)
        variants.append(str(output_file))
    return variants


def _apply_scale_up(img, original_path, output_path, stem):
    """应用放大 (小目标优先放大)"""
    import cv2

    variants = []
    for scale in [1.2, 1.5]:
        h, w = img.shape[:2]
        new_h, new_w = int(h * scale), int(w * scale)
        scaled = cv2.resize(img, (new_w, new_h))
        output_file = output_path / f"{stem}_aug_scaleup_{int(scale*10)}.jpg"
        cv2.imwrite(str(output_file), scaled)
        variants.append(str(output_file))
    return variants
```

Run: `uv run pytest tests/tools/test_hard_example_miner.py::test_augment_image -v`
Expected: PASS

### 子任务 3.3: 提交

```bash
git add src/tools/hard_example_miner.py tests/tools/test_hard_example_miner.py
git commit -m "feat: add image augmentation functions"
```

---

## Task 4: 过采样合并

**Files:**
- Modify: `src/tools/hard_example_miner.py`
- Test: `tests/tools/test_hard_example_miner.py`

### 子任务 4.1: 写过采样合并测试

```python
def test_get_variant_count():
    from src.tools.hard_example_miner import get_variant_count

    assert get_variant_count(0.3) == 2  # 低分
    assert get_variant_count(0.6) == 2  # 0.5-0.7
    assert get_variant_count(0.8) == 3  # 0.7-0.9
    assert get_variant_count(0.95) == 4  # >0.9
```

Run: `uv run pytest tests/tools/test_hard_example_miner.py::test_get_variant_count -v`
Expected: FAIL - function not defined

### 子任务 4.2: 写变体数量函数

```python
def get_variant_count(hardness_score: float, max_variants: int = 5) -> int:
    """
    根据难例分数计算变体数量

    0.5以下: 0个新变体 (只用原图)
    0.5-0.7: 2个变体
    0.7-0.9: 3个变体
    >0.9: 4个变体
    """
    if hardness_score < 0.5:
        return 0
    elif hardness_score < 0.7:
        return 2
    elif hardness_score < 0.9:
        return 3
    else:
        return min(4, max_variants)
```

Run: `uv run pytest tests/tools/test_hard_example_miner.py::test_get_variant_count -v`
Expected: PASS

### 子任务 4.3: 提交

```bash
git add src/tools/hard_example_miner.py tests/tools/test_hard_example_miner.py
git commit -m "feat: add variant count calculation"
```

---

## Task 5: 完整 HardExampleMiner 类

**Files:**
- Modify: `src/tools/hard_example_miner.py`
- Test: `tests/tools/test_hard_example_miner.py`

### 子任务 5.1: 写 HardExampleMiner 类骨架测试

```python
def test_hard_example_miner_init():
    from src.tools.hard_example_miner import HardExampleMiner, HardExampleMiningConfig

    config = HardExampleMiningConfig(
        model="best.pt",
        data="dataset.yaml",
        output="./test_output"
    )
    miner = HardExampleMiner(config)
    assert miner.config == config
    assert miner.fp_cases == []
    assert miner.fn_cases == []
```

Run: `uv run pytest tests/tools/test_hard_example_miner.py::test_hard_example_miner_init -v`
Expected: PASS

### 子任务 5.2: 写 HardExampleMiner.run 方法

```python
# HardExampleMiner 类完整实现
class HardExampleMiner:
    """难例挖掘器"""

    def __init__(self, config: HardExampleMiningConfig):
        self.config = config
        self.fp_cases: List[HardExample] = []
        self.fn_cases: List[HardExample] = []
        self.small_cases: List[HardExample] = []
        self.model = None

    def mine(self) -> Dict[str, Any]:
        """执行难例挖掘"""
        # 1. 加载模型和数据集
        self._load_model()
        images_dir, labels_dir = self._load_dataset()

        # 2. 推理验证集，收集 FP/FN/小目标
        predictions, ground_truths, image_paths = self._collect_predictions_and_gt(
            images_dir, labels_dir
        )

        # 3. 分类错误
        fp_cases, fn_cases, _ = classify_errors(
            predictions, ground_truths, image_paths,
            iou_threshold=self.config.iou_threshold,
            conf_threshold=self.config.conf_threshold
        )
        self.fp_cases = fp_cases
        self.fn_cases = fn_cases

        # 4. 识别小目标（从预测和真值中）
        self._identify_small_objects(predictions, ground_truths, image_paths)

        # 5. 应用增强策略
        if self.config.strategy == "oversample":
            merged_yaml = self._oversample_and_merge()
        elif self.config.strategy == "weighted":
            merged_yaml = self._generate_weighted_config()
        else:  # filter
            merged_yaml = None

        # 6. 生成报告
        report = self._generate_report()

        return {
            "fp_count": len(self.fp_cases),
            "fn_count": len(self.fn_cases),
            "small_count": len(self.small_cases),
            "merged_dataset_yaml": merged_yaml,
            "report": report,
        }

    def _load_model(self):
        """加载模型"""
        from ultralytics import YOLO
        self.model = YOLO(self.config.model)

    def _load_dataset(self) -> Tuple[Path, Path]:
        """加载数据集路径"""
        import yaml
        with open(self.config.data, 'r') as f:
            data_config = yaml.safe_load(f)

        dataset_path = Path(data_config.get('path', Path(self.config.data).parent))
        val_path = dataset_path / data_config.get('val', 'images/val')
        test_path = dataset_path / data_config.get('test', 'images/test')

        if test_path.exists():
            images_dir = test_path
        elif val_path.exists():
            images_dir = val_path
        else:
            raise ValueError(f"Could not find validation or test images. Tried: {test_path} and {val_path}")

        labels_dir = images_dir.parent / 'labels' / images_dir.name
        if not labels_dir.exists():
            labels_dir = dataset_path / 'labels' / 'val'

        if not labels_dir.exists():
            raise ValueError(f"Could not find labels directory. Tried: {labels_dir}")

        return images_dir, labels_dir

    def _collect_predictions_and_gt(self, images_dir, labels_dir):
        """收集预测和真值"""
        predictions = []
        ground_truths = {}
        image_paths = {}

        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            image_files.extend(list(images_dir.glob(ext)))
            image_files.extend(list(images_dir.glob(ext.upper())))

        for img_path in image_files:
            image_name = img_path.name
            image_paths[image_name] = str(img_path)

            # 加载真值
            label_path = labels_dir / f"{img_path.stem}.txt"
            gt_boxes = []
            if label_path.exists():
                img = cv2.imread(str(img_path))
                if img is not None:
                    h, w = img.shape[:2]
                    with open(label_path, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                cls = int(parts[0])
                                cx, cy, bw, bh = map(float, parts[1:5])
                                x1 = (cx - bw/2) * w
                                y1 = (cy - bh/2) * h
                                x2 = (cx + bw/2) * w
                                y2 = (cy + bh/2) * h
                                gt_boxes.append([x1, y1, x2, y2, cls])
            ground_truths[image_name] = gt_boxes

            # 推理
            results = self.model.predict(
                source=str(img_path),
                conf=self.config.conf_threshold,
                verbose=False
            )

            pred_boxes = []
            if len(results) > 0 and results[0].boxes is not None:
                for box in results[0].boxes:
                    xyxy = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    pred_boxes.append([*xyxy, conf, cls])

            predictions.append({"image": image_name, "boxes": pred_boxes})

        return predictions, ground_truths, image_paths

    def _identify_small_objects(self, predictions, ground_truths, image_paths):
        """识别小目标（从预测和真值中）"""
        self.small_cases = []
        seen_images = set()

        def process_boxes(boxes, img_path, h, w, img_area, source):
            """处理一组框，识别小目标"""
            for box in boxes:
                x1, y1, x2, y2 = box[:4]
                box_area = (x2 - x1) * (y2 - y1)
                area_ratio = box_area / img_area if img_area > 0 else 0

                if area_ratio < self.config.small_area_threshold:
                    confidence = box[4] if len(box) > 4 else 0.0
                    score = compute_hardness_score("small", 0.0, confidence)
                    self.small_cases.append(HardExample(
                        image_path=img_path,
                        error_type="small",
                        box=[x1, y1, x2, y2],
                        score=score,
                        confidence=confidence
                    ))

        for image_name in set(list(ground_truths.keys()) + [p["image"] for p in predictions]):
            if image_name in seen_images:
                continue
            seen_images.add(image_name)

            img_path = image_paths.get(image_name)
            if not img_path:
                continue
            img = cv2.imread(img_path)
            if img is None:
                continue
            h, w = img.shape[:2]
            img_area = h * w

            # 处理真值
            if image_name in ground_truths:
                process_boxes(ground_truths[image_name], img_path, h, w, img_area, "gt")

            # 处理预测
            for pred in predictions:
                if pred["image"] == image_name:
                    process_boxes(pred.get("boxes", []), img_path, h, w, img_area, "pred")

    def _oversample_and_merge(self) -> str:
        """过采样并合并数据集"""
        import shutil

        output_dir = Path(self.config.output)
        merged_images_dir = output_dir / "merged" / "images"
        merged_labels_dir = output_dir / "merged" / "labels"

        # 创建目录
        for d in [merged_images_dir, merged_labels_dir]:
            d.mkdir(parents=True, exist_ok=True)

        all_hard_examples = self.fp_cases + self.fn_cases + self.small_cases
        variant_stats = {"fp": 0, "fn": 0, "small": 0}

        for case in all_hard_examples:
            src_img = case.image_path
            src_stem = Path(src_img).stem
            src_label = Path(src_img).parent.parent / "labels" / f"{src_stem}.txt"

            # 复制原图和标签
            dst_img = merged_images_dir / Path(src_img).name
            shutil.copy(src_img, dst_img)
            if src_label.exists():
                dst_label = merged_labels_dir / f"{src_stem}.txt"
                shutil.copy(src_label, dst_label)

            # 生成增强（variant_count 控制变体数量）
            error_type = case.error_type
            variant_count = get_variant_count(case.score)

            if variant_count > 0:
                # 增强图片和标签（标签与原图相同）
                variants = augment_image(src_img, error_type, str(merged_images_dir), variant_count)
                for v in variants:
                    v_stem = Path(v).stem
                    variant_stats[error_type] += 1
                    # 复制标签
                    if src_label.exists():
                        dst_label = merged_labels_dir / f"{v_stem}.txt"
                        shutil.copy(src_label, dst_label)

        # 生成 merged 数据集 YAML
        merged_yaml = output_dir / "merged" / "dataset_merged.yaml"
        self._generate_merged_yaml(merged_yaml, len(all_hard_examples))

        return str(merged_yaml)

    def _generate_merged_yaml(self, output_path, total_hard_examples):
        """生成合并后的数据集 YAML"""
        import yaml

        # 读取原始数据集配置
        with open(self.config.data, 'r') as f:
            data_config = yaml.safe_load(f)

        merged_path = Path(self.config.output) / "merged"
        data_config['path'] = str(merged_path.resolve())
        data_config['train'] = 'images'
        data_config['val'] = 'images'

        with open(output_path, 'w') as f:
            yaml.dump(data_config, f, default_flow_style=False)

    def _generate_weighted_config(self) -> str:
        """生成加权配置 (weighted 策略)"""
        output_dir = Path(self.config.output)
        config_path = output_dir / "weighted_config.yaml"

        # 按类别统计难例数量，生成 class_weights
        class_weights = {}
        for case in self.fp_cases + self.fn_cases + self.small_cases:
            cls = int(case.box[4]) if len(case.box) > 4 else 0
            if cls not in class_weights:
                class_weights[cls] = 0
            class_weights[cls] += 1

        # 转换为权重
        max_count = max(class_weights.values()) if class_weights else 1
        weights = [max_count / class_weights.get(i, 1) for i in range(len(class_weights))]

        config = {
            "hard_example_mining": {
                "strategy": "weighted",
                "class_weights": weights,
            }
        }

        with open(config_path, 'w') as f:
            yaml.dump(config, f)

        return str(config_path)

    def _generate_report(self) -> Dict[str, Any]:
        """生成难例分析报告"""
        fp_scores = [c.score for c in self.fp_cases] if self.fp_cases else [0]
        fn_scores = [c.score for c in self.fn_cases] if self.fn_cases else [0]
        small_scores = [c.score for c in self.small_cases] if self.small_cases else [0]

        return {
            "fp": {
                "count": len(self.fp_cases),
                "avg_score": sum(fp_scores) / len(fp_scores) if fp_scores else 0
            },
            "fn": {
                "count": len(self.fn_cases),
                "avg_score": sum(fn_scores) / len(fn_scores) if fn_scores else 0
            },
            "small": {
                "count": len(self.small_cases),
                "avg_score": sum(small_scores) / len(small_scores) if small_scores else 0
            }
        }
```

### 子任务 5.3: 写 main 函数

```python
def main():
    """CLI 入口"""
    import argparse
    import logging as log_module

    parser = argparse.ArgumentParser(description='YOLO 难例挖掘工具')
    parser.add_argument('--model', type=str, required=True, help='模型路径')
    parser.add_argument('--data', type=str, required=True, help='数据集 YAML')
    parser.add_argument('--output', type=str, default='./hard_examples', help='输出目录')
    parser.add_argument('--strategy', type=str, default='oversample',
                        choices=['oversample', 'weighted', 'filter'], help='策略')
    parser.add_argument('--iou-threshold', type=float, default=0.5, help='IoU 阈值')
    parser.add_argument('--conf-threshold', type=float, default=0.25, help='置信度阈值')
    parser.add_argument('--small-area-threshold', type=float, default=0.01, help='小目标面积阈值')
    parser.add_argument('--max-oversample', type=int, default=5, help='最大过采样倍数')
    parser.add_argument('--device', type=str, default='cpu', help='设备')

    args = parser.parse_args()

    log_module.basicConfig(
        level=log_module.INFO,
        format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s'
    )

    config = HardExampleMiningConfig(
        model=args.model,
        data=args.data,
        output=args.output,
        strategy=args.strategy,
        iou_threshold=args.iou_threshold,
        conf_threshold=args.conf_threshold,
        small_area_threshold=args.small_area_threshold,
        max_oversample=args.max_oversample,
        device=args.device,
    )

    miner = HardExampleMiner(config)
    result = miner.mine()

    print("\n" + "=" * 60)
    print("难例挖掘完成")
    print("=" * 60)
    print(f"FP (误检): {result['fp_count']}")
    print(f"FN (漏检): {result['fn_count']}")
    print(f"小目标: {result['small_count']}")
    print(f"合并数据集: {result['merged_dataset_yaml']}")
    print("=" * 60)
```

Run: `uv run pytest tests/tools/test_hard_example_miner.py -v` (全部测试)
Expected: PASS

### 子任务 5.4: 提交

```bash
git add src/tools/hard_example_miner.py tests/tools/test_hard_example_miner.py
git commit -m "feat: complete HardExampleMiner class implementation"
```

---

## Task 6: Pipeline 集成

**Files:**
- Modify: `src/tools/pipeline.py`
- Create: `configs/nodes/hard-example-mining.yaml`
- Modify: `pyproject.toml`

### 子任务 6.1: 创建 Pipeline 节点配置

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

### 子任务 6.2: 注册 Pipeline 工具

在 `pipeline.py` 中添加:

```python
@register_tool("hard-example-mining")
def tool_hard_example_mining(params: Dict[str, Any]) -> Dict[str, Any]:
    """难例挖掘"""
    from .hard_example_miner import HardExampleMiner, HardExampleMiningConfig

    model = params.get("model")
    data = params.get("data")

    if not model:
        raise ValueError("Parameter 'model' is required for hard-example-mining")
    if not data:
        raise ValueError("Parameter 'data' is required for hard-example-mining")

    config = HardExampleMiningConfig(
        model=model,
        data=data,
        output=params.get("output", "./hard_examples"),
        strategy=params.get("strategy", "oversample"),
        iou_threshold=params.get("iou_threshold", 0.5),
        conf_threshold=params.get("conf_threshold", 0.25),
        small_area_threshold=params.get("small_area_threshold", 0.01),
        max_oversample=params.get("max_oversample", 5),
        device=str(params.get("device", "cpu")),
    )

    miner = HardExampleMiner(config)
    return miner.mine()
```

### 子任务 6.3: 更新 pyproject.toml

添加 CLI 入口:

```toml
[project.scripts]
yolo-hard-example-mining = "src.tools.hard_example_miner:main"
```

### 子任务 6.4: 验证工具注册

```bash
uv run python -m src.tools.pipeline --list-tools
```

确认 `hard-example-mining` 在列表中

### 子任务 6.5: 提交

```bash
git add src/tools/pipeline.py configs/nodes/hard-example-mining.yaml pyproject.toml
git commit -m "feat: integrate hard-example-mining into pipeline"
```

---

## 验证命令

所有任务完成后，运行以下验证：

```bash
# 1. 运行所有测试
uv run pytest tests/tools/test_hard_example_miner.py -v

# 2. 验证 CLI 帮助
uv run python -m src.tools.hard_example_miner --help

# 3. 验证 Pipeline 工具注册
uv run python -m src.tools.pipeline --list-tools

# 4. 格式化代码
black src/tools/hard_example_miner.py
```
