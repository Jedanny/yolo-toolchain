from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple


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


def get_oversample_ratio(hardness_score: float, max_oversample: int = 5) -> int:
    """根据难例分数返回过采样倍数"""
    if hardness_score <= 0.5:
        return 1
    elif hardness_score < 0.7:
        return 2
    elif hardness_score < 0.9:
        return 3
    else:
        return min(5, max_oversample)


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
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
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
                correct_cases.append({
                    "image": image_name,
                    "image_path": image_paths.get(image_name, image_name),
                    "error_type": "correct",
                    "box": pred_xyxy,
                    "score": 0.0,
                    "confidence": pred_conf
                })
            else:
                # FP
                score = compute_hardness_score("FP", 0.0, pred_conf)
                fp_cases.append({
                    "image": image_name,
                    "image_path": image_paths.get(image_name, image_name),
                    "error_type": "FP",
                    "box": pred_xyxy,
                    "score": score,
                    "confidence": pred_conf
                })

        # 找出 FN
        for gt_idx, gt_box in enumerate(gt_boxes):
            if gt_idx in matched_gt:
                continue
            gt_xyxy = gt_box[:4]
            score = compute_hardness_score("FN", 0.0, 0.0)
            fn_cases.append({
                "image": image_name,
                "image_path": image_paths.get(image_name, image_name),
                "error_type": "FN",
                "box": gt_xyxy,
                "score": score,
                "confidence": 0.0
            })

    return fp_cases, fn_cases, correct_cases


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
