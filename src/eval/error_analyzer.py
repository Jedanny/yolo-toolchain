"""
错误分析模块 - 分析检测错误并进行归因

分析 FP (False Positive) 和 FN (False Negative) 错误案例，
帮助识别模型弱点并指导数据增强策略。
"""

import argparse
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import cv2
import numpy as np
import yaml

logger = logging.getLogger("yolo_toolchain.error_analyzer")


@dataclass
class ErrorCase:
    """错误案例"""
    image_path: str
    error_type: str  # 'FP' or 'FN'
    predicted_box: List[float] = field(default_factory=list)  # xyxy
    gt_box: List[float] = field(default_factory=list)  # xyxy
    confidence: float = 0.0
    iou: float = 0.0
    reason: str = ""  # 错误归因: 'blur', 'occlusion', 'small', 'background', 'similar_class'


@dataclass
class ErrorAnalysisConfig:
    """错误分析配置"""
    iou_threshold: float = 0.5  # IoU 阈值判断匹配
    conf_threshold: float = 0.25  # 置信度阈值
    blur_threshold: int = 100  # 模糊判断阈值 (方差)
    small_area_threshold: float = 0.01  # 小目标面积比例
    occlusion_threshold: float = 0.3  # 遮挡比例阈值


class ErrorAnalyzer:
    """错误分析器"""

    def __init__(self, config: ErrorAnalysisConfig = None):
        self.config = config or ErrorAnalysisConfig()
        self.fp_errors: List[ErrorCase] = []  # 误检
        self.fn_errors: List[ErrorCase] = []  # 漏检
        self.correct: List[ErrorCase] = []  # 正确检测

    def compute_iou(self, box1: List[float], box2: List[float]) -> float:
        """计算两个框的 IoU"""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2

        # 计算交集
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)

        if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
            return 0.0

        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)

        # 计算各自面积
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)

        union_area = box1_area + box2_area - inter_area

        return inter_area / union_area if union_area > 0 else 0.0

    def analyze_image_blur(self, image_path: str) -> bool:
        """判断图像是否模糊"""
        img = cv2.imread(str(image_path))
        if img is None:
            return False

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        return variance < self.config.blur_threshold

    def analyze_box_size(self, box: List[float], image_area: float) -> str:
        """分析目标大小"""
        x_min, y_min, x_max, y_max = box
        box_area = (x_max - x_min) * (y_max - y_min)
        area_ratio = box_area / image_area

        if area_ratio < self.config.small_area_threshold:
            return 'small'
        return 'normal'

    def analyze_occlusion(self, box1: List[float], box2: List[float]) -> float:
        """分析两个框之间的遮挡程度"""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2

        # 计算交集
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)

        if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
            return 0.0

        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        smaller_box = min(
            (x1_max - x1_min) * (y1_max - y1_min),
            (x2_max - x2_min) * (y2_max - y2_min)
        )

        return inter_area / smaller_box if smaller_box > 0 else 0.0

    def analyze_errors(
        self,
        predictions: List[Dict],
        ground_truths: Dict[str, List[List[float]]],
        image_paths: Dict[str, str]
    ) -> Dict:
        """分析错误案例

        Args:
            predictions: 预测结果列表 [{'image': path, 'boxes': [[x1,y1,x2,y2,conf,cls], ...]}]
            ground_truths: 真值字典 {image_name: [[x1,y1,x2,y2,cls], ...]}
            image_paths: 图像路径字典 {image_name: full_path}
        """
        self.fp_errors = []
        self.fn_errors = []
        self.correct = []

        for pred in predictions:
            image_name = Path(pred['image']).name
            pred_boxes = pred.get('boxes', [])
            gt_boxes = ground_truths.get(image_name, [])

            matched_gt = set()
            matched_pred = set()

            # 按置信度排序
            pred_boxes_sorted = sorted(pred_boxes, key=lambda x: x[4], reverse=True)

            # 匹配预测和真值
            for pred_idx, pred_box in enumerate(pred_boxes_sorted):
                pred_xyxy = pred_box[:4]
                pred_conf = pred_box[4]
                pred_cls = int(pred_box[5]) if len(pred_box) > 5 else 0

                if pred_conf < self.config.conf_threshold:
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

                    iou = self.compute_iou(pred_xyxy, gt_xyxy)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx

                if best_iou >= self.config.iou_threshold:
                    # 正确检测
                    matched_gt.add(best_gt_idx)
                    matched_pred.add(pred_idx)

                    case = ErrorCase(
                        image_path=image_paths.get(image_name, image_name),
                        error_type='correct',
                        predicted_box=pred_xyxy,
                        gt_box=gt_boxes[best_gt_idx][:4],
                        confidence=pred_conf,
                        iou=best_iou
                    )
                    self.correct.append(case)
                elif best_iou > 0:
                    # 定位错误但类别正确
                    pass
                else:
                    # FP - 误检
                    image_path = image_paths.get(image_name, image_name)
                    reason = 'background'

                    if Path(image_path).exists() and self.analyze_image_blur(image_path):
                        reason = 'blur'

                    case = ErrorCase(
                        image_path=image_path,
                        error_type='FP',
                        predicted_box=pred_xyxy,
                        confidence=pred_conf,
                        iou=best_iou,
                        reason=reason
                    )
                    self.fp_errors.append(case)

            # 找出 FN - 漏检
            for gt_idx, gt_box in enumerate(gt_boxes):
                if gt_idx in matched_gt:
                    continue

                gt_xyxy = gt_box[:4]
                image_path = image_paths.get(image_name, image_name)

                reason = 'missing'
                if Path(image_path).exists():
                    if self.analyze_box_size(gt_xyxy, 640 * 640) == 'small':
                        reason = 'small_target'
                    elif self.analyze_image_blur(image_path):
                        reason = 'blur'

                case = ErrorCase(
                    image_path=image_path,
                    error_type='FN',
                    gt_box=gt_xyxy,
                    reason=reason
                )
                self.fn_errors.append(case)

        return self.build_report()

    def build_report(self) -> Dict:
        """构建错误分析报告"""
        total_fp = len(self.fp_errors)
        total_fn = len(self.fn_errors)
        total_correct = len(self.correct)

        # FP 归因统计
        fp_reasons = defaultdict(int)
        for err in self.fp_errors:
            fp_reasons[err.reason] += 1

        # FN 归因统计
        fn_reasons = defaultdict(int)
        for err in self.fn_errors:
            fn_reasons[err.reason] += 1

        report = {
            'summary': {
                'total_fp': total_fp,
                'total_fn': total_fn,
                'total_correct': total_correct,
                'precision': total_correct / (total_correct + total_fp) if (total_correct + total_fp) > 0 else 0,
                'recall': total_correct / (total_correct + total_fn) if (total_correct + total_fn) > 0 else 0,
            },
            'fp_analysis': {
                'count': total_fp,
                'reasons': dict(fp_reasons),
                'examples': [
                    {
                        'image': e.image_path,
                        'confidence': e.confidence,
                        'reason': e.reason,
                        'box': e.predicted_box
                    }
                    for e in self.fp_errors[:10]  # 最多10个例子
                ]
            },
            'fn_analysis': {
                'count': total_fn,
                'reasons': dict(fn_reasons),
                'examples': [
                    {
                        'image': e.image_path,
                        'reason': e.reason,
                        'box': e.gt_box
                    }
                    for e in self.fn_errors[:10]  # 最多10个例子
                ]
            },
            'recommendations': self._generate_recommendations(fp_reasons, fn_reasons)
        }

        return report

    def _generate_recommendations(
        self,
        fp_reasons: Dict[str, int],
        fn_reasons: Dict[str, int]
    ) -> List[str]:
        """根据错误分析生成建议"""
        recommendations = []

        total_errors = sum(fp_reasons.values()) + sum(fn_reasons.values())
        if total_errors == 0:
            recommendations.append("模型表现良好，未发现明显错误")
            return recommendations

        # FP 分析
        blur_fp = fp_reasons.get('blur', 0)
        background_fp = fp_reasons.get('background', 0)

        if blur_fp > sum(fp_reasons.values()) * 0.2:
            recommendations.append("检测到较多模糊图像误检，建议增强数据模糊 augmentation")

        if background_fp > sum(fp_reasons.values()) * 0.3:
            recommendations.append("检测到较多背景误检，建议增加负样本训练或提高置信度阈值")

        # FN 分析
        small_fn = fn_reasons.get('small_target', 0)
        blur_fn = fn_reasons.get('blur', 0)

        if small_fn > sum(fn_reasons.values()) * 0.3:
            recommendations.append("检测到较多小目标漏检，建议使用更小的输入尺寸或增加小目标增强")

        if blur_fn > sum(fn_reasons.values()) * 0.2:
            recommendations.append("检测到较多模糊图像漏检，建议增加模糊数据训练")

        if not recommendations:
            recommendations.append("错误分布较为均匀，建议综合优化数据增强策略")

        return recommendations

    def save_report(self, report: Dict, output_path: str) -> str:
        """保存报告到文件"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 转换 numpy 类型为 Python 原生类型
        def convert_to_native(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_native(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native(i) for i in obj]
            return obj

        report = convert_to_native(report)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        logger.info(f"Error analysis report saved to: {output_path}")
        return str(output_path)


def run_error_analysis(
    model_path: str,
    data_yaml: str,
    output_dir: str = "runs/error_analysis",
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.5
) -> Dict:
    """运行错误分析

    Args:
        model_path: 模型路径
        data_yaml: 数据集配置
        output_dir: 输出目录
        conf_threshold: 置信度阈值
        iou_threshold: IoU 阈值

    Returns:
        分析报告字典
    """
    from ultralytics import YOLO

    # 加载模型
    model = YOLO(model_path)

    # 加载数据集配置
    with open(data_yaml, 'r') as f:
        data_config = yaml.safe_load(f)

    # 获取测试集路径
    dataset_path = Path(data_config.get('path', Path(data_yaml).parent))
    test_path = dataset_path / data_config.get('test', '')
    val_path = dataset_path / data_config.get('val', test_path)

    # 优先使用测试集
    if not val_path.exists():
        val_path = test_path

    images_dir = val_path
    labels_dir = images_dir.parent / 'labels' / images_dir.name

    if not labels_dir.exists():
        # 尝试 labels/train 或 labels/val
        labels_dir = dataset_path / 'labels' / 'train'

    # 获取所有图像
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        image_files.extend(list(images_dir.glob(ext)))
        image_files.extend(list(images_dir.glob(ext.upper())))

    logger.info(f"Found {len(image_files)} images for error analysis")

    # 收集预测和真值
    predictions = []
    ground_truths = {}

    for img_path in image_files:
        image_name = img_path.name

        # 加载真值
        label_path = labels_dir / f"{img_path.stem}.txt"
        gt_boxes = []

        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls = int(parts[0])
                        cx, cy, w, h = map(float, parts[1:5])
                        # 转换为 xyxy
                        x1 = (cx - w/2) * img_path.stat().st_size  # 简化，实际需要图像尺寸
                        y1 = (cy - h/2) * img_path.stat().st_size
                        x2 = (cx + w/2) * img_path.stat().st_size
                        y2 = (cy + h/2) * img_path.stat().st_size
                        gt_boxes.append([x1, y1, x2, y2, cls])

            # 使用实际图像尺寸
            img = cv2.imread(str(img_path))
            if img is not None:
                h, w = img.shape[:2]
                gt_boxes = []
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

        # 运行预测
        results = model.predict(source=str(img_path), conf=conf_threshold, verbose=False)

        pred_boxes = []
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for box in boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cls = int(box.cls[0].cpu().numpy())
                pred_boxes.append([*xyxy, conf, cls])

        predictions.append({
            'image': str(img_path),
            'boxes': pred_boxes
        })

    # 分析错误
    image_paths = {img.name: str(img) for img in image_files}

    config = ErrorAnalysisConfig(
        iou_threshold=iou_threshold,
        conf_threshold=conf_threshold
    )

    analyzer = ErrorAnalyzer(config)
    report = analyzer.analyze_errors(predictions, ground_truths, image_paths)

    # 保存报告
    output_path = Path(output_dir) / 'error_analysis_report.json'
    analyzer.save_report(report, output_path)

    # 保存错误图片
    save_error_images(analyzer.fp_errors, Path(output_dir) / 'fp_errors')
    save_error_images(analyzer.fn_errors, Path(output_dir) / 'fn_errors')

    return report


def save_error_images(errors: List[ErrorCase], output_dir: Path):
    """保存错误案例图片"""
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, error in enumerate(errors[:20]):  # 最多保存20张
        img_path = Path(error.image_path)
        if not img_path.exists():
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        color = (0, 0, 255) if error.error_type == 'FP' else (255, 0, 0)
        box = error.predicted_box if error.error_type == 'FP' else error.gt_box

        if box:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            label = f"{error.error_type}: {error.reason}"
            if error.confidence > 0:
                label += f" {error.confidence:.2f}"

            cv2.putText(img, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        output_path = output_dir / f"{img_path.stem}_error_{i}.jpg"
        cv2.imwrite(str(output_path), img)


def main():
    """CLI 入口"""
    parser = argparse.ArgumentParser(description='YOLO 错误分析工具')
    parser.add_argument('--model', type=str, required=True, help='模型路径')
    parser.add_argument('--data', type=str, required=True, help='数据集 YAML')
    parser.add_argument('--output', type=str, default='runs/error_analysis', help='输出目录')
    parser.add_argument('--conf', type=float, default=0.25, help='置信度阈值')
    parser.add_argument('--iou', type=float, default=0.5, help='IoU 阈值')

    args = parser.parse_args()

    import logging as log_module
    log_module.basicConfig(
        level=log_module.INFO,
        format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s'
    )

    report = run_error_analysis(
        model_path=args.model,
        data_yaml=args.data,
        output_dir=args.output,
        conf_threshold=args.conf,
        iou_threshold=args.iou
    )

    print("\n" + "=" * 60)
    print("错误分析报告")
    print("=" * 60)
    print(f"误检 (FP): {report['summary']['total_fp']}")
    print(f"漏检 (FN): {report['summary']['total_fn']}")
    print(f"正确检测: {report['summary']['total_correct']}")
    print(f"Precision: {report['summary']['precision']:.4f}")
    print(f"Recall: {report['summary']['recall']:.4f}")
    print("\n建议:")
    for rec in report['recommendations']:
        print(f"  - {rec}")
    print("=" * 60)


if __name__ == '__main__':
    main()
