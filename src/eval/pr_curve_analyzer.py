"""
PR/F1 曲线分析模块

分析不同置信度阈值下的 Precision-Recall 曲线，
计算最优阈值，生成 F1 曲线，帮助选择最佳检测参数。
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict

import cv2
import numpy as np
import yaml

logger = logging.getLogger("yolo_toolchain.pr_analyzer")


@dataclass
class PRCurveConfig:
    """PR曲线分析配置"""
    model_path: str = ""
    data_yaml: str = ""
    output_dir: str = "runs/pr_analysis"
    conf_thresholds: List[float] = field(default_factory=lambda: None)  # None = 自动生成
    iou_threshold: float = 0.5  # IoU 阈值
    min_conf: float = 0.01  # 最小置信度
    max_conf: float = 0.99  # 最大置信度
    num_thresholds: int = 100  # 阈值数量


class PRCurveAnalyzer:
    """PR/F1 曲线分析器"""

    def __init__(self, config: PRCurveConfig):
        self.config = config
        self.results = {}
        self.pr_data = []

    def compute_iou(self, box1: List[float], box2: List[float]) -> float:
        """计算两个框的 IoU"""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2

        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)

        if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
            return 0.0

        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area

        return inter_area / union_area if union_area > 0 else 0.0

    def load_ground_truth(self, labels_dir: Path, image_path: Path, img_h: int, img_w: int) -> List[Dict]:
        """加载真值"""
        label_path = labels_dir / f"{image_path.stem}.txt"
        gt_boxes = []

        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls = int(parts[0])
                        cx, cy, bw, bh = map(float, parts[1:5])
                        x1 = (cx - bw/2) * img_w
                        y1 = (cy - bh/2) * img_h
                        x2 = (cx + bw/2) * img_w
                        y2 = (cy + bh/2) * img_h
                        gt_boxes.append({'class_id': cls, 'box': [x1, y1, x2, y2]})

        return gt_boxes

    def evaluate_at_threshold(
        self,
        predictions: List[Dict],
        ground_truths: List[Dict],
        conf_threshold: float
    ) -> Tuple[float, float, float]:
        """在给定置信度阈值下计算 Precision 和 Recall

        Returns:
            (precision, recall, f1)
        """
        tp = 0
        fp = 0
        fn = 0

        # Filter predictions by confidence
        filtered_preds = [p for p in predictions if p['confidence'] >= conf_threshold]

        matched_gt = set()

        for pred in filtered_preds:
            pred_box = pred['box']
            pred_cls = pred['class_id']

            best_iou = 0
            best_gt_idx = -1

            for gt_idx, gt in enumerate(ground_truths):
                if gt_idx in matched_gt:
                    continue
                if gt['class_id'] != pred_cls:
                    continue

                iou = self.compute_iou(pred_box, gt['box'])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            if best_iou >= self.config.iou_threshold:
                tp += 1
                matched_gt.add(best_gt_idx)
            else:
                fp += 1

        fn = len(ground_truths) - len(matched_gt)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return precision, recall, f1

    def analyze(self, model, images_dir: Path, labels_dir: Path) -> Dict[str, Any]:
        """运行 PR 曲线分析

        Args:
            model: YOLO 模型
            images_dir: 图像目录
            labels_dir: 标签目录

        Returns:
            分析结果字典
        """
        # Generate confidence thresholds
        if self.config.conf_thresholds is None:
            thresholds = np.linspace(self.config.min_conf, self.config.max_conf, self.config.num_thresholds)
            thresholds = thresholds.tolist()
        else:
            thresholds = self.config.conf_thresholds

        # Collect all predictions and ground truths
        all_predictions = []
        all_ground_truths = []

        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            image_files.extend(list(images_dir.glob(ext)))
            image_files.extend(list(images_dir.glob(ext.upper())))

        logger.info(f"Found {len(image_files)} images")

        for img_path in image_files:
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            h, w = img.shape[:2]

            # Load ground truth
            gt_boxes = self.load_ground_truth(labels_dir, img_path, h, w)
            all_ground_truths.extend([{'image': str(img_path), **gt} for gt in gt_boxes])

            # Run prediction
            results = model.predict(source=str(img_path), verbose=False)

            predictions = []
            if len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                for box in boxes:
                    xyxy = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())
                    cls = int(box.cls[0].cpu().numpy())
                    predictions.append({
                        'image': str(img_path),
                        'class_id': cls,
                        'confidence': conf,
                        'box': xyxy.tolist()
                    })

            all_predictions.extend(predictions)

        # Evaluate at each threshold
        pr_curve = []
        f1_curve = []

        for conf in thresholds:
            # Filter predictions for this image
            img_preds = [p for p in all_predictions if p['image'] == str(img_path)]
            img_gts = [g for g in all_ground_truths if g['image'] == str(img_path)]

            precision, recall, f1 = self.evaluate_at_threshold(
                all_predictions, all_ground_truths, conf
            )

            pr_curve.append({'confidence': conf, 'precision': precision, 'recall': recall})
            f1_curve.append({'confidence': conf, 'f1': f1})

        # Find optimal threshold
        best_f1 = 0
        best_threshold = 0.5
        best_precision = 0
        best_recall = 0

        for item in f1_curve:
            if item['f1'] > best_f1:
                best_f1 = item['f1']
                best_threshold = item['confidence']

        # Get precision/recall at best threshold
        for item in pr_curve:
            if abs(item['confidence'] - best_threshold) < 0.01:
                best_precision = item['precision']
                best_recall = item['recall']
                break

        # Calculate AUC PR
        precisions = [p['precision'] for p in pr_curve]
        recalls = [p['recall'] for p in pr_curve]

        # Sort by recall
        sorted_pairs = sorted(zip(recalls, precisions), key=lambda x: x[0])
        recalls_sorted, precisions_sorted = zip(*sorted_pairs)

        auc_pr = 0.0
        for i in range(len(recalls_sorted) - 1):
            width = recalls_sorted[i+1] - recalls_sorted[i]
            height = (precisions_sorted[i] + precisions_sorted[i+1]) / 2
            auc_pr += width * height

        # Build result
        self.results = {
            'pr_curve': pr_curve,
            'f1_curve': f1_curve,
            'optimal': {
                'threshold': best_threshold,
                'f1': best_f1,
                'precision': best_precision,
                'recall': best_recall,
            },
            'auc_pr': auc_pr,
            'thresholds': thresholds[:10],  # Top 10 thresholds for display
        }

        return self.results

    def generate_report(self) -> Dict[str, Any]:
        """生成分析报告"""
        return {
            'optimal_threshold': self.results.get('optimal', {}).get('threshold', 0.5),
            'optimal_f1': self.results.get('optimal', {}).get('f1', 0),
            'optimal_precision': self.results.get('optimal', {}).get('precision', 0),
            'optimal_recall': self.results.get('optimal', {}).get('recall', 0),
            'auc_pr': self.results.get('auc_pr', 0),
            'total_thresholds_evaluated': len(self.results.get('pr_curve', [])),
        }

    def save_results(self, output_path: Path) -> None:
        """保存结果"""
        output_path.mkdir(parents=True, exist_ok=True)

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

        # Save JSON results
        results_file = output_path / 'pr_analysis_results.json'
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump({
                'config': {
                    'iou_threshold': self.config.iou_threshold,
                    'min_conf': self.config.min_conf,
                    'max_conf': self.config.max_conf,
                    'num_thresholds': self.config.num_thresholds,
                },
                'results': convert_to_native(self.results),
                'report': convert_to_native(self.generate_report()),
            }, f, indent=2, ensure_ascii=False)

        logger.info(f"Results saved to: {results_file}")

    def plot_curves(self, output_path: Path) -> None:
        """绘制 PR 和 F1 曲线"""
        try:
            import matplotlib.pyplot as plt
            plt.rcParams['font.size'] = 10
        except ImportError:
            logger.warning("matplotlib not installed, skipping visualization")
            return

        pr_curve = self.results.get('pr_curve', [])
        f1_curve = self.results.get('f1_curve', [])
        optimal = self.results.get('optimal', {})

        if not pr_curve:
            return

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Plot 1: PR Curve
        ax1 = axes[0]
        recalls = [p['recall'] for p in pr_curve]
        precisions = [p['precision'] for p in pr_curve]
        confs = [p['confidence'] for p in pr_curve]

        # Color by confidence
        scatter = ax1.scatter(recalls, precisions, c=confs, cmap='viridis', s=30, alpha=0.7)
        ax1.plot(recalls, precisions, 'b-', alpha=0.3, linewidth=1)
        ax1.set_xlabel('Recall')
        ax1.set_ylabel('Precision')
        ax1.set_title(f'PR Curve (AUC={self.results.get("auc_pr", 0):.4f})')
        ax1.set_xlim([0, 1.05])
        ax1.set_ylim([0, 1.05])
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Mark optimal point
        opt_recall = optimal.get('recall', 0)
        opt_precision = optimal.get('precision', 0)
        ax1.scatter([opt_recall], [opt_precision], c='red', s=100, marker='*', zorder=5, label=f'Optimal ({opt_recall:.2f}, {opt_precision:.2f})')
        ax1.legend()

        # Colorbar
        cbar = plt.colorbar(scatter, ax=ax1)
        cbar.set_label('Confidence')

        # Plot 2: F1 vs Confidence
        ax2 = axes[1]
        f1_confs = [f['confidence'] for f in f1_curve]
        f1_scores = [f['f1'] for f in f1_curve]

        ax2.plot(f1_confs, f1_scores, 'g-', linewidth=2)
        ax2.axvline(x=optimal.get('threshold', 0.5), color='r', linestyle='--', alpha=0.7, label=f'Best threshold={optimal.get("threshold", 0.5):.3f}')
        ax2.axhline(y=optimal.get('f1', 0), color='r', linestyle=':', alpha=0.7)
        ax2.scatter([optimal.get('threshold', 0.5)], [optimal.get('f1', 0)], c='red', s=100, marker='*', zorder=5)
        ax2.set_xlabel('Confidence Threshold')
        ax2.set_ylabel('F1 Score')
        ax2.set_title(f'F1 Score vs Confidence (Best F1={optimal.get("f1", 0):.4f})')
        ax2.set_xlim([0, 1])
        ax2.set_ylim([0, 1.05])
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        plt.tight_layout()
        plt.savefig(output_path / 'pr_curves.png', dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"PR curves saved to: {output_path / 'pr_curves.png'}")


def run_pr_analysis(
    model_path: str,
    data_yaml: str,
    output_dir: str = "runs/pr_analysis",
    conf_thresholds: Optional[List[float]] = None,
    iou_threshold: float = 0.5,
    num_thresholds: int = 100,
    device: str = ""
) -> Dict[str, Any]:
    """运行 PR 曲线分析

    Args:
        model_path: 模型路径
        data_yaml: 数据集配置
        output_dir: 输出目录
        conf_thresholds: 自定义置信度阈值列表
        iou_threshold: IoU 阈值
        num_thresholds: 自动生成的阈值数量
        device: 设备 (如 '0', 'cpu', 'mps')

    Returns:
        分析结果
    """
    from ultralytics import YOLO

    # Load model
    model = YOLO(model_path)
    if device:
        model.to(device)

    # Load dataset config
    with open(data_yaml, 'r') as f:
        data_config = yaml.safe_load(f)

    # Get paths
    dataset_path = Path(data_config.get('path', Path(data_yaml).parent))
    val_path = dataset_path / data_config.get('val', 'images/val')
    test_path = dataset_path / data_config.get('test', 'images/test')

    if test_path.exists():
        images_dir = test_path
    elif val_path.exists():
        images_dir = val_path
    else:
        raise ValueError(f"Could not find validation or test images")

    labels_dir = images_dir.parent / 'labels' / images_dir.name

    if not labels_dir.exists():
        labels_dir = dataset_path / 'labels' / 'val'

    logger.info(f"Images: {images_dir}")
    logger.info(f"Labels: {labels_dir}")

    # Create config
    config = PRCurveConfig(
        model_path=model_path,
        data_yaml=data_yaml,
        output_dir=output_dir,
        conf_thresholds=conf_thresholds,
        iou_threshold=iou_threshold,
        num_thresholds=num_thresholds,
    )

    # Run analysis
    analyzer = PRCurveAnalyzer(config)
    results = analyzer.analyze(model, images_dir, labels_dir)

    # Save and plot
    output_path = Path(output_dir)
    analyzer.save_results(output_path)
    analyzer.plot_curves(output_path)

    return results


def main():
    """CLI 入口"""
    parser = argparse.ArgumentParser(description='PR/F1 曲线分析工具')
    parser.add_argument('--model', type=str, required=True, help='模型路径')
    parser.add_argument('--data', type=str, required=True, help='数据集 YAML')
    parser.add_argument('--output-dir', type=str, default='runs/pr_analysis', help='输出目录')
    parser.add_argument('--iou', type=float, default=0.5, help='IoU 阈值')
    parser.add_argument('--num-thresholds', type=int, default=100, help='阈值数量')
    parser.add_argument('--device', type=str, default='', help='设备 (如 "0", "cpu", "mps")')

    args = parser.parse_args()

    import logging as log_module
    log_module.basicConfig(
        level=log_module.INFO,
        format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s'
    )

    results = run_pr_analysis(
        model_path=args.model,
        data_yaml=args.data,
        output_dir=args.output_dir,
        iou_threshold=args.iou,
        num_thresholds=args.num_thresholds,
        device=args.device,
    )

    # Print summary
    print("\n" + "=" * 60)
    print("PR/F1 曲线分析结果")
    print("=" * 60)
    print(f"最优置信度阈值: {results['optimal']['threshold']:.4f}")
    print(f"最优 F1 分数: {results['optimal']['f1']:.4f}")
    print(f"最优 Precision: {results['optimal']['precision']:.4f}")
    print(f"最优 Recall: {results['optimal']['recall']:.4f}")
    print(f"AUC-PR: {results['auc_pr']:.4f}")
    print("=" * 60)
    print(f"详细结果已保存到: {args.output_dir}/")


if __name__ == '__main__':
    main()
