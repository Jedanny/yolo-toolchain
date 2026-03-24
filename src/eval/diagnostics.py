"""
评估模块 - 诊断工具
用于分析误报和漏报问题
"""

import os
import json
import argparse
import shutil
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import cv2

from ultralytics import YOLO


class DetectionDiagnostics:
    """检测诊断工具"""

    def __init__(self, model_path: str, data_yaml: str, output_dir: str = "diagnostics"):
        self.model = YOLO(model_path)
        self.data_yaml = data_yaml
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 获取类别信息
        self.class_names = list(self.model.names.values())
        self.num_classes = len(self.class_names)

        # 统计数据
        self.stats = {
            'true_positives': defaultdict(int),
            'false_positives': defaultdict(int),
            'false_negatives': defaultdict(int),
            'confusion_matrix': np.zeros((self.num_classes + 1, self.num_classes + 1), dtype=int),
            'missed_detections': [],  # 漏检样本
            'false_alarms': [],       # 误报样本
        }

    def run_full_diagnostics(self, split: str = 'val', conf_threshold: float = 0.25) -> Dict:
        """运行完整诊断"""
        print(f"运行诊断分析...")

        # 获取验证数据路径
        import yaml
        with open(self.data_yaml, 'r') as f:
            data_config = yaml.safe_load(f)

        val_dir = Path(data_config.get('path', '.')) / data_config.get('val', 'images/val')

        if not val_dir.exists():
            print(f"验证目录不存在: {val_dir}")
            return {}

        # 分析每张图像
        image_files = list(val_dir.glob('*.jpg')) + list(val_dir.glob('*.png'))
        image_files.extend(val_dir.glob('*.jpeg'))

        for img_path in image_files:
            self._analyze_image(img_path, conf_threshold)

        # 生成诊断报告
        report = self._generate_report()

        # 生成可视化
        self._plot_confusion_matrix()
        self._plot_class_performance()
        self._plot_confidence_distribution()

        return report

    def _analyze_image(self, img_path: Path, conf_threshold: float):
        """分析单张图像"""
        # 尝试多个可能的标签路径
        possible_label_paths = [
            img_path.parent.parent / 'labels' / f"{img_path.stem}.txt",
            Path(str(img_path).replace('images', 'labels').replace('.jpg', '.txt').replace('.png', '.txt')),
        ]

        gt_boxes = []
        for lp in possible_label_paths:
            if lp.exists():
                with open(lp, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            cls_id = int(parts[0])
                            gt_boxes.append({
                                'class_id': cls_id,
                                'bbox': [float(x) for x in parts[1:5]]
                            })
                break

        # 运行预测
        results = self.model.predict(
            source=str(img_path),
            conf=conf_threshold,
            verbose=False,
            save=False
        )

        pred_boxes = []
        if len(results) > 0 and hasattr(results[0], 'boxes') and results[0].boxes is not None:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                pred_boxes.append({
                    'class_id': int(boxes.cls[i].item()),
                    'conf': boxes.conf[i].item(),
                    'bbox': boxes.xywhn[i].tolist()  # 归一化的xywh
                })

        # 匹配GT和预测
        self._match_boxes(gt_boxes, pred_boxes, img_path)

    def _match_boxes(self, gt_boxes: List, pred_boxes: List, img_path: Path, iou_threshold: float = 0.5):
        """匹配GT和预测框"""
        matched_gt = set()
        matched_pred = set()

        # 计算IoU
        for i, pred in enumerate(pred_boxes):
            best_iou = 0
            best_gt_idx = -1

            for j, gt in enumerate(gt_boxes):
                if j in matched_gt:
                    continue
                if pred['class_id'] != gt['class_id']:
                    continue

                iou = self._compute_iou(pred['bbox'], gt['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j

            if best_iou >= iou_threshold:
                matched_gt.add(best_gt_idx)
                matched_pred.add(i)
                self.stats['true_positives'][pred['class_id']] += 1
                self.stats['confusion_matrix'][pred['class_id']][gt['class_id']] += 1
            else:
                self.stats['false_positives'][pred['class_id']] += 1
                self.stats['confusion_matrix'][pred['class_id']][self.num_classes] += 1
                # 记录误报
                if len(self.stats['false_alarms']) < 100:
                    self.stats['false_alarms'].append({
                        'image': str(img_path),
                        'predicted_class': self.class_names[pred['class_id']],
                        'confidence': pred['conf']
                    })

        # 统计漏检
        for j, gt in enumerate(gt_boxes):
            if j not in matched_gt:
                self.stats['false_negatives'][gt['class_id']] += 1
                # 记录漏检
                if len(self.stats['missed_detections']) < 100:
                    self.stats['missed_detections'].append({
                        'image': str(img_path),
                        'gt_class': self.class_names[gt['class_id']],
                        'bbox': gt['bbox']
                    })

    def _compute_iou(self, box1: List, box2: List) -> float:
        """计算两个归一化边界框的IoU"""
        # box format: [cx, cy, w, h]
        x1_min = box1[0] - box1[2] / 2
        y1_min = box1[1] - box1[3] / 2
        x1_max = box1[0] + box1[2] / 2
        y1_max = box1[1] + box1[3] / 2

        x2_min = box2[0] - box2[2] / 2
        y2_min = box2[1] - box2[3] / 2
        x2_max = box2[0] + box2[2] / 2
        y2_max = box2[1] + box2[3] / 2

        # 计算交集
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)

        if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
            return 0.0

        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)

        # 计算并集
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area

        return inter_area / union_area if union_area > 0 else 0.0

    def _generate_report(self) -> Dict:
        """生成诊断报告"""
        report = {
            'summary': {},
            'per_class': {},
            'recommendations': []
        }

        # 计算整体指标
        total_tp = sum(self.stats['true_positives'].values())
        total_fp = sum(self.stats['false_positives'].values())
        total_fn = sum(self.stats['false_negatives'].values())

        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        report['summary'] = {
            'total_images_analyzed': len(self.stats['missed_detections']) + len(self.stats['false_alarms']),
            'true_positives': total_tp,
            'false_positives': total_fp,
            'false_negatives': total_fn,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }

        # 各类别指标
        for i, class_name in enumerate(self.class_names):
            tp = self.stats['true_positives'].get(i, 0)
            fp = self.stats['false_positives'].get(i, 0)
            fn = self.stats['false_negatives'].get(i, 0)

            cls_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            cls_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            cls_f1 = 2 * cls_precision * cls_recall / (cls_precision + cls_recall) if (cls_precision + cls_recall) > 0 else 0

            report['per_class'][class_name] = {
                'tp': tp,
                'fp': fp,
                'fn': fn,
                'precision': cls_precision,
                'recall': cls_recall,
                'f1': cls_f1
            }

        # 生成建议
        self._generate_recommendations(report)

        # 保存报告
        report_path = self.output_dir / 'diagnostics_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"\n诊断报告已保存: {report_path}")
        return report

    def _generate_recommendations(self, report: Dict):
        """生成改进建议"""
        recommendations = []

        # 分析误报严重的类别
        for class_name, metrics in report['per_class'].items():
            if metrics['precision'] < 0.5:
                recommendations.append(
                    f"类别 '{class_name}' 误报严重 (Precision={metrics['precision']:.2%})。"
                    f"建议：1) 检查该类别标注是否准确; 2) 增加该类别样本; 3) 提高conf阈值"
                )

            if metrics['recall'] < 0.5:
                recommendations.append(
                    f"类别 '{class_name}' 漏报严重 (Recall={metrics['recall']:.2%})。"
                    f"建议：1) 补充该类别更多样本; 2) 使用数据增强; 3) 检查遮挡情况"
                )

        if report['summary']['precision'] < report['summary']['recall']:
            recommendations.append(
                f"整体误报较多 (Precision={report['summary']['precision']:.2%})。"
                f"建议：提高conf阈值或检查类别间相似性"
            )
        elif report['summary']['recall'] < report['summary']['precision']:
            recommendations.append(
                f"整体漏报较多 (Recall={report['summary']['recall']:.2%})。"
                f"建议：降低conf阈值或使用TTA增强"
            )

        report['recommendations'] = recommendations

    def _plot_confusion_matrix(self):
        """绘制混淆矩阵"""
        fig, ax = plt.subplots(figsize=(12, 10))

        # 添加背景类和总计列
        labels = self.class_names + ['Background']
        confusion = self.stats['confusion_matrix']

        # 添加总计
        totals = confusion.sum(axis=1, keepdims=True)
        confusion_with_totals = np.hstack([confusion, totals])

        sns.heatmap(
            confusion_with_totals,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=labels,
            yticklabels=labels,
            ax=ax
        )

        ax.set_xlabel('Ground Truth')
        ax.set_ylabel('Prediction')
        ax.set_title('Detection Confusion Matrix')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'confusion_matrix.png', dpi=150)
        plt.close()

    def _plot_class_performance(self):
        """绘制各类别性能对比"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        classes = list(self.stats['true_positives'].keys())
        precision_vals = []
        recall_vals = []
        f1_vals = []

        for i, class_name in enumerate(self.class_names):
            tp = self.stats['true_positives'].get(i, 0)
            fp = self.stats['false_positives'].get(i, 0)
            fn = self.stats['false_negatives'].get(i, 0)

            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

            precision_vals.append(prec)
            recall_vals.append(rec)
            f1_vals.append(f1)

        x = np.arange(len(self.class_names))

        axes[0].bar(x, precision_vals)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(self.class_names, rotation=45, ha='right')
        axes[0].set_ylabel('Precision')
        axes[0].set_title('Precision by Class')
        axes[0].set_ylim([0, 1])

        axes[1].bar(x, recall_vals)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(self.class_names, rotation=45, ha='right')
        axes[1].set_ylabel('Recall')
        axes[1].set_title('Recall by Class')
        axes[1].set_ylim([0, 1])

        axes[2].bar(x, f1_vals)
        axes[2].set_xticks(x)
        axes[2].set_xticklabels(self.class_names, rotation=45, ha='right')
        axes[2].set_ylabel('F1 Score')
        axes[2].set_title('F1 Score by Class')
        axes[2].set_ylim([0, 1])

        plt.tight_layout()
        plt.savefig(self.output_dir / 'class_performance.png', dpi=150)
        plt.close()

    def _plot_confidence_distribution(self):
        """绘制置信度分布"""
        # 这个功能需要在预测时记录置信度
        # 简化版本：绘制各阈值下的precision-recall曲线

        thresholds = np.arange(0.1, 0.9, 0.1)
        precision_curve = []
        recall_curve = []

        for thresh in thresholds:
            # 重新评估（简化版，仅统计）
            tp = sum(self.stats['true_positives'].values())
            fp = sum(self.stats['false_positives'].values())

            # 模拟阈值影响
            prec = tp / (tp + fp * (1 - thresh)) if (tp + fp)> 0 else 0
            rec = prec * thresh  # 简化

            precision_curve.append(prec)
            recall_curve.append(rec)

        fig, ax = plt.subplots()
        ax.plot(thresholds, precision_curve, 'b-', label='Precision')
        ax.plot(thresholds, recall_curve, 'g-', label='Recall')
        ax.set_xlabel('Confidence Threshold')
        ax.set_ylabel('Score')
        ax.set_title('Precision-Recall vs Confidence Threshold')
        ax.legend()
        ax.grid(True)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'confidence_distribution.png', dpi=150)
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='YOLO检测诊断工具')
    parser.add_argument('--model', type=str, required=True, help='模型路径')
    parser.add_argument('--data', type=str, required=True, help='数据集配置')
    parser.add_argument('--output', type=str, default='diagnostics', help='输出目录')
    parser.add_argument('--conf', type=float, default=0.25, help='置信度阈值')
    parser.add_argument('--config', type=str, help='配置文件路径')
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='日志级别')

    args = parser.parse_args()

    import logging as log_module
    log_level = getattr(log_module, args.log_level.upper(), log_module.INFO)
    log_module.basicConfig(
        level=log_level,
        format='%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    diagnostics = DetectionDiagnostics(args.model, args.data, args.output)
    report = diagnostics.run_full_diagnostics(conf_threshold=args.conf)

    print("\n=== 诊断报告 ===")
    print(f"\n整体性能:")
    print(f"  Precision: {report['summary'].get('precision', 0):.2%}")
    print(f"  Recall: {report['summary'].get('recall', 0):.2%}")
    print(f"  F1 Score: {report['summary'].get('f1_score', 0):.2%}")

    print(f"\n各类别性能:")
    for class_name, metrics in report.get('per_class', {}).items():
        print(f"  {class_name}: P={metrics['precision']:.2%}, R={metrics['recall']:.2%}, F1={metrics['f1']:.2%}")

    if report.get('recommendations'):
        print(f"\n改进建议:")
        for i, rec in enumerate(report['recommendations'], 1):
            print(f"  {i}. {rec}")


if __name__ == '__main__':
    main()
