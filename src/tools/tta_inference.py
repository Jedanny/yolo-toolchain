"""
TTA 推理工具 - 测试时增强，提升推理精度
"""

import argparse
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
from ultralytics import YOLO

logger = logging.getLogger("yolo_toolchain.tta_inference")


def _to_python_scalar(val) -> float:
    """将 numpy 标量转换为 Python 标量"""
    if hasattr(val, 'item'):
        return float(val.item())
    return float(val)


def wbf_fusion(
    boxes_list: List[List[np.ndarray]],
    scores_list: List[List[np.ndarray]],
    labels_list: List[List[np.ndarray]],
    iou_threshold: float = 0.5
) -> tuple:
    """
    Weighted Boxes Fusion - 加权框融合

    Args:
        boxes_list: 每张图片的检测框列表 [x1, y1, x2, y2] (归一化 0-1)
        scores_list: 每张图片的置信度列表
        labels_list: 每张图片的类别标签列表
        iou_threshold: IoU 阈值

    Returns:
        融合后的 (boxes, scores, labels)
    """
    # 展平所有图片的检测结果
    all_boxes = []
    all_scores = []
    all_labels = []

    for img_boxes, img_scores, img_labels in zip(boxes_list, scores_list, labels_list):
        for box, score, label in zip(img_boxes, img_scores, img_labels):
            all_boxes.append(box)
            all_scores.append(score)
            all_labels.append(label)

    if not all_boxes:
        return [], [], []

    # 按类别分组
    class_boxes = {}
    for box, score, label in zip(all_boxes, all_scores, all_labels):
        label_int = _to_python_scalar(label)
        if label_int not in class_boxes:
            class_boxes[label_int] = []
        class_boxes[label_int].append((box, score))

    result_boxes = []
    result_scores = []
    result_labels = []

    # 对每个类别分别处理
    for label_int, detections in class_boxes.items():
        # 按置信度排序
        detections = sorted(detections, key=lambda x: _to_python_scalar(x[1]), reverse=True)

        clusters = []  # List of (boxes, scores) for each cluster

        for box, score in detections:
            box = np.array(box)
            matched = False

            # 尝试匹配已有簇
            for i, cluster in enumerate(clusters):
                cluster_boxes, cluster_scores = cluster
                # 计算与簇中所有框的最大 IoU
                max_iou = 0
                for c_box in cluster_boxes:
                    iou = compute_iou_xyxy(box, c_box)
                    max_iou = max(max_iou, iou)

                if max_iou >= iou_threshold:
                    # 融合到该簇
                    cluster_boxes.append(box)
                    cluster_scores.append(score)
                    matched = True
                    break

            if not matched:
                # 创建新簇
                clusters.append(([box], [score]))

        # 计算每个簇的融合结果
        for cluster_boxes, cluster_scores in clusters:
            cluster_boxes = np.array(cluster_boxes)
            cluster_scores = np.array([_to_python_scalar(s) for s in cluster_scores])

            # 加权平均坐标和置信度
            weights = cluster_scores / cluster_scores.sum()
            fused_box = np.average(cluster_boxes, axis=0, weights=weights)
            fused_score = float(np.average(cluster_scores, weights=weights))  # 加权平均置信度

            result_boxes.append(fused_box)
            result_scores.append(fused_score)
            result_labels.append(int(label_int))

    return result_boxes, result_scores, result_labels


def compute_iou_xyxy(box1: np.ndarray, box2: np.ndarray) -> float:
    """计算两个框的 IoU [x1, y1, x2, y2]"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0


@dataclass
class TTAConfig:
    """TTA 推理配置"""
    model: str
    images: str
    output_dir: str = "./tta_results"
    scales: List[float] = field(default_factory=lambda: [0.8, 1.0, 1.2])
    flip: bool = True
    conf: float = 0.25
    iou: float = 0.7
    wbf_iou: float = 0.5
    device: str = "cpu"
    save_txt: bool = False
    save_conf: bool = True
    save_vis: bool = True


class TTAInference:
    """TTA 推理器"""

    def __init__(self, config: TTAConfig):
        self.config = config
        self.model = None

    def _load_model(self):
        """加载模型"""
        if self.model is None:
            logger.info(f"加载模型: {self.config.model}")
            self.model = YOLO(self.config.model)

    def _get_image_files(self) -> List[Path]:
        """获取图片文件列表"""
        images_path = Path(self.config.images)
        if images_path.is_file():
            return [images_path]
        elif images_path.is_dir():
            files = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                files.extend(images_path.glob(ext))
                files.extend(images_path.glob(ext.upper()))
            return sorted(set(files))
        raise ValueError(f"Images path not found: {self.config.images}")

    def _tta_predict(self, img: np.ndarray, orig_h: int, orig_w: int) -> tuple:
        """对单张图片进行 TTA 推理"""
        all_boxes = []
        all_scores = []
        all_labels = []

        # 多尺度推理
        for scale in self.config.scales:
            new_h, new_w = int(orig_h * scale), int(orig_w * scale)
            resized = cv2.resize(img, (new_w, new_h))

            # 原图
            results = self.model.predict(
                source=resized,
                conf=self.config.conf,
                iou=self.config.iou,
                device=self.config.device,
                verbose=False,
            )

            for r in results:
                if r.boxes is None:
                    continue
                for box in r.boxes:
                    xyxy = box.xyxy[0].cpu().numpy()
                    # 还原到原图坐标
                    xyxy[0] /= scale
                    xyxy[1] /= scale
                    xyxy[2] /= scale
                    xyxy[3] /= scale
                    all_boxes.append(xyxy)
                    all_scores.append(float(box.conf[0]))
                    all_labels.append(int(box.cls[0]))

            # 水平翻转
            if self.config.flip:
                flipped = cv2.flip(resized, 1)
                results = self.model.predict(
                    source=flipped,
                    conf=self.config.conf,
                    iou=self.config.iou,
                    device=self.config.device,
                    verbose=False,
                )

                for r in results:
                    if r.boxes is None:
                        continue
                    for box in r.boxes:
                        xyxy = box.xyxy[0].cpu().numpy()
                        # 翻转后坐标转换回原图坐标系
                        x1_orig, y1, x2_orig, y2 = xyxy
                        # 水平翻转: x1' = new_w - x2_orig, x2' = new_w - x1_orig
                        x1_flipped = new_w - x2_orig
                        x2_flipped = new_w - x1_orig
                        # 缩放到原始图片尺寸
                        x1_flipped /= scale
                        x2_flipped /= scale
                        y1 /= scale
                        y2 /= scale
                        all_boxes.append(np.array([x1_flipped, y1, x2_flipped, y2]))
                        all_scores.append(float(box.conf[0]))
                        all_labels.append(int(box.cls[0]))

        # 转换为 numpy 数组
        if all_boxes:
            all_boxes = [np.array(b) for b in all_boxes]
            all_scores = [np.array(s) for s in all_scores]
            all_labels = [np.array(l) for l in all_labels]
            return [all_boxes], [all_scores], [all_labels]
        return [], [], []

    def run(self) -> Dict[str, Any]:
        """执行 TTA 推理"""
        self._load_model()
        image_files = self._get_image_files()
        logger.info(f"找到 {len(image_files)} 张图片")

        output_path = Path(self.config.output_dir)
        if self.config.save_vis:
            (output_path / "images").mkdir(parents=True, exist_ok=True)
        if self.config.save_txt:
            (output_path / "labels").mkdir(parents=True, exist_ok=True)

        total_detections = 0
        results_list = []

        for img_path in image_files:
            img = cv2.imread(str(img_path))
            if img is None:
                logger.warning(f"读取图片失败: {img_path}")
                continue

            orig_h, orig_w = img.shape[:2]

            # TTA 推理
            boxes_list, scores_list, labels_list = self._tta_predict(img, orig_h, orig_w)

            # WBF 融合
            if boxes_list and len(boxes_list[0]) > 0:
                fused_boxes, fused_scores, fused_labels = wbf_fusion(
                    boxes_list, scores_list, labels_list, iou_threshold=self.config.wbf_iou
                )
            else:
                fused_boxes, fused_scores, fused_labels = [], [], []

            total_detections += len(fused_boxes)

            # 保存结果
            if self.config.save_vis:
                self._save_annotated_image(
                    img, fused_boxes, fused_scores, fused_labels, output_path / "images" / img_path.name
                )

            if self.config.save_txt:
                self._save_labels(
                    img_path.stem, orig_w, orig_h,
                    fused_boxes, fused_scores, fused_labels,
                    output_path / "labels" / f"{img_path.stem}.txt"
                )

            results_list.append({
                "image": str(img_path),
                "detections": len(fused_boxes),
            })

        # 保存报告
        report = {
            "model": self.config.model,
            "images": len(image_files),
            "total_detections": total_detections,
            "avg_detections_per_image": total_detections / len(image_files) if image_files else 0,
            "tta_settings": {
                "scales": self.config.scales,
                "flip": self.config.flip,
                "wbf_iou": self.config.wbf_iou,
            }
        }

        with open(output_path / "tta_report.json", 'w') as f:
            json.dump(report, f, indent=2)

        return report

    def _save_annotated_image(self, img, boxes, scores, labels, output_path):
        """保存标注图片"""
        import random
        colors = {i: (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
                  for i in range(100)}

        for box, score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = map(int, box)
            color = colors.get(int(label), (0, 255, 0))
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            label_text = f"{int(label)} {score:.2f}"
            cv2.putText(img, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.imwrite(str(output_path), img)

    def _save_labels(self, stem, orig_w, orig_h, boxes, scores, labels, output_path):
        """保存 YOLO 格式标签"""
        with open(output_path, 'w') as f:
            for box, score, label in zip(boxes, scores, labels):
                x1, y1, x2, y2 = box
                cx = ((x1 + x2) / 2) / orig_w
                cy = ((y1 + y2) / 2) / orig_h
                bw = (x2 - x1) / orig_w
                bh = (y2 - y1) / orig_h
                if self.config.save_conf:
                    f.write(f"{int(label)} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} {score:.4f}\n")
                else:
                    f.write(f"{int(label)} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")


def main():
    parser = argparse.ArgumentParser(
        description='YOLO TTA 推理工具 - 测试时增强'
    )
    parser.add_argument('--model', type=str, required=True, help='模型路径')
    parser.add_argument('--images', type=str, required=True, help='图片目录或文件')
    parser.add_argument('--output', type=str, default='./tta_results', help='输出目录')
    parser.add_argument('--scales', type=str, default='0.8 1.0 1.2',
                        help='尺度列表 (空格分隔)')
    parser.add_argument('--flip', action='store_true', default=True,
                        help='启用水平翻转')
    parser.add_argument('--no-flip', action='store_true',
                        help='禁用水平翻转')
    parser.add_argument('--conf', type=float, default=0.25, help='置信度阈值')
    parser.add_argument('--iou', type=float, default=0.7, help='NMS IoU 阈值')
    parser.add_argument('--wbf-iou', type=float, default=0.5, help='WBF 融合 IoU 阈值')
    parser.add_argument('--device', type=str, default='cpu', help='推理设备')
    parser.add_argument('--save-txt', action='store_true', default=False,
                        help='保存检测结果为 TXT')
    parser.add_argument('--save-conf', action='store_true', default=True,
                        help='TXT 中包含置信度')
    parser.add_argument('--save-vis', action='store_true', default=True,
                        help='保存标注可视化图片')

    args = parser.parse_args()

    # 配置日志
    import logging as log_module
    log_module.basicConfig(
        level=log_module.INFO,
        format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 解析 scales
    scales = [float(s) for s in args.scales.split()]

    # 处理 flip 参数
    flip = True
    if args.no_flip:
        flip = False

    config = TTAConfig(
        model=args.model,
        images=args.images,
        output_dir=args.output,
        scales=scales,
        flip=flip,
        conf=args.conf,
        iou=args.iou,
        wbf_iou=args.wbf_iou,
        device=args.device,
        save_txt=args.save_txt,
        save_conf=args.save_conf,
        save_vis=args.save_vis,
    )

    inference = TTAInference(config)
    report = inference.run()

    print("\n" + "=" * 60)
    print("TTA 推理完成")
    print("=" * 60)
    print(f"模型: {report['model']}")
    print(f"图片数: {report['images']}")
    print(f"总检测数: {report['total_detections']}")
    print(f"平均检测数/图: {report['avg_detections_per_image']:.2f}")
    print(f"输出目录: {config.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()