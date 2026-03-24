"""
标注核验模块 - 人工核验 AI 自动标注结果
"""

import os
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image
import yaml

logger = logging.getLogger("yolo_toolchain.verify")


def load_classes_from_yaml(yaml_path: str) -> List[str]:
    """从 YOLO 数据集配置文件中加载类别列表"""
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    names = config.get('names', {})
    if isinstance(names, dict):
        classes = [names[i] for i in sorted(names.keys())]
    elif isinstance(names, list):
        classes = names
    else:
        classes = []
    return classes


@dataclass
class BBox:
    """边界框"""
    class_id: int
    class_name: str
    cx: float
    cy: float
    width: float
    height: float
    confidence: float = 1.0

    def to_yolo_line(self) -> str:
        """转换为 YOLO 格式行"""
        return f"{self.class_id} {self.cx:.6f} {self.cy:.6f} {self.width:.6f} {self.height:.6f}"

    @staticmethod
    def from_yolo_line(line: str, class_names: List[str]) -> 'BBox':
        """从 YOLO 格式行解析"""
        parts = line.strip().split()
        if len(parts) < 5:
            raise ValueError(f"Invalid YOLO line: {line}")
        class_id = int(parts[0])
        cx, cy, w, h = map(float, parts[1:5])
        confidence = float(parts[5]) if len(parts) > 5 else 1.0
        class_name = class_names[class_id] if class_id < len(class_names) else "unknown"
        return BBox(class_id, class_name, cx, cy, w, h, confidence)


class AnnotationVerifier:
    """标注核验器"""

    def __init__(self, images_dir: str, labels_dir: str = None, class_names: List[str] = None):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir) if labels_dir else None
        self.class_names = class_names or []
        self.current_idx = 0
        self.image_files = self._get_image_files()

    def _get_image_files(self) -> List[Path]:
        """获取所有图片文件"""
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        files = []
        for ext in extensions:
            files.extend(list(self.images_dir.glob(ext)))
            files.extend(list(self.images_dir.glob(ext.upper())))
        return sorted(files)

    def load_annotations(self, label_path: Path) -> List[BBox]:
        """加载标注"""
        annotations = []
        if not label_path.exists():
            return annotations

        with open(label_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    ann = BBox.from_yolo_line(line, self.class_names)
                    annotations.append(ann)
                except Exception as e:
                    logger.warning(f"Failed to parse line: {line}, error: {e}")
                    continue
        return annotations

    def save_annotations(self, annotations: List[BBox], output_path: Path):
        """保存标注"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            for ann in annotations:
                f.write(ann.to_yolo_line() + '\n')

    def draw_annotations(self, image_path: Path, annotations: List[BBox]) -> np.ndarray:
        """在图片上绘制标注"""
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Failed to read image: {image_path}")

        h, w = img.shape[:2]

        colors = [
            (255, 0, 0),    # 红色
            (0, 255, 0),    # 绿色
            (0, 0, 255),    # 蓝色
            (255, 255, 0),  # 黄色
            (255, 0, 255),  # 紫色
            (0, 255, 255),  # 青色
        ]

        for i, ann in enumerate(annotations):
            color = colors[ann.class_id % len(colors)]

            cx, cy = int(ann.cx * w), int(ann.cy * h)
            bw, bh = int(ann.width * w), int(ann.height * h)

            x1, y1 = cx - bw // 2, cy - bh // 2
            x2, y2 = cx + bw // 2, cy + bh // 2

            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            label = f"{ann.class_name}:{ann.confidence:.2f}"
            if ann.class_id == -1:
                label = "DELETED"

            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (x1, y1 - label_size[1] - 5), (x1 + label_size[0], y1), color, -1)
            cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            idx_label = f"[{i}]"
            cv2.putText(img, idx_label, (x1, y2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        return img

    def get_label_path(self, image_path: Path) -> Path:
        """获取对应的标签文件路径"""
        if self.labels_dir:
            return self.labels_dir / f"{image_path.stem}.txt"
        return image_path.parent / f"{image_path.stem}.txt"

    def verify_interactive(self):
        """交互式核验"""
        if not self.image_files:
            logger.error(f"No images found in {self.images_dir}")
            return

        total = len(self.image_files)
        verified = 0
        skipped = 0

        print(f"\n{'='*60}")
        print(f"标注核验工具")
        print(f"{'='*60}")
        print(f"图片目录: {self.images_dir}")
        print(f"标签目录: {self.labels_dir}")
        print(f"类别列表: {self.class_names}")
        print(f"图片数量: {total}")
        print(f"{'='*60}")
        print("\n操作说明:")
        print("  [0-9] - 选择要编辑的标注")
        print("  [d] - 删除选中的标注")
        print("  [c] - 修改选中标注的类别")
        print("  [s] - 保存并跳到下一张")
        print("  [a] - 接受所有标注（跳过）")
        print("  [r] - 返回上一张")
        print("  [q] - 退出并保存")
        print("  [ESC] - 退出不保存")
        print(f"{'='*60}\n")

        self.current_idx = 0
        annotations = []
        modified = False

        while self.current_idx < total:
            image_path = self.image_files[self.current_idx]
            label_path = self.get_label_path(image_path)

            annotations = self.load_annotations(label_path)

            while True:
                img = self.draw_annotations(image_path, annotations)
                img_resized = cv2.resize(img, (800, 600))

                status = f"[{self.current_idx + 1}/{total}] {image_path.name}"
                if modified:
                    status += " *"
                cv2.rectangle(img_resized, (0, 0), (300, 25), (0, 0, 0), -1)
                cv2.putText(img_resized, status, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                cv2.imshow("Annotation Verifier", img_resized)

                key = cv2.waitKey(0) & 0xFF

                if key == 27 or key == ord('q'):
                    if modified:
                        self.save_annotations(annotations, label_path)
                        logger.info(f"Saved annotations for {image_path.name}")
                    cv2.destroyAllWindows()
                    print("\n已退出")
                    return

                elif key == ord('a'):
                    verified += 1
                    logger.info(f"Accepted all annotations for {image_path.name}")
                    break

                elif key == ord('s'):
                    self.save_annotations(annotations, label_path)
                    modified = False
                    verified += 1
                    logger.info(f"Saved annotations for {image_path.name}")
                    break

                elif key == ord('d'):
                    try:
                        idx_str = input("输入要删除的标注索引: ").strip()
                        if idx_str.isdigit():
                            idx = int(idx_str)
                            if 0 <= idx < len(annotations):
                                removed = annotations.pop(idx)
                                logger.info(f"Deleted annotation {idx}: {removed.class_name}")
                                modified = True
                            else:
                                print(f"索引 {idx} 超出范围")
                        else:
                            print("请输入有效数字")
                    except EOFError:
                        print("输入已取消")

                elif key == ord('c'):
                    try:
                        idx_str = input("输入要修改类别的标注索引: ").strip()
                        if idx_str.isdigit():
                            idx = int(idx_str)
                            if 0 <= idx < len(annotations):
                                print(f"可用类别: {list(enumerate(self.class_names))}")
                                class_str = input("输入新的类别索引: ").strip()
                                if class_str.isdigit():
                                    new_class_id = int(class_str)
                                    if 0 <= new_class_id < len(self.class_names):
                                        annotations[idx].class_id = new_class_id
                                        annotations[idx].class_name = self.class_names[new_class_id]
                                        logger.info(f"Changed annotation {idx} class to {self.class_names[new_class_id]}")
                                        modified = True
                                    else:
                                        print(f"类别索引 {new_class_id} 超出范围")
                                else:
                                    print("请输入有效数字")
                            else:
                                print(f"索引 {idx} 超出范围")
                        else:
                            print("请输入有效数字")
                    except EOFError:
                        print("输入已取消")

                elif key == ord('r'):
                    if self.current_idx > 0:
                        self.current_idx -= 1
                        skipped += 1
                    else:
                        print("已是第一张图片")
                    break

                elif key == ord('n'):
                    break

                elif key >= ord('0') and key <= ord('9'):
                    idx = key - ord('0')
                    if idx < len(annotations):
                        print(f"选中标注 {idx}: {annotations[idx].class_name} @ ({annotations[idx].cx:.3f}, {annotations[idx].cy:.3f})")
                    else:
                        print(f"标注数量只有 {len(annotations)} 个")

            self.current_idx += 1

        cv2.destroyAllWindows()
        print(f"\n核验完成! 已核验 {verified} 张图片, 跳过 {skipped} 张")

    def verify_auto(self, output_dir: str = None, accept_threshold: float = 0.9):
        """自动核验 - 只保留高置信度标注

        Args:
            output_dir: 输出目录（不指定则原地修改）
            accept_threshold: 接受的置信度阈值
        """
        if not self.image_files:
            logger.error(f"No images found in {self.images_dir}")
            return

        output_dir = Path(output_dir) if output_dir else self.labels_dir
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)

        total = len(self.image_files)
        accepted = 0
        filtered = 0

        for i, image_path in enumerate(self.image_files):
            label_path = self.get_label_path(image_path)
            annotations = self.load_annotations(label_path)

            original_count = len(annotations)
            filtered_annotations = [ann for ann in annotations if ann.confidence >= accept_threshold]

            if filtered_annotations:
                accepted += 1
            else:
                filtered += 1

            out_label_path = output_dir / label_path.name if output_dir else label_path
            self.save_annotations(filtered_annotations, out_label_path)

            logger.info(f"[{i+1}/{total}] {image_path.name}: {len(filtered_annotations)}/{original_count} kept")

        logger.info(f"Done! {accepted} images with high-conf annotations, {filtered} images filtered out")


def main():
    parser = argparse.ArgumentParser(description='标注核验工具 - 人工核验 AI 自动标注结果')
    parser.add_argument('--images', type=str, required=True, help='图片目录')
    parser.add_argument('--labels', type=str, help='标签目录（默认与图片同目录）')
    parser.add_argument('--classes', type=str, nargs='+', help='类别列表（优先级：--classes > --dataset）')
    parser.add_argument('--dataset', type=str, help='YOLO 数据集配置文件路径（自动加载类别）')
    parser.add_argument('--mode', type=str, default='interactive',
                        choices=['interactive', 'auto'],
                        help='核验模式：interactive（交互）或 auto（自动过滤）')
    parser.add_argument('--output', type=str, help='输出目录（auto 模式）')
    parser.add_argument('--conf', type=float, default=0.5,
                        help='置信度阈值（auto 模式）')
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])

    args = parser.parse_args()

    import logging as log_module
    log_level = getattr(log_module, args.log_level.upper(), log_module.INFO)
    log_module.basicConfig(
        level=log_level,
        format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    if args.classes:
        class_names = args.classes
    elif args.dataset:
        class_names = load_classes_from_yaml(args.dataset)
        logger.info(f"Loaded {len(class_names)} classes from {args.dataset}: {class_names}")
    else:
        class_names = []
        logger.warning("No classes specified, class names will show as 'unknown'")

    verifier = AnnotationVerifier(
        images_dir=args.images,
        labels_dir=args.labels,
        class_names=class_names
    )

    if args.mode == 'interactive':
        verifier.verify_interactive()
    else:
        verifier.verify_auto(output_dir=args.output, accept_threshold=args.conf)


if __name__ == '__main__':
    main()
