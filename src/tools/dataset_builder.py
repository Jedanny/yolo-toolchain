"""
数据准备模块 - 数据集构建工具
支持从多种格式构建YOLO数据集
"""

import os
import json
import shutil
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import xml.etree.ElementTree as ET
from tqdm import tqdm


class DatasetBuilder:
    """数据集构建器"""

    SUPPORTED_FORMATS = ['yolo', 'coco', 'voc', 'labelme', 'cvat']

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.images_dir = self.output_dir / 'images'
        self.labels_dir = self.output_dir / 'labels'

        # 创建目录结构
        for split in ['train', 'val', 'test']:
            (self.images_dir / split).mkdir(parents=True, exist_ok=True)
            (self.labels_dir / split).mkdir(parents=True, exist_ok=True)

        self.class_names = []
        self.class_mapping = {}

    def add_classes(self, classes: List[str]):
        """设置类别名称"""
        self.class_names = classes
        self.class_mapping = {name: idx for idx, name in enumerate(classes)}

    def convert_voc_to_yolo(self, voc_dir: str, output_dir: str, split: str = 'train') -> int:
        """将VOC格式转换为YOLO格式"""
        voc_dir = Path(voc_dir)
        annotations_dir = voc_dir / 'Annotations'
        images_dir = voc_dir / 'JPEGImages'

        if not annotations_dir.exists():
            raise FileNotFoundError(f"Annotations directory not found: {annotations_dir}")

        if not images_dir.exists():
            raise FileNotFoundError(f"JPEGImages directory not found: {images_dir}")

        converted_count = 0

        for xml_file in tqdm(list(annotations_dir.glob('*.xml')), desc=f"Converting VOC to YOLO ({split})"):
            try:
                # 解析XML
                tree = ET.parse(xml_file)
                root = tree.getroot()

                # 获取图像尺寸
                size = root.find('size')
                img_width = int(size.find('width').text)
                img_height = int(size.find('height').text)

                # 获取类别映射（如果还没有）
                if not self.class_mapping:
                    self._extract_voc_classes(root)

                # 解析边界框
                yolo_labels = []
                for obj in root.findall('object'):
                    class_name = obj.find('name').text

                    if class_name not in self.class_mapping:
                        continue

                    class_id = self.class_mapping[class_name]

                    bbox = obj.find('bndbox')
                    xmin = float(bbox.find('xmin').text)
                    ymin = float(bbox.find('ymin').text)
                    xmax = float(bbox.find('xmax').text)
                    ymax = float(bbox.find('ymax').text)

                    # 转换为YOLO格式 (center_x, center_y, width, height) 归一化
                    x_center = ((xmin + xmax) / 2) / img_width
                    y_center = ((ymin + ymax) / 2) / img_height
                    width = (xmax - xmin) / img_width
                    height = (ymax - ymin) / img_height

                    yolo_labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

                # 如果有有效标注
                if yolo_labels:
                    # 复制图像
                    img_path = images_dir / f"{xml_file.stem}.jpg"
                    if img_path.exists():
                        shutil.copy(img_path, self.images_dir / split / img_path.name)
                    else:
                        img_path = images_dir / f"{xml_file.stem}.png"
                        if img_path.exists():
                            shutil.copy(img_path, self.images_dir / split / img_path.name)

                    # 保存标签
                    label_path = self.labels_dir / split / f"{xml_file.stem}.txt"
                    with open(label_path, 'w') as f:
                        f.write('\n'.join(yolo_labels))

                    converted_count += 1

            except Exception as e:
                print(f"Error processing {xml_file}: {e}")
                continue

        return converted_count

    def _extract_voc_classes(self, root):
        """从VOC XML中提取所有类别"""
        classes = set()
        for obj in root.findall('object'):
            classes.add(obj.find('name').text)
        self.class_names = sorted(list(classes))
        self.class_mapping = {name: idx for idx, name in enumerate(self.class_names)}

    def convert_coco_to_yolo(self, coco_json: str, images_dir: str, output_dir: str, split: str = 'train') -> int:
        """将COCO格式转换为YOLO格式"""
        with open(coco_json, 'r') as f:
            coco_data = json.load(f)

        images_dir = Path(images_dir)
        converted_count = 0

        # 构建图像ID到文件名的映射
        img_id_to_info = {img['id']: img for img in coco_data['images']}

        # 构建类别映射
        if not self.class_mapping:
            for cat in coco_data['categories']:
                self.class_names.append(cat['name'])
            self.class_mapping = {cat['name']: cat['id'] - 1 for cat in coco_data['categories']}

        # 按图像分组标注
        ann_by_image = {}
        for ann in coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in ann_by_image:
                ann_by_image[img_id] = []
            ann_by_image[img_id].append(ann)

        # 处理每个图像
        for img_id, annotations in tqdm(ann_by_image.items(), desc=f"Converting COCO to YOLO ({split})"):
            img_info = img_id_to_info[img_id]
            img_path = images_dir / img_info['file_name']

            if not img_path.exists():
                continue

            img_width = img_info['width']
            img_height = img_info['height']

            yolo_labels = []
            for ann in annotations:
                class_id = ann['category_id'] - 1  # COCO类别ID从1开始

                # COCO bbox格式: [x_min, y_min, width, height]
                bbox = ann['bbox']
                x_center = (bbox[0] + bbox[2] / 2) / img_width
                y_center = (bbox[1] + bbox[3] / 2) / img_height
                width = bbox[2] / img_width
                height = bbox[3] / img_height

                yolo_labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

            # 复制图像
            shutil.copy(img_path, self.images_dir / split / img_path.name)

            # 保存标签
            label_path = self.labels_dir / split / f"{img_path.stem}.txt"
            with open(label_path, 'w') as f:
                f.write('\n'.join(yolo_labels))

            converted_count += 1

        return converted_count

    def create_dataset_yaml(self, output_path: str, train_path: str = None, val_path: str = None, test_path: str = None):
        """创建YOLO数据集配置文件"""
        yaml_content = {
            'path': str(self.output_dir.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'names': {idx: name for idx, name in enumerate(self.class_names)}
        }

        if train_path:
            yaml_content['train'] = train_path
        if val_path:
            yaml_content['val'] = val_path
        if test_path:
            yaml_content['test'] = test_path

        with open(output_path, 'w') as f:
            for key, value in yaml_content.items():
                if isinstance(value, dict):
                    f.write(f"{key}:\n")
                    for k, v in value.items():
                        f.write(f"  {k}: {v}\n")
                else:
                    f.write(f"{key}: {value}\n")

        return yaml_content

    def analyze_dataset(self) -> Dict:
        """分析数据集统计信息"""
        stats = {
            'total_images': 0,
            'total_labels': 0,
            'class_distribution': {name: 0 for name in self.class_names},
            'box_stats': {
                'avg_width': 0,
                'avg_height': 0,
                'avg_area': 0,
                'min_width': float('inf'),
                'max_width': 0,
                'min_height': float('inf'),
                'max_height': 0
            },
            'split_distribution': {}
        }

        total_boxes = 0

        for split in ['train', 'val', 'test']:
            split_dir = self.labels_dir / split
            if not split_dir.exists():
                continue

            split_count = 0
            for label_file in split_dir.glob('*.txt'):
                split_count += 1
                with open(label_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            width = float(parts[3])
                            height = float(parts[4])
                            area = width * height

                            if class_id < len(self.class_names):
                                stats['class_distribution'][self.class_names[class_id]] += 1

                            stats['box_stats']['avg_width'] += width
                            stats['box_stats']['avg_height'] += height
                            stats['box_stats']['avg_area'] += area
                            stats['box_stats']['min_width'] = min(stats['box_stats']['min_width'], width)
                            stats['box_stats']['max_width'] = max(stats['box_stats']['max_width'], width)
                            stats['box_stats']['min_height'] = min(stats['box_stats']['min_height'], height)
                            stats['box_stats']['max_height'] = max(stats['box_stats']['max_height'], height)

                            total_boxes += 1

            stats['split_distribution'][split] = split_count
            stats['total_images'] += split_count

        stats['total_labels'] = total_boxes

        if total_boxes > 0:
            stats['box_stats']['avg_width'] /= total_boxes
            stats['box_stats']['avg_height'] /= total_boxes
            stats['box_stats']['avg_area'] /= total_boxes

        # 检测类别不平衡
        max_count = max(stats['class_distribution'].values()) if stats['class_distribution'] else 1
        min_count = min([c for c in stats['class_distribution'].values() if c > 0]) if any(c > 0 for c in stats['class_distribution'].values()) else 1
        stats['imbalance_ratio'] = max_count / min_count if min_count > 0 else float('inf')

        return stats


def main():
    parser = argparse.ArgumentParser(description='YOLO数据集构建工具')
    parser.add_argument('--mode', type=str, required=True, choices=['voc', 'coco'],
                        help='输入数据格式')
    parser.add_argument('--input', type=str, required=True,
                        help='输入目录或文件路径')
    parser.add_argument('--output', type=str, required=True,
                        help='输出目录')
    parser.add_argument('--split', type=str, default='train',
                        help='数据集划分 (train/val/test)')
    parser.add_argument('--images_dir', type=str, default=None,
                        help='图像目录 (COCO模式需要)')
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

    builder = DatasetBuilder(args.output)

    if args.mode == 'voc':
        count = builder.convert_voc_to_yolo(args.input, args.output, args.split)
    elif args.mode == 'coco':
        count = builder.convert_coco_to_yolo(args.input, args.images_dir, args.output, args.split)

    print(f"\n转换完成: {count} 张图像")

    # 生成dataset.yaml
    yaml_path = Path(args.output) / 'dataset.yaml'
    builder.create_dataset_yaml(str(yaml_path))
    print(f"数据集配置已保存: {yaml_path}")

    # 分析数据集
    stats = builder.analyze_dataset()
    print(f"\n数据集统计:")
    print(f"  总图像数: {stats['total_images']}")
    print(f"  总标注数: {stats['total_labels']}")
    print(f"  类别不平衡比例: {stats['imbalance_ratio']:.2f}")


if __name__ == '__main__':
    main()
