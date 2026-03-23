"""
数据增强模块 - 提供YOLO数据增强功能
包括YOLO内置增强和Albumentations扩展增强
"""

import os
import random
import argparse
import shutil
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import cv2
import numpy as np
from PIL import Image


class YOLOAugmentor:
    """YOLO风格数据增强器"""

    def __init__(self, config: Dict = None):
        self.config = config or self._default_config()

    def _default_config(self) -> Dict:
        """默认增强配置"""
        return {
            'degrees': 0.0,
            'translate': 0.1,
            'scale': 0.5,
            'shear': 0.0,
            'perspective': 0.0,
            'flipud': 0.0,
            'fliplr': 0.5,
            'hsv_h': 0.015,
            'hsv_s': 0.7,
            'hsv_v': 0.4,
            'mosaic': 1.0,
            'mixup': 0.0,
            'cutmix': 0.0,
        }

    def load_config(self, config_path: str):
        """从YAML文件加载配置"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

    def apply_hsv(self, image: np.ndarray) -> np.ndarray:
        """应用HSV颜色空间增强"""
        h_gain = random.uniform(-self.config['hsv_h'], self.config['hsv_h'])
        s_gain = random.uniform(1 - self.config['hsv_s'], 1 + self.config['hsv_s'])
        v_gain = random.uniform(1 - self.config['hsv_v'], 1 + self.config['hsv_v'])

        r = np.random.uniform(-1, 1, 3) * [h_gain, s_gain, v_gain] + 1
        hue, sat, val = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))

        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype('uint8')
        lut_val = (np.clip(x * r[2], 0, 255)).astype('uint8')
        lut_sat = (np.clip(x * r[1], 0, 255)).astype('uint8')

        img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        return cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)

    def apply_flip(self, image: np.ndarray, bboxes: List) -> Tuple[np.ndarray, List]:
        """应用水平翻转"""
        if random.random() < self.config['fliplr']:
            image = np.fliplr(image)
            for bbox in bboxes:
                bbox[0] = 1 - bbox[0]  # 翻转center_x
        return image, bboxes

    def apply_scale(self, image: np.ndarray, bboxes: List) -> Tuple[np.ndarray, List]:
        """应用尺度变换"""
        scale_factor = random.uniform(1 - self.config['scale'], 1 + self.config['scale'])
        h, w = image.shape[:2]
        new_h, new_w = int(h * scale_factor), int(w * scale_factor)

        scaled_image = cv2.resize(image, (new_w, new_h))

        # 填充或裁剪到原始尺寸
        if scale_factor > 1:
            # 裁剪
            start_h = (new_h - h) // 2
            start_w = (new_w - w) // 2
            image = scaled_image[start_h:start_h + h, start_w:start_w + w]
        else:
            # 填充
            pad_h = (h - new_h) // 2
            pad_w = (w - new_w) // 2
            padded = np.full((h, w, 3), 114, dtype=np.uint8)
            padded[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = scaled_image
            image = padded

        return image, bboxes

    def apply_translate(self, image: np.ndarray, bboxes: List) -> Tuple[np.ndarray, List]:
        """应用平移变换"""
        translate_factor = random.uniform(-self.config['translate'], self.config['translate'])
        h, w = image.shape[:2]
        dx = int(w * translate_factor)
        dy = int(h * translate_factor)

        M = np.float32([[1, 0, -dx], [0, 1, -dy]])
        image = cv2.warpAffine(image, M, (w, h))

        # 平移边界框
        for bbox in bboxes:
            bbox[0] += translate_factor  # center_x

        return image, bboxes

    def apply_rotation(self, image: np.ndarray, bboxes: List) -> Tuple[np.ndarray, List]:
        """应用旋转变换（简单版本，仅支持小角度）"""
        angle = random.uniform(-self.config['degrees'], self.config['degrees'])
        h, w = image.shape[:2]
        center = (w // 2, h // 2)

        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        image = cv2.warpAffine(image, M, (w, h))

        # 注意：完整旋转需要对边界框进行更复杂的变换
        # 这里简化处理
        return image, bboxes

    def augment_image(self, image: np.ndarray, bboxes: List) -> Tuple[np.ndarray, List]:
        """对单张图像应用所有增强"""
        # HSV增强
        image = self.apply_hsv(image)

        # 几何变换
        if random.random() < 0.5:
            image, bboxes = self.apply_flip(image, bboxes)
        if random.random() < 0.5:
            image, bboxes = self.apply_scale(image, bboxes)
        if random.random() < 0.5:
            image, bboxes = self.apply_translate(image, bboxes)
        if random.random() < 0.5:
            image, bboxes = self.apply_rotation(image, bboxes)

        return image, bboxes

    def mosaic(self, images: List[np.ndarray], bboxes_list: List[List]) -> Tuple[np.ndarray, List]:
        """Mosaic增强：将4张图像拼接为1张"""
        if len(images) != 4:
            raise ValueError("Mosaic需要4张图像")

        # 获取图像尺寸
        h, w = images[0].shape[:2]

        # 创建输出图像
        mosaic_img = np.zeros((h * 2, w * 2, 3), dtype=np.uint8)

        # 定义4个位置的坐标
        positions = [
            (0, 0, 0, w, 0, h),      # 左上
            (0, w, w, w * 2, 0, h),   # 右上
            (h, 0, 0, w, h, h * 2),   # 左下
            (h, w, w, w * 2, h, h * 2)  # 右下
        ]

        all_bboxes = []

        for i, (img, bboxes) in enumerate(zip(images, bboxes_list)):
            y1, x1, x2_start, x2_end, y2_start, y2_end = positions[i]
            resized = cv2.resize(img, (w, h))
            mosaic_img[y1:y2_end, x1:x2_end] = resized

            # 调整边界框坐标
            for bbox in bboxes:
                new_bbox = bbox.copy()
                if i == 0:  # 左上
                    pass  # 不变
                elif i == 1:  # 右上
                    new_bbox[0] = 0.5 + bbox[0] / 2
                elif i == 2:  # 左下
                    new_bbox[1] = 0.5 + bbox[1] / 2
                elif i == 3:  # 右下
                    new_bbox[0] = 0.5 + bbox[0] / 2
                    new_bbox[1] = 0.5 + bbox[1] / 2
                all_bboxes.append(new_bbox)

        return mosaic_img, all_bboxes

    def mixup(self, img1: np.ndarray, bboxes1: List, img2: np.ndarray, bboxes2: List, alpha: float = 0.5) -> Tuple[np.ndarray, List]:
        """Mixup增强：混合两张图像"""
        mixed_img = cv2.addWeighted(img1, alpha, img2, 1 - alpha, 0)
        all_bboxes = bboxes1 + bboxes2
        return mixed_img, all_bboxes

    def augment_dataset(self, input_dir: str, output_dir: str, num_augment: int = 5):
        """对整个数据集进行增强"""
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)

        # 创建输出目录
        (output_dir / 'images').mkdir(parents=True, exist_ok=True)
        (output_dir / 'labels').mkdir(parents=True, exist_ok=True)

        # 获取所有图像
        image_files = list(input_dir.glob('**/*.jpg')) + list(input_dir.glob('**/*.png'))
        image_files.extend(input_dir.glob('**/*.jpeg'))

        for img_path in image_files:
            # 读取图像和标签
            image = cv2.imread(str(img_path))
            label_path = input_dir / 'labels' / f"{img_path.stem}.txt"

            bboxes = []
            if label_path.exists():
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            bboxes.append([float(x) for x in parts])

            # 保存原始图像和标签
            shutil.copy(img_path, output_dir / 'images' / img_path.name)
            if bboxes:
                with open(output_dir / 'labels' / f"{img_path.stem}.txt", 'w') as f:
                    for bbox in bboxes:
                        f.write(' '.join(map(str, bbox)) + '\n')

            # 生成增强图像
            for i in range(num_augment):
                aug_image, aug_bboxes = self.augment_image(image.copy(), [b.copy() for b in bboxes])

                # 保存增强图像
                aug_name = f"{img_path.stem}_aug{i}{img_path.suffix}"
                cv2.imwrite(str(output_dir / 'images' / aug_name), aug_image)

                # 保存增强标签
                if aug_bboxes:
                    with open(output_dir / 'labels' / f"{img_path.stem}_aug{i}.txt", 'w') as f:
                        for bbox in aug_bboxes:
                            # 确保边界框在有效范围内
                            bbox[0] = max(0, min(1, bbox[0]))
                            bbox[1] = max(0, min(1, bbox[1]))
                            bbox[2] = max(0, min(1, bbox[2]))
                            bbox[3] = max(0, min(1, bbox[3]))
                            f.write(' '.join(map(str, bbox)) + '\n')


class AlbumentationsAugmentor:
    """基于Albumentations的增强器"""

    def __init__(self, config: Dict = None):
        try:
            import albumentations as A
            self.A = A
            self.available = True
        except ImportError:
            print("Warning: albumentations not installed. Using basic augmentation only.")
            self.available = False

        self.config = config or self._default_config()
        self.transform = None
        self._build_transform()

    def _default_config(self) -> Dict:
        return {
            'HorizontalFlip': {'p': 0.5},
            'RandomBrightnessContrast': {'p': 0.5},
            'ShiftScaleRotate': {'p': 0.5, 'shift_limit': 0.0625, 'scale_limit': 0.1, 'rotate_limit': 15},
            'HueSaturationValue': {'p': 0.5},
            'CLAHE': {'p': 0.3},
            'Blur': {'p': 0.2, 'blur_limit': 3},
            'GaussNoise': {'p': 0.2},
        }

    def _build_transform(self):
        """构建Albumentations变换管道"""
        if not self.available:
            return

        transforms = []
        for name, params in self.config.items():
            if hasattr(self.A, name):
                p = params.get('p', 0.5)
                args = {k: v for k, v in params.items() if k != 'p'}
                transforms.append(getattr(self.A, name)(p=p, **args))

        self.transform = self.A.Compose(
            transforms,
            bbox_params=self.A.BboxParams(format='yolo', label_fields=['class_labels'])
        )

    def augment(self, image: np.ndarray, bboxes: List, class_labels: List) -> Tuple[np.ndarray, List, List]:
        """应用Albumentations增强"""
        if not self.available or self.transform is None:
            return image, bboxes, class_labels

        transformed = self.transform(image=image, bboxes=bboxes, class_labels=class_labels)
        return transformed['image'], transformed['bboxes'], transformed['class_labels']


def main():
    parser = argparse.ArgumentParser(description='YOLO数据增强工具')
    parser.add_argument('--input', type=str, required=True, help='输入数据目录')
    parser.add_argument('--output', type=str, required=True, help='输出数据目录')
    parser.add_argument('--num_augment', type=int, default=5, help='每张图像生成的增强数量')
    parser.add_argument('--config', type=str, default=None, help='增强配置文件')
    parser.add_argument('--use_albumentations', action='store_true', help='使用Albumentations增强')

    args = parser.parse_args()

    augmentor = YOLOAugmentor()

    if args.config:
        augmentor.load_config(args.config)

    print(f"开始增强数据: {args.input} -> {args.output}")
    augmentor.augment_dataset(args.input, args.output, args.num_augment)
    print("增强完成!")


if __name__ == '__main__':
    main()
