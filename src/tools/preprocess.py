"""
图片批量预处理模块
"""

import os
import argparse
import logging
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass
from PIL import Image, ImageEnhance, ImageFilter
import cv2
import numpy as np

logger = logging.getLogger("yolo_toolchain.preprocess")


@dataclass
class PreprocessConfig:
    """预处理配置"""
    resize: Tuple[int, int] = None
    normalize: bool = False
    enhance: bool = False
    brightness: float = 1.0
    contrast: float = 1.0
    sharpness: float = 1.0
    denoise: bool = False
    grayscale: bool = False
    format: str = None
    quality: int = 95


class ImagePreprocessor:
    """图片预处理器"""

    def __init__(self, config: PreprocessConfig = None):
        self.config = config or PreprocessConfig()

    def process_image(self, image_path: str, output_path: str = None) -> str:
        """处理单张图片

        Args:
            image_path: 输入图片路径
            output_path: 输出路径（默认覆盖原图）

        Returns:
            输出路径
        """
        if output_path is None:
            output_path = image_path

        img = Image.open(image_path)
        original_mode = img.mode

        if self.config.grayscale and img.mode != 'L':
            img = img.convert('L')
            logger.debug(f"Converted to grayscale: {image_path}")

        if self.config.resize:
            img = self.resize_image(img, self.config.resize)
            logger.debug(f"Resized to {self.config.resize}: {image_path}")

        if self.config.enhance:
            img = self.enhance_image(img)

        if self.config.denoise:
            img = self.denoise_image(img)

        if self.config.normalize and img.mode == 'L':
            img = self.normalize_image(img)

        if self.config.format:
            output_path = self._change_format(output_path, self.config.format)

        output_path = self._ensure_dir(output_path)
        img.save(output_path, quality=self.config.quality)
        logger.info(f"Saved: {output_path}")

        return output_path

    def resize_image(self, img: Image.Image, size: Tuple[int, int]) -> Image.Image:
        """调整图片尺寸"""
        return img.resize(size, Image.LANCZOS)

    def enhance_image(self, img: Image.Image) -> Image.Image:
        """增强图片"""
        if self.config.brightness != 1.0:
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(self.config.brightness)

        if self.config.contrast != 1.0:
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(self.config.contrast)

        if self.config.sharpness != 1.0:
            enhancer = ImageEnhance.Sharpness(img)
            img = enhancer.enhance(self.config.sharpness)

        return img

    def denoise_image(self, img: Image.Image) -> Image.Image:
        """降噪"""
        img_array = np.array(img)
        if len(img_array.shape) == 3:
            img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            img_cv = cv2.fastNlMeansDenoisingColored(img_cv, None, 10, 10, 7, 21)
            img_array = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
            return Image.fromarray(img_array)
        else:
            img_cv = cv2.fastNlMeansDenoising(img_array, None, 10, 7, 21)
            return Image.fromarray(img_cv)

    def normalize_image(self, img: Image.Image) -> Image.Image:
        """归一化"""
        img_array = np.array(img, dtype=np.float32)
        img_array = img_array / 255.0
        img_array = (img_array - 0.5) / 0.5
        img_array = ((img_array + 1) * 127.5).astype(np.uint8)
        return Image.fromarray(img_array)

    def _change_format(self, path: str, format: str) -> str:
        """更改图片格式"""
        p = Path(path)
        return str(p.parent / f"{p.stem}.{format.lower()}")

    def _ensure_dir(self, path: str) -> str:
        """确保目录存在"""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        return path

    def process_batch(self, input_dir: str, output_dir: str = None) -> List[Tuple[str, str]]:
        """批量处理图片

        Args:
            input_dir: 输入目录
            output_dir: 输出目录（默认覆盖原目录）

        Returns:
            [(输入路径, 输出路径), ...]
        """
        input_dir = Path(input_dir)
        if output_dir is None:
            output_dir = input_dir
        else:
            output_dir = Path(output_dir)

        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
        image_files = []
        for ext in extensions:
            image_files.extend(list(input_dir.glob(ext)))
            image_files.extend(list(input_dir.glob(ext.upper())))

        image_files = sorted(set(image_files))

        if not image_files:
            logger.warning(f"No images found in {input_dir}")
            return []

        results = []
        for i, img_path in enumerate(image_files):
            if output_dir == input_dir:
                out_path = str(img_path)
            else:
                rel_path = img_path.relative_to(input_dir)
                out_path = str(output_dir / rel_path)

            try:
                self.process_image(str(img_path), out_path)
                results.append((str(img_path), out_path))
            except Exception as e:
                logger.error(f"Failed to process {img_path}: {e}")
                results.append((str(img_path), None))

            if (i + 1) % 10 == 0:
                logger.info(f"Progress: {i + 1}/{len(image_files)}")

        return results


def preprocess_dataset(
    input_dir: str,
    output_dir: str = None,
    resize: Tuple[int, int] = None,
    enhance: bool = False,
    brightness: float = 1.0,
    contrast: float = 1.0,
    denoise: bool = False,
    grayscale: bool = False,
    format: str = None,
    quality: int = 95
):
    """预处理数据集图片

    Args:
        input_dir: 输入目录
        output_dir: 输出目录
        resize: 调整尺寸 (width, height)
        enhance: 是否增强
        brightness: 亮度调整
        contrast: 对比度调整
        denoise: 是否降噪
        grayscale: 是否转灰度
        format: 输出格式
        quality: JPEG 质量
    """
    config = PreprocessConfig(
        resize=resize,
        enhance=enhance,
        brightness=brightness,
        contrast=contrast,
        denoise=denoise,
        grayscale=grayscale,
        format=format,
        quality=quality
    )

    preprocessor = ImagePreprocessor(config)
    results = preprocessor.process_batch(input_dir, output_dir)

    success = sum(1 for _, out in results if out is not None)
    logger.info(f"Done! {success}/{len(results)} images processed")


def main():
    parser = argparse.ArgumentParser(description='图片批量预处理工具')
    parser.add_argument('--input', type=str, required=True, help='输入目录')
    parser.add_argument('--output', type=str, help='输出目录（默认覆盖原目录）')
    parser.add_argument('--resize', type=int, nargs=2, metavar=('W', 'H'),
                        help='调整尺寸，如: --resize 640 480')
    parser.add_argument('--enhance', action='store_true', help='增强图片')
    parser.add_argument('--brightness', type=float, default=1.0,
                        help='亮度调整（1.0=不变）')
    parser.add_argument('--contrast', type=float, default=1.0,
                        help='对比度调整（1.0=不变）')
    parser.add_argument('--denoise', action='store_true', help='降噪')
    parser.add_argument('--grayscale', action='store_true', help='转灰度图')
    parser.add_argument('--format', type=str, choices=['jpg', 'png', 'bmp'],
                        help='输出格式')
    parser.add_argument('--quality', type=int, default=95, help='JPEG 质量')
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

    resize = tuple(args.resize) if args.resize else None

    preprocess_dataset(
        input_dir=args.input,
        output_dir=args.output,
        resize=resize,
        enhance=args.enhance,
        brightness=args.brightness,
        contrast=args.contrast,
        denoise=args.denoise,
        grayscale=args.grayscale,
        format=args.format,
        quality=args.quality
    )


if __name__ == '__main__':
    main()
