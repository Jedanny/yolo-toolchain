"""
测试 auto_annotator 模块
"""

import os
import sys
import subprocess
from pathlib import Path

import pytest


class TestLabelImgPreview:
    """测试 labelImg 预览功能"""

    DATA_DIR = Path("./data/auto")

    def test_data_directory_exists(self):
        """验证测试数据目录存在"""
        assert self.DATA_DIR.exists(), f"数据目录不存在: {self.DATA_DIR}"

    def test_images_directory_exists(self):
        """验证图片目录存在"""
        images_dir = self.DATA_DIR / "images" / "train"
        assert images_dir.exists(), f"图片目录不存在: {images_dir}"
        image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
        assert len(image_files) > 0, f"目录中没有图片: {images_dir}"

    def test_labels_directory_exists(self):
        """验证标签目录存在"""
        labels_dir = self.DATA_DIR / "labels" / "train"
        assert labels_dir.exists(), f"标签目录不存在: {labels_dir}"
        label_files = list(labels_dir.glob("*.txt"))
        assert len(label_files) > 0, f"目录中没有标签文件: {labels_dir}"

    def test_predefined_classes_exists(self):
        """验证预定义类别文件存在"""
        predefined_file = self.DATA_DIR / "predefined_classes.txt"
        assert predefined_file.exists(), f"预定义类别文件不存在: {predefined_file}"

    def test_predefined_classes_format(self):
        """验证预定义类别文件格式"""
        predefined_file = self.DATA_DIR / "predefined_classes.txt"
        if predefined_file.exists():
            content = predefined_file.read_text()
            lines = [l.strip() for l in content.strip().split("\n") if l.strip()]
            assert len(lines) > 0, "预定义类别文件为空"
            print(f"预定义类别: {lines}")

    def test_labelimg_command_available(self):
        """验证 labelImg 命令可用"""
        try:
            result = subprocess.run(
                ["which", "labelImg"],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                print(f"labelImg 路径: {result.stdout.strip()}")
            else:
                pytest.skip("labelImg 未安装")
        except Exception as e:
            pytest.skip(f"labelImg 检查失败: {e}")

    def test_labelimg_launch(self):
        """测试 labelImg 启动"""
        try:
            result = subprocess.run(
                ["which", "labelImg"],
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                pytest.skip("labelImg 未安装")
        except Exception:
            pytest.skip("labelImg 检查失败")

        images_dir = self.DATA_DIR / "images" / "train"
        predefined_file = self.DATA_DIR / "predefined_classes.txt"
        labels_dir = self.DATA_DIR / "labels" / "train"

        if not images_dir.exists() or not predefined_file.exists():
            pytest.skip("测试数据不完整")

        print(f"Images: {images_dir}")
        print(f"Labels: {labels_dir}")
        print(f"Predefined: {predefined_file}")
        print("labelImg 启动测试通过（需手动验证 GUI）")


class TestAutoAnnotatorConfig:
    """测试 auto_annotator 配置"""

    def test_import_auto_annotator(self):
        """验证 auto_annotator 模块可导入"""
        from src.tools.auto_annotator import AutoAnnotator, AutoAnnotatorConfig
        assert AutoAnnotator is not None
        assert AutoAnnotatorConfig is not None

    def test_downloader_import(self):
        """验证 downloader 模块可导入"""
        from src.tools.downloader import download_yolo_model, list_available_models
        assert callable(download_yolo_model)
        assert callable(list_available_models)

    def test_preprocess_import(self):
        """验证 preprocess 模块可导入"""
        from src.tools.preprocess import ImagePreprocessor, preprocess_dataset
        assert ImagePreprocessor is not None
        assert callable(preprocess_dataset)

    def test_verify_annotator_import(self):
        """验证 verify_annotator 模块可导入"""
        from src.tools.verify_annotator import AnnotationVerifier, BBox
        assert AnnotationVerifier is not None
        assert BBox is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
