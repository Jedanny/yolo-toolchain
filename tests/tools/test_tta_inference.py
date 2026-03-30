# tests/tools/test_tta_inference.py
import numpy as np
import pytest
from src.tools.tta_inference import wbf_fusion


class TestWBFFusion:
    """Tests for WBF fusion function."""

    def test_wbf_fusion_empty(self):
        """空输入返回空输出"""
        boxes, scores, labels = wbf_fusion([], [], [], iou_threshold=0.5)
        assert len(boxes) == 0
        assert len(scores) == 0
        assert len(labels) == 0

    def test_wbf_fusion_single_image_single_box(self):
        """单张图单个框直接返回"""
        boxes = [[np.array([0.2, 0.2, 0.4, 0.4])]]
        scores = [[np.array([0.9])]]
        labels = [[np.array([0])]]
        result_boxes, result_scores, result_labels = wbf_fusion(boxes, scores, labels, iou_threshold=0.5)
        assert len(result_boxes) == 1

    def test_wbf_fusion_two_overlapping_boxes(self):
        """两个重叠框应融合"""
        # 两个几乎相同的框
        boxes = [[np.array([0.2, 0.2, 0.4, 0.4]), np.array([0.21, 0.21, 0.41, 0.41])]]
        scores = [[np.array([0.9]), np.array([0.8])]]
        labels = [[np.array([0]), np.array([0])]]
        result_boxes, result_scores, result_labels = wbf_fusion(boxes, scores, labels, iou_threshold=0.5)
        # 应该融合成一个框
        assert len(result_boxes) == 1

    def test_wbf_fusion_different_classes(self):
        """不同类别的框不应融合"""
        boxes = [[np.array([0.2, 0.2, 0.4, 0.4]), np.array([0.25, 0.25, 0.4, 0.4])]]
        scores = [[np.array([0.9]), np.array([0.8])]]
        labels = [[np.array([0]), np.array([1])]]  # 不同类别
        result_boxes, result_scores, result_labels = wbf_fusion(boxes, scores, labels, iou_threshold=0.5)
        # 应该有两个框
        assert len(result_boxes) == 2


class TestTTAConfig:
    """Tests for TTAConfig dataclass."""

    def test_default_config(self):
        """默认配置值正确"""
        from src.tools.tta_inference import TTAConfig
        config = TTAConfig(
            model="yolo11n.pt",
            images="./images"
        )
        assert config.model == "yolo11n.pt"
        assert config.images == "./images"
        assert config.scales == [0.8, 1.0, 1.2]
        assert config.flip is True
        assert config.conf == 0.25
        assert config.wbf_iou == 0.5