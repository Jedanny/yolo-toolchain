import pytest
from dataclasses import dataclass
from src.tools.best_model_selector import BestModelSelectorConfig


class TestBestModelSelectorConfig:
    """Tests for BestModelSelectorConfig dataclass."""

    def test_default_config(self):
        """默认配置值正确"""
        config = BestModelSelectorConfig(
            model="yolo11n.pt",
            data="dataset.yaml"
        )
        assert config.model == "yolo11n.pt"
        assert config.data == "dataset.yaml"
        assert config.metric == "fitness"
        assert config.output is None
        assert config.device == "0"

    def test_custom_config(self):
        """自定义配置值正确"""
        config = BestModelSelectorConfig(
            model="runs/train/exp/weights",
            data="dataset.yaml",
            metric="mAP50",
            output="selected.txt",
            device="mps"
        )
        assert config.metric == "mAP50"
        assert config.output == "selected.txt"
        assert config.device == "mps"


class TestMetricMap:
    """Tests for METRIC_MAP metric extraction."""

    def test_metric_map_keys(self):
        """METRIC_MAP 包含所有支持的指标"""
        from src.tools.best_model_selector import BestModelSelector
        expected_keys = {"mAP50", "mAP50-95", "recall", "precision", "fitness"}
        assert set(BestModelSelector.METRIC_MAP.keys()) == expected_keys