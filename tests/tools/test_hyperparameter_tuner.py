"""
Unit tests for Hyperparameter Tuner Module
"""

import pytest
from dataclasses import is_dataclass

from src.tools.hyperparameter_tuner import TunerConfig, HyperparameterTuner


class TestTunerConfigDefaults:
    """Test TunerConfig with default values."""

    def test_TunerConfig_defaults(self):
        """Test TunerConfig initializes with correct defaults."""
        config = TunerConfig()

        assert config.model == "yolo11n.pt"
        assert config.data == "data.yaml"
        assert config.epochs == 100
        assert config.iterations == 10
        assert config.output_dir == "runs/tune"
        assert config.storage == "sqlite:///:memory:"
        assert config.resume is False
        assert config.space is None
        assert config.patience == 50
        assert config.device == ""
        assert config.tune_kwargs == {}

    def test_TunerConfig_is_dataclass(self):
        """Test that TunerConfig is a dataclass."""
        assert is_dataclass(TunerConfig)


class TestTunerConfigCustom:
    """Test TunerConfig with custom values."""

    def test_TunerConfig_custom(self):
        """Test TunerConfig initializes with custom values."""
        config = TunerConfig(
            model="yolo11s.pt",
            data="custom_data.yaml",
            epochs=50,
            iterations=5,
            output_dir="runs/custom_tune",
            storage="sqlite:///tuning.db",
            resume=True,
            space={"lr0": (0.01, 0.1)},
            patience=30,
            device="0",
            tune_kwargs={"batch": 16},
        )

        assert config.model == "yolo11s.pt"
        assert config.data == "custom_data.yaml"
        assert config.epochs == 50
        assert config.iterations == 5
        assert config.output_dir == "runs/custom_tune"
        assert config.storage == "sqlite:///tuning.db"
        assert config.resume is True
        assert config.space == {"lr0": (0.01, 0.1)}
        assert config.patience == 30
        assert config.device == "0"
        assert config.tune_kwargs == {"batch": 16}


class TestTunerConfigToDict:
    """Test TunerConfig.to_dict() method."""

    def test_TunerConfig_to_dict(self):
        """Test TunerConfig.to_dict() returns correct dictionary."""
        config = TunerConfig(
            model="yolo11n.pt",
            data="test.yaml",
            epochs=100,
            iterations=10,
            output_dir="runs/tune",
            storage="sqlite:///:memory:",
            resume=False,
            patience=50,
            device="",
        )

        result = config.to_dict()

        assert isinstance(result, dict)
        assert result["data"] == "test.yaml"
        assert result["epochs"] == 100
        assert result["iterations"] == 10
        assert result["storage"] == "sqlite:///:memory:"
        assert result["resume"] is False
        assert result["patience"] == 50
        assert "device" not in result  # Empty string should be filtered out
        assert "model" not in result  # Model is not in to_dict output

    def test_TunerConfig_to_dict_with_tune_kwargs(self):
        """Test TunerConfig.to_dict() merges tune_kwargs."""
        config = TunerConfig(
            data="test.yaml",
            epochs=100,
            tune_kwargs={"batch": 32, "workers": 8},
        )

        result = config.to_dict()

        assert result["batch"] == 32
        assert result["workers"] == 8
        assert result["data"] == "test.yaml"
        assert result["epochs"] == 100

    def test_TunerConfig_to_dict_filters_none(self):
        """Test TunerConfig.to_dict() filters out None values."""
        config = TunerConfig(
            data="test.yaml",
            space=None,  # Explicitly set to None
        )

        result = config.to_dict()

        assert "space" not in result


class TestHyperparameterTunerInit:
    """Test HyperparameterTuner initialization."""

    def test_HyperparameterTuner_init_default(self):
        """Test HyperparameterTuner initializes with default config."""
        tuner = HyperparameterTuner(config=TunerConfig())

        assert tuner.config is not None
        assert isinstance(tuner.config, TunerConfig)
        assert tuner.model is None
        assert tuner.results == {}

    def test_HyperparameterTuner_init_custom(self):
        """Test HyperparameterTuner initializes with custom config."""
        custom_config = TunerConfig(
            model="yolo11m.pt",
            data="custom.yaml",
            epochs=200,
            iterations=20,
        )

        tuner = HyperparameterTuner(config=custom_config)

        assert tuner.config is custom_config
        assert tuner.config.model == "yolo11m.pt"
        assert tuner.config.data == "custom.yaml"
        assert tuner.config.epochs == 200
        assert tuner.config.iterations == 20
        assert tuner.model is None
        assert tuner.results == {}
