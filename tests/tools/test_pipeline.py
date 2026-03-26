"""
Unit tests for pipeline module.
"""

import pytest
import tempfile
import os
from pathlib import Path

from src.tools.pipeline import (
    PipelineConfig,
    StageConfig,
    StageResult,
    ToolRegistry,
    PipelineExecutor,
)


class TestPipelineConfig:
    """Tests for PipelineConfig."""

    def test_PipelineConfig_from_dict(self):
        """Create config from dict."""
        data = {
            "name": "test_pipeline",
            "description": "Test pipeline description",
            "stages": [
                {"name": "stage1", "tool": "download", "params": {"model_name": "yolo11n"}},
                {"name": "stage2", "tool": "train", "params": {"epochs": 100}},
            ],
            "global_params": {"device": "0"},
            "output_dir": "test_output",
            "log_level": "DEBUG",
        }

        config = PipelineConfig.from_dict(data)

        assert config.name == "test_pipeline"
        assert config.description == "Test pipeline description"
        assert len(config.stages) == 2
        assert config.stages[0].name == "stage1"
        assert config.stages[0].tool == "download"
        assert config.stages[0].params == {"model_name": "yolo11n"}
        assert config.stages[1].name == "stage2"
        assert config.stages[1].tool == "train"
        assert config.global_params == {"device": "0"}
        assert config.output_dir == "test_output"
        assert config.log_level == "DEBUG"

    def test_PipelineConfig_from_yaml(self, tmp_path):
        """Load config from YAML file."""
        yaml_content = """
name: yaml_pipeline
description: Loaded from YAML
stages:
  - name: download_stage
    tool: download
    params:
      model_name: yolo11n
  - train_stage
output_dir: yaml_output
log_level: INFO
"""
        yaml_file = tmp_path / "pipeline.yaml"
        yaml_file.write_text(yaml_content)

        config = PipelineConfig.from_yaml(str(yaml_file))

        assert config.name == "yaml_pipeline"
        assert config.description == "Loaded from YAML"
        assert len(config.stages) == 2
        assert config.stages[0].name == "download_stage"
        assert config.stages[0].tool == "download"
        assert config.stages[1].name == "train_stage"
        assert config.stages[1].tool == "train_stage"
        assert config.output_dir == "yaml_output"


class TestToolRegistry:
    """Tests for ToolRegistry."""

    def test_ToolRegistry_register_and_get(self):
        """Register and retrieve a tool."""
        registry = ToolRegistry()

        def dummy_tool(params):
            return {"result": "success"}

        registry.register("dummy", dummy_tool)
        tool = registry.get("dummy")

        assert tool is not None
        assert tool({"test": "param"}) == {"result": "success"}

    def test_ToolRegistry_get_nonexistent(self):
        """Returns None for unknown tool."""
        registry = ToolRegistry()

        tool = registry.get("nonexistent_tool")

        assert tool is None

    def test_ToolRegistry_execute_success(self):
        """Execute tool successfully."""
        registry = ToolRegistry()

        def add_tool(params):
            return {"sum": params["a"] + params["b"]}

        registry.register("add", add_tool)
        result = registry.execute("add", {"a": 1, "b": 2}, {})

        assert result == {"sum": 3}

    def test_ToolRegistry_execute_failure(self):
        """Handle tool execution failure."""
        registry = ToolRegistry()

        def failing_tool(params):
            raise ValueError("Tool execution failed")

        registry.register("failing", failing_tool)

        with pytest.raises(ValueError, match="Tool execution failed"):
            registry.execute("failing", {}, {})


class TestStageResult:
    """Tests for StageResult."""

    def test_StageResult_to_dict(self):
        """Convert result to dict."""
        result = StageResult(
            stage_name="test_stage",
            tool_name="test_tool",
            success=True,
            output={"key": "value"},
            error=None,
            duration=1.5,
            start_time=1000.0,
            end_time=1001.5,
        )

        result_dict = result.to_dict()

        assert result_dict["stage_name"] == "test_stage"
        assert result_dict["tool_name"] == "test_tool"
        assert result_dict["success"] is True
        assert result_dict["output"] == "{'key': 'value'}"
        assert result_dict["error"] is None
        assert result_dict["duration"] == 1.5
        assert result_dict["start_time"] == 1000.0
        assert result_dict["end_time"] == 1001.5


class TestPipelineExecutor:
    """Tests for PipelineExecutor."""

    def test_PipelineExecutor_init(self):
        """Initialize executor with config."""
        config_data = {
            "name": "init_test_pipeline",
            "stages": [],
            "global_params": {"key": "value"},
        }

        executor = PipelineExecutor(config_data)

        assert executor.config.name == "init_test_pipeline"
        assert executor.context == {"key": "value"}
        assert executor.stage_results == {}
