"""
Pipeline Module - Orchestrates execution of multiple YOLO toolchain stages

This module provides a pipeline executor that can run multiple stages
defined in a YAML configuration file, enabling automated end-to-end workflows.
"""

import argparse
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union
import yaml

logger = logging.getLogger("yolo_toolchain.pipeline")


# =============================================================================
# Configuration Dataclasses
# =============================================================================

@dataclass
class StageConfig:
    """Configuration for a single pipeline stage."""
    name: str
    tool: str
    params: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    condition: Optional[str] = None  # e.g., "previous.failed" or "previous.success"
    timeout: Optional[int] = None  # Timeout in seconds
    continue_on_error: bool = False


@dataclass
class PipelineConfig:
    """Configuration for the entire pipeline."""
    name: str = "yolo_pipeline"
    description: str = ""
    stages: List[StageConfig] = field(default_factory=list)
    global_params: Dict[str, Any] = field(default_factory=dict)
    output_dir: str = "pipeline_output"
    log_level: str = "INFO"

    @classmethod
    def from_dict(cls, data: Dict) -> "PipelineConfig":
        """Create PipelineConfig from dictionary."""
        stages = []
        for stage_data in data.get("stages", []):
            if isinstance(stage_data, dict):
                stages.append(StageConfig(**stage_data))
            elif isinstance(stage_data, str):
                # Simple format: just tool name
                stages.append(StageConfig(name=stage_data, tool=stage_data))
            else:
                raise ValueError(f"Invalid stage config: {stage_data}")

        return cls(
            name=data.get("name", "yolo_pipeline"),
            description=data.get("description", ""),
            stages=stages,
            global_params=data.get("global_params", {}),
            output_dir=data.get("output_dir", "pipeline_output"),
            log_level=data.get("log_level", "INFO"),
        )

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "PipelineConfig":
        """Load PipelineConfig from YAML file."""
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)


@dataclass
class StageResult:
    """Result of a pipeline stage execution."""
    stage_name: str
    tool_name: str
    success: bool
    output: Any = None
    error: Optional[str] = None
    duration: float = 0.0
    start_time: Optional[float] = None
    end_time: Optional[float] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "stage_name": self.stage_name,
            "tool_name": self.tool_name,
            "success": self.success,
            "output": str(self.output) if self.output else None,
            "error": self.error,
            "duration": self.duration,
            "start_time": self.start_time,
            "end_time": self.end_time,
        }


# =============================================================================
# Tool Registry
# =============================================================================

class ToolRegistry:
    """Registry for pipeline tools."""

    _instance: Optional["ToolRegistry"] = None
    _tools: Dict[str, Callable] = {}

    def __new__(cls):
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._tools = {}
        return cls._instance

    def register(self, name: str, func: Callable) -> None:
        """Register a tool function.

        Args:
            name: Tool name
            func: Tool function
        """
        self._tools[name] = func
        logger.debug(f"Registered tool: {name}")

    def get(self, name: str) -> Optional[Callable]:
        """Get a registered tool by name.

        Args:
            name: Tool name

        Returns:
            Tool function or None if not found
        """
        return self._tools.get(name)

    def execute(self, name: str, params: Dict[str, Any], context: Dict[str, Any]) -> Any:
        """Execute a registered tool.

        Args:
            name: Tool name
            params: Tool parameters
            context: Pipeline context (shared state between stages)

        Returns:
            Tool execution result
        """
        tool = self.get(name)
        if tool is None:
            raise ValueError(f"Tool not found: {name}. Available tools: {list(self._tools.keys())}")

        # Merge context into params for tool execution
        merged_params = {**context, **params}
        return tool(merged_params)

    def list_tools(self) -> List[str]:
        """List all registered tools."""
        return list(self._tools.keys())


def register_tool(name: str) -> Callable:
    """Decorator to register a tool function.

    Args:
        name: Tool name

    Returns:
        Decorator function

    Example:
        @register_tool("download")
        def tool_download(params):
            ...
    """
    def decorator(func: Callable) -> Callable:
        registry = ToolRegistry()
        registry.register(name, func)
        return func
    return decorator


# =============================================================================
# Pipeline Executor
# =============================================================================

class PipelineExecutor:
    """Executes a pipeline of stages."""

    def __init__(self, config: Union[PipelineConfig, str, Dict]):
        """Initialize pipeline executor.

        Args:
            config: PipelineConfig, path to YAML file, or dict
        """
        if isinstance(config, PipelineConfig):
            self.config = config
        elif isinstance(config, str):
            self.config = PipelineConfig.from_yaml(config)
        elif isinstance(config, Dict):
            self.config = PipelineConfig.from_dict(config)
        else:
            raise ValueError(f"Invalid config type: {type(config)}")

        self.context: Dict[str, Any] = {}
        self.stage_results: Dict[str, StageResult] = {}
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None

        # Merge global params into context
        self.context.update(self.config.global_params)

    def execute(self) -> Dict[str, Any]:
        """Execute the pipeline.

        Returns:
            Pipeline execution report
        """
        self.start_time = time.time()
        logger.info(f"Starting pipeline: {self.config.name}")
        logger.info(f"Description: {self.config.description}")
        logger.info(f"Stages: {len(self.config.stages)}")

        for i, stage in enumerate(self.config.stages):
            stage_name = stage.name or f"stage_{i}"
            logger.info(f"\n{'='*60}")
            logger.info(f"Executing stage: {stage_name} (tool: {stage.tool})")
            logger.info(f"{'='*60}")

            # Check if stage should run
            if not stage.enabled:
                logger.info(f"Stage {stage_name} is disabled, skipping")
                continue

            # Evaluate condition if present
            if stage.condition and not self._evaluate_condition(stage.condition):
                logger.info(f"Stage {stage_name} condition '{stage.condition}' not met, skipping")
                continue

            # Execute stage
            result = self._execute_stage(stage, stage_name)
            self.stage_results[stage_name] = result

            # Store output in context
            if result.success and result.output is not None:
                self.context[stage_name] = result.output
                self.context[f"{stage_name}_output"] = result.output

            # Handle failure
            if not result.success:
                if stage.continue_on_error:
                    logger.warning(f"Stage {stage_name} failed, but continuing: {result.error}")
                else:
                    logger.error(f"Stage {stage_name} failed: {result.error}")
                    break

        self.end_time = time.time()
        return self._build_report()

    def _execute_stage(self, stage: StageConfig, stage_name: str) -> StageResult:
        """Execute a single stage.

        Args:
            stage: Stage configuration
            stage_name: Stage name

        Returns:
            Stage result
        """
        registry = ToolRegistry()
        tool = registry.get(stage.tool)

        if tool is None:
            return StageResult(
                stage_name=stage_name,
                tool_name=stage.tool,
                success=False,
                error=f"Tool not found: {stage.tool}. Available: {registry.list_tools()}",
            )

        start_time = time.time()
        try:
            logger.info(f"Tool params: {stage.params}")

            # Execute tool with merged params
            output = tool({**self.context, **stage.params})

            end_time = time.time()
            duration = end_time - start_time

            logger.info(f"Stage {stage_name} completed successfully in {duration:.2f}s")

            return StageResult(
                stage_name=stage_name,
                tool_name=stage.tool,
                success=True,
                output=output,
                duration=duration,
                start_time=start_time,
                end_time=end_time,
            )

        except Exception as e:
            end_time = time.time()
            duration = end_time - start_time
            error_msg = f"{type(e).__name__}: {str(e)}"
            logger.error(f"Stage {stage_name} failed: {error_msg}")

            return StageResult(
                stage_name=stage_name,
                tool_name=stage.tool,
                success=False,
                error=error_msg,
                duration=duration,
                start_time=start_time,
                end_time=end_time,
            )

    def _evaluate_condition(self, condition: str) -> bool:
        """Evaluate a stage condition.

        Args:
            condition: Condition string (e.g., "previous.failed", "previous.success")

        Returns:
            True if condition is met
        """
        parts = condition.split(".")
        if len(parts) != 2:
            logger.warning(f"Invalid condition format: {condition}")
            return False

        _, status = parts

        # Get the last executed stage result
        if not self.stage_results:
            return status == "success"  # Default to success for first stage

        last_result = list(self.stage_results.values())[-1]

        if status == "success":
            return last_result.success
        elif status == "failed":
            return not last_result.success
        else:
            logger.warning(f"Unknown condition status: {status}")
            return False

    def _build_report(self) -> Dict[str, Any]:
        """Build pipeline execution report.

        Returns:
            Report dictionary
        """
        total_duration = (self.end_time or time.time()) - (self.start_time or time.time())
        successful_stages = sum(1 for r in self.stage_results.values() if r.success)
        failed_stages = sum(1 for r in self.stage_results.values() if not r.success)

        report = {
            "pipeline_name": self.config.name,
            "description": self.config.description,
            "status": "success" if failed_stages == 0 else "failed",
            "total_duration": total_duration,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "total_stages": len(self.config.stages),
            "successful_stages": successful_stages,
            "failed_stages": failed_stages,
            "stage_results": {name: result.to_dict() for name, result in self.stage_results.items()},
            "context_keys": list(self.context.keys()),
        }

        return report

    def save_report(self, output_path: str) -> str:
        """Save pipeline report to file.

        Args:
            output_path: Output file path

        Returns:
            Path to saved report
        """
        report = self._build_report()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            yaml.dump(report, f, default_flow_style=False, sort_keys=False)

        logger.info(f"Report saved to: {output_path}")
        return str(output_path)


# =============================================================================
# Tool Implementations
# =============================================================================

@register_tool("download")
def tool_download(params: Dict[str, Any]) -> Dict[str, Any]:
    """Download YOLO model.

    Params:
        model_name: Model name (e.g., yolo11n)
        output_dir: Output directory (default: models)

    Returns:
        Dict with downloaded model path
    """
    from .downloader import download_yolo_model

    model_name = params.get("model_name", "yolo11n")
    output_dir = params.get("output_dir", "models")

    logger.info(f"Downloading model: {model_name}")

    model_path = download_yolo_model(
        model_name=model_name,
        output_dir=output_dir,
    )

    return {"model_path": model_path, "model_name": model_name}


@register_tool("validate")
def tool_validate(params: Dict[str, Any]) -> Dict[str, Any]:
    """Validate dataset.

    Params:
        dataset: Path to dataset YAML
        model: Model path (optional)

    Returns:
        Validation results
    """
    from ultralytics import YOLO

    dataset = params.get("dataset")
    model_path = params.get("model", "yolo11n.pt")

    if not dataset:
        raise ValueError("Parameter 'dataset' is required for validate tool")

    logger.info(f"Validating dataset: {dataset}")

    # Load model
    if Path(model_path).exists():
        model = YOLO(model_path)
    else:
        logger.warning(f"Model {model_path} not found, using default")
        model = YOLO("yolo11n.pt")

    # Run validation
    results = model.val(data=dataset, verbose=False)

    metrics = {
        "mAP50": float(results.box.map50) if hasattr(results.box, 'map50') else 0.0,
        "mAP50-95": float(results.box.map) if hasattr(results.box, 'map') else 0.0,
        "precision": float(results.box.mp) if hasattr(results.box, 'mp') else 0.0,
        "recall": float(results.box.mr) if hasattr(results.box, 'mr') else 0.0,
    }

    logger.info(f"Validation results: mAP50={metrics['mAP50']:.4f}, mAP50-95={metrics['mAP50-95']:.4f}")

    return {
        "dataset": dataset,
        "metrics": metrics,
        "success": metrics["mAP50"] > 0,
    }


@register_tool("label-qc")
def tool_label_qc(params: Dict[str, Any]) -> Dict[str, Any]:
    """Label quality check.

    Params:
        dataset: Path to dataset directory
        apply_fixes: Whether to apply auto-fixes (default: False)
        duplicate_iou: IoU threshold for duplicate detection (default: 0.8)
        min_area: Minimum box area in pixels (default: 100)

    Returns:
        QC check results
    """
    from .label_qc import LabelQCChecker, LabelQCConfig

    dataset = params.get("dataset")
    if not dataset:
        raise ValueError("Parameter 'dataset' is required for label-qc tool")

    apply_fixes = params.get("apply_fixes", False)
    duplicate_iou = params.get("duplicate_iou", 0.8)
    min_area = params.get("min_area", 100)

    logger.info(f"Running label quality check on: {dataset}")

    config = LabelQCConfig(
        duplicate_iou_threshold=duplicate_iou,
        min_box_area=min_area,
    )

    checker = LabelQCChecker(config)
    results = checker.check(dataset)

    if apply_fixes:
        logger.info("Applying auto-fixes...")
        fix_results = checker.apply_fixes(dataset)
        results["fixes_applied"] = fix_results

    # Summarize
    stats = results.get("stats", {})
    summary = {
        "total_images": stats.get("total_images", 0),
        "images_with_issues": stats.get("images_with_issues", 0),
        "duplicate_boxes": stats.get("duplicate_boxes", 0),
        "tiny_boxes": stats.get("tiny_boxes", 0),
        "oversized_boxes": stats.get("oversized_boxes", 0),
    }

    logger.info(f"Label QC complete: {summary}")

    return {
        "dataset": dataset,
        "summary": summary,
        "results": results,
        "success": summary["images_with_issues"] == 0,
    }


@register_tool("anchors")
def tool_anchors(params: Dict[str, Any]) -> Dict[str, Any]:
    """Generate anchors.

    Params:
        data: Path to dataset YAML
        output: Output file path (default: anchors.yaml)
        min_k: Minimum clusters (default: 5)
        max_k: Maximum clusters (default: 15)

    Returns:
        Anchor generation results
    """
    from .anchor_generator import AnchorGenerator, AnchorConfig

    data = params.get("data")
    if not data:
        raise ValueError("Parameter 'data' is required for anchors tool")

    output = params.get("output", "anchors.yaml")
    min_k = params.get("min_k", 5)
    max_k = params.get("max_k", 15)

    logger.info(f"Generating anchors for dataset: {data}")

    config = AnchorConfig(
        n_clusters_min=min_k,
        n_clusters_max=max_k,
    )

    generator = AnchorGenerator(config)
    generator.load_dataset(data)
    anchors = generator.generate()
    output_path = generator.save_anchors(output)

    # Validate if possible
    validate_results = None
    try:
        validate_results = generator.validate_with_anchors(data)
    except Exception as e:
        logger.warning(f"Anchor validation failed: {e}")

    logger.info(f"Anchors saved to: {output_path}")

    return {
        "data": data,
        "anchors": anchors,
        "output_path": output_path,
        "validation": validate_results,
        "success": True,
    }


@register_tool("train")
def tool_train(params: Dict[str, Any]) -> Dict[str, Any]:
    """Train YOLO model.

    Params:
        model: Model path (default: yolo11n.pt)
        data: Dataset YAML path
        epochs: Number of epochs (default: 100)
        imgsz: Image size (default: 640)
        batch: Batch size (default: 16)
        device: Device (default: 0)
        project: Project directory (default: runs/train)
        name: Experiment name (default: exp)
        resume: Whether to resume training (default: False)

    Returns:
        Training results
    """
    from ..train.trainer import Trainer, TrainConfig

    model = params.get("model", "yolo11n.pt")
    data = params.get("data")
    epochs = params.get("epochs", 100)
    imgsz = params.get("imgsz", 640)
    batch = params.get("batch", 16)
    device = params.get("device", "0")
    project = params.get("project", "runs/train")
    name = params.get("name", "exp")
    resume = params.get("resume", False)

    if not data:
        raise ValueError("Parameter 'data' is required for train tool")

    logger.info(f"Training model: {model}, data: {data}, epochs: {epochs}")

    config = TrainConfig(
        model=model,
        data=data,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=str(device),
        project=project,
        name=name,
        resume=resume,
    )

    trainer = Trainer(config)

    # Find checkpoint if resuming
    if resume and not config.model.endswith('.pt'):
        checkpoint = trainer.find_last_checkpoint()
        if checkpoint:
            config.model = checkpoint
            logger.info(f"Resuming from: {checkpoint}")

    # Train
    results = trainer.train(resume=resume)

    # Get best model path
    best_model = trainer.get_best_checkpoint()

    return {
        "model": model,
        "data": data,
        "epochs": epochs,
        "best_model": best_model,
        "results": results,
        "success": best_model is not None,
    }


@register_tool("export")
def tool_export(params: Dict[str, Any]) -> Dict[str, Any]:
    """Export YOLO model.

    Params:
        model: Model path
        format: Export format (default: onnx)
        imgsz: Image size (default: 640)
        half: Whether to use FP16 (default: False)

    Returns:
        Export results
    """
    from ..export.exporter import ModelExporter, ExportConfig

    model = params.get("model")
    if not model:
        raise ValueError("Parameter 'model' is required for export tool")

    format = params.get("format", "onnx")
    imgsz = params.get("imgsz", 640)
    half = params.get("half", False)

    logger.info(f"Exporting model: {model}, format: {format}")

    config = ExportConfig(
        model=model,
        format=format,
        imgsz=imgsz,
        half=half,
    )

    exporter = ModelExporter(model, config)
    export_path = exporter.export()

    return {
        "model": model,
        "format": format,
        "export_path": export_path,
        "success": Path(export_path).exists() if export_path else False,
    }


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    """CLI entry point for pipeline executor."""
    parser = argparse.ArgumentParser(
        description="YOLO Toolchain Pipeline Executor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run pipeline from YAML config
  yolo-pipeline --config pipeline.yaml

  # Run with custom output directory
  yolo-pipeline --config pipeline.yaml --output my_output

  # Run pipeline and save report
  yolo-pipeline --config pipeline.yaml --report report.yaml

  # List available tools
  yolo-pipeline --list-tools

Pipeline YAML format:
  name: my_pipeline
  description: Training pipeline
  output_dir: ./output
  global_params:
    model: yolo11n.pt
  stages:
    - name: download_model
      tool: download
      params:
        model_name: yolo11n
        output_dir: models
    - name: train
      tool: train
      params:
        data: dataset.yaml
        epochs: 100
      condition: previous.success
        """
    )

    parser.add_argument(
        "--config",
        type=str,
        help="Path to pipeline YAML config file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="pipeline_output",
        help="Output directory for pipeline artifacts"
    )
    parser.add_argument(
        "--report",
        type=str,
        help="Path to save pipeline execution report"
    )
    parser.add_argument(
        "--list-tools",
        action="store_true",
        help="List all available pipeline tools"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )

    args = parser.parse_args()

    # Configure logging
    import logging as log_module
    log_level = getattr(log_module, args.log_level.upper(), log_module.INFO)
    log_module.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # List tools
    if args.list_tools:
        registry = ToolRegistry()
        tools = registry.list_tools()
        print("\nAvailable pipeline tools:")
        print("-" * 40)
        for tool in sorted(tools):
            print(f"  - {tool}")
        print()
        return

    # Require config
    if not args.config:
        parser.error("--config is required unless --list-tools is specified")

    # Execute pipeline
    try:
        executor = PipelineExecutor(args.config)
        executor.config.output_dir = args.output
        report = executor.execute()

        # Print summary
        print("\n" + "=" * 60)
        print("Pipeline Execution Summary")
        print("=" * 60)
        print(f"Pipeline: {report['pipeline_name']}")
        print(f"Status: {report['status']}")
        print(f"Total Duration: {report['total_duration']:.2f}s")
        print(f"Stages: {report['successful_stages']}/{report['total_stages']} successful")
        print("=" * 60)

        if report['failed_stages'] > 0:
            print("\nFailed stages:")
            for name, result in report['stage_results'].items():
                if not result['success']:
                    print(f"  - {name}: {result['error']}")
            print()

        # Save report
        if args.report:
            executor.save_report(args.report)

    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        raise


if __name__ == "__main__":
    main()
