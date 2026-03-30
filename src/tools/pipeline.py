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
import numpy as np

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
# Helper Functions
# =============================================================================

import re

def _resolve_var_refs(params: Dict[str, Any], context: Dict[str, Any], global_params: Dict[str, Any] = None) -> Dict[str, Any]:
    """Resolve ${var} references in params using context and global_params values.

    Args:
        params: Parameters that may contain ${var} references
        context: Context dict with values to substitute
        global_params: Global params dict (higher priority for variable resolution)

    Returns:
        Params with resolved references
    """
    import re
    var_pattern = re.compile(r'\$\{([^}]+)\}')

    def resolve_value(value: Any) -> Any:
        if isinstance(value, str):
            # Resolve ${var} references in strings
            def replace_var(m):
                var_name = m.group(1)
                # Handle nested context like global_params.project
                if '.' in var_name:
                    parts = var_name.split('.')
                    val = context
                    for part in parts:
                        if isinstance(val, dict):
                            val = val.get(part, m.group(0))
                        else:
                            return m.group(0)
                    return str(val) if val is not None else m.group(0)
                else:
                    # First check global_params (higher priority), then context
                    if global_params and var_name in global_params:
                        return str(global_params[var_name])
                    return str(context.get(var_name, m.group(0)))
            return var_pattern.sub(replace_var, value)
        elif isinstance(value, dict):
            return {k: resolve_value(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [resolve_value(item) for item in value]
        else:
            return value

    return resolve_value(params)


def _resolve_project_in_path(path_value: str, params: Dict[str, Any]) -> str:
    """Resolve ${project} in path strings.

    Args:
        path_value: Path string that may contain ${project}
        params: Full params dict to get global_params.project as fallback

    Returns:
        Path with ${project} resolved
    """
    if not path_value or '${project}' not in str(path_value):
        return path_value

    global_project = params.get("global_params", {}).get("project", None)
    if global_project:
        resolved = path_value.replace('${project}', global_project)
        logger.debug(f"Resolved {path_value} -> {resolved}")
        return resolved
    return path_value


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
            # Merge global_params into stage.params (global_params as defaults, stage.params override)
            merged_params = {**self.config.global_params, **stage.params}
            # Resolve ${var} references in params
            merged_params = _resolve_var_refs(merged_params, {**self.context, **merged_params}, self.config.global_params)
            logger.info(f"Tool params: {merged_params}")

            # Execute tool with context and merged params
            output = tool({**self.context, **merged_params})

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
        project: Project directory (default: runs/val)
        name: Experiment name (default: exp)

    Returns:
        Validation results
    """
    from ultralytics import YOLO

    dataset = params.get("dataset")
    model_path = params.get("model", "yolo11n.pt")
    project = params.get("project", "runs/val")
    # Resolve ${project} in path
    project = _resolve_project_in_path(project, params)
    # Resolve to absolute path
    if project and project.startswith('./'):
        project = str(Path(project).resolve())
    name = params.get("name", "exp")

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
    results = model.val(data=dataset, project=project, name=name, verbose=False)

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
        max_box_area_ratio: Maximum box area as ratio of image (default: 0.9)
        min_box_area_ratio: Minimum box area as ratio of image (default: 0.001)
        occlusion_variance_threshold: Variance threshold for occlusion detection (default: 50.0)
        image_size: Image size for normalization, tuple (w, h) (default: (640, 640))
        backup_before_fix: Backup labels before fixing (default: True)
        report_format: Report format, 'text' or 'json' (default: 'text')

    Returns:
        QC check results
    """
    from .label_qc import LabelQCChecker, LabelQCConfig

    dataset = params.get("dataset")
    if not dataset:
        raise ValueError("Parameter 'dataset' is required for label-qc tool")

    apply_fixes = params.get("apply_fixes", False)

    logger.info(f"Running label quality check on: {dataset}")

    config = LabelQCConfig(
        duplicate_iou_threshold=params.get("duplicate_iou", 0.8),
        min_box_area=params.get("min_area", 100),
        max_box_area_ratio=params.get("max_box_area_ratio", 0.9),
        min_box_area_ratio=params.get("min_box_area_ratio", 0.001),
        occlusion_variance_threshold=params.get("occlusion_variance_threshold", 50.0),
        image_size=tuple(params.get("image_size", [640, 640])),
        backup_before_fix=params.get("backup_before_fix", True),
        report_format=params.get("report_format", "text"),
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
        "occlusion_boxes": stats.get("occlusion_boxes", 0),
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
    """Generate anchors and optionally update dataset YAML.

    Params:
        data: Path to dataset YAML
        output: Output file path (default: anchors.yaml)
        min_k: Minimum clusters (default: 5)
        max_k: Maximum clusters (default: 15)
        scales: List of scales to use (default: ['P3', 'P4', 'P5'])
        min_bbox_area: Minimum bbox area to consider (default: 0.0001)
        validate: Whether to validate anchors after generation (default: False)
        update_dataset: Whether to update dataset.yaml with anchors path (default: True)

    Returns:
        Anchor generation results
    """
    from .anchor_generator import AnchorGenerator, AnchorConfig
    import yaml

    data = params.get("data")
    if not data:
        raise ValueError("Parameter 'data' is required for anchors tool")

    output = params.get("output", "anchors.yaml")
    # Resolve ${project} in path
    output = _resolve_project_in_path(output, params)
    scales = params.get("scales", ["P3", "P4", "P5"])
    do_validate = params.get("validate", False)
    update_dataset = params.get("update_dataset", True)

    logger.info(f"Generating anchors for dataset: {data}")

    config = AnchorConfig(
        n_clusters_min=params.get("min_k", 5),
        n_clusters_max=params.get("max_k", 15),
        scales=scales,
        min_bbox_area=params.get("min_bbox_area", 0.0001),
        validate=do_validate,
    )

    generator = AnchorGenerator(config)
    generator.load_dataset(data)
    anchors = generator.generate()
    output_path = generator.save_anchors(output)

    # Update dataset.yaml to include anchors reference
    if update_dataset:
        try:
            dataset_path = Path(data)
            with open(dataset_path, 'r') as f:
                dataset_config = yaml.safe_load(f)

            # Add anchors path to dataset config
            dataset_config['anchors'] = str(Path(output).resolve())

            # Write back
            with open(dataset_path, 'w') as f:
                yaml.dump(dataset_config, f, default_flow_style=False, sort_keys=False)

            logger.info(f"Updated dataset.yaml with anchors reference: {data}")
        except Exception as e:
            logger.warning(f"Failed to update dataset.yaml: {e}")

    # Validate if requested
    validate_results = None
    if do_validate:
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
        data: Dataset YAML path (required)
        epochs: Number of epochs (default: 100)
        imgsz: Image size (default: 640)
        batch: Batch size (default: 16)
        device: Device (default: '0')
        project: Project directory (default: runs/train)
        name: Experiment name (default: exp)
        resume: Whether to resume training (default: False)

        # Training strategy
        optimizer: Optimizer (default: 'auto')
        lr0: Initial learning rate (default: 0.01)
        lrf: Final learning rate factor (default: 0.01)
        momentum: SGD momentum (default: 0.937)
        weight_decay: Weight decay (default: 0.0005)
        warmup_epochs: Warmup epochs (default: 3.0)
        patience: Early stopping patience (default: 50)

        # Model options
        pretrained: Use pretrained weights (default: True)
        freeze: Freeze layers (default: [])
        single_cls: Single class training (default: False)

        # Training behavior
        save: Save checkpoints (default: True)
        save_period: Save every N epochs (default: -1)
        cache: Cache images in memory (default: False)
        workers: Data loader workers (default: 8)
        verbose: Verbose output (default: True)
        seed: Random seed (default: 0)
        deterministic: Deterministic training (default: True)
        rect: Rectangular training (default: False)
        cos_lr: Cosine learning rate (default: False)
        close_mosaic: Disable mosaic in last N epochs (default: 10)
        amp: Mixed precision (default: True)
        fraction: Data fraction to use (default: 1.0)
        profile: Profile training speed (default: False)
        dropout: Dropout rate (default: 0.0)

        # Validation & visualization
        val: Validate during training (default: True)
        plots: Generate training plots (default: True)
        conf: Confidence threshold for detection (default: 0.25)
        iou: NMS IoU threshold (default: 0.7)
        anchors: Number of anchors (auto-k calculation, default: 0 = use dataset anchors)

        # Data augmentation
        hsv_h: Hue augmentation (default: 0.015)
        hsv_s: Saturation augmentation (default: 0.7)
        hsv_v: Value augmentation (default: 0.4)
        degrees: Random rotation (default: 0.0)
        translate: Random translation (default: 0.1)
        scale: Random scale (default: 0.5)
        shear: Random shear (default: 0.0)
        perspective: Random perspective (default: 0.0)
        flipud: Vertical flip probability (default: 0.0)
        fliplr: Horizontal flip probability (default: 0.5)
        mosaic: Mosaic augmentation probability (default: 1.0)
        mixup: MixUp augmentation probability (default: 0.0)
        copy_paste: Copy-paste augmentation probability (default: 0.0)

        # Class imbalance handling
        class_weights: List of class weights for imbalance (e.g., [1.0, 2.0])
        cls_loss_gain: Classification loss gain (default: 0.0)
        box_loss_gain: Box loss gain (default: 0.0)
        dfl_loss_gain: DFL loss gain (default: 0.0)

        # Regularization
        label_smoothing: Label smoothing factor (default: 0.0)

        # EMA (Exponential Moving Average)
        ema: EMA decay rate (default: 0.0 = disabled, 0.9999 = recommended)

        # Gradient accumulation (simulate larger batch)
        accumulate: Gradient accumulation steps (default: 1, e.g., batch=4, accumulate=4 → effective_batch=16)

    Returns:
        Training results
    """
    from ..train.trainer import Trainer, TrainConfig

    model = params.get("model", "yolo11n.pt")
    data = params.get("data")

    # Resolve ${project} in model path (if not resolved by pipeline)
    if '${project}' in str(model):
        global_project = params.get("global_params", {}).get("project", "runs/train")
        if global_project:
            model = model.replace('${project}', global_project)
            logger.info(f"Resolved model path: {model}")

    # Resolve project path to absolute path for YOLO
    project = params.get("project", "runs/train")
    # Handle ${project} variable reference (if not resolved by pipeline)
    if '${project}' in str(project):
        global_project = params.get("global_params", {}).get("project", "runs/train")
        if global_project and global_project != project:
            project = project.replace('${project}', global_project)
            logger.info(f"Resolved ${project} to: {project}")
    # Resolve relative paths to absolute
    if project and project.startswith('./'):
        project = str(Path(project).resolve())
        logger.info(f"Resolved project path to absolute: {project}")

    # If anchors output path is in context (from anchors stage), ensure dataset has it
    anchors_output = params.get("anchors_output") or params.get("生成 Anchors_output")
    if anchors_output and data:
        import yaml
        try:
            with open(data, 'r') as f:
                dataset_config = yaml.safe_load(f)
            if 'anchors' not in dataset_config:
                dataset_config['anchors'] = anchors_output
                with open(data, 'w') as f:
                    yaml.dump(dataset_config, f, default_flow_style=False, sort_keys=False)
                logger.info(f"Injected anchors {anchors_output} into dataset.yaml")
        except Exception as e:
            logger.warning(f"Failed to inject anchors into dataset.yaml: {e}")

    if not data:
        raise ValueError("Parameter 'data' is required for train tool")

    logger.info(f"Training model: {model}, data: {data}, epochs: {params.get('epochs', 100)}")

    config = TrainConfig(
        model=model,
        data=data,
        epochs=params.get("epochs", 100),
        imgsz=params.get("imgsz", 640),
        batch=params.get("batch", 16),
        device=str(params.get("device", "0")),
        project=project,
        name=params.get("name", "exp"),
        resume=params.get("resume", False),
        optimizer=params.get("optimizer", "auto"),
        lr0=params.get("lr0", 0.01),
        lrf=params.get("lrf", 0.01),
        momentum=params.get("momentum", 0.937),
        weight_decay=params.get("weight_decay", 0.0005),
        warmup_epochs=params.get("warmup_epochs", 3.0),
        patience=params.get("patience", 50),
        save=params.get("save", True),
        save_period=params.get("save_period", -1),
        cache=params.get("cache", False),
        workers=params.get("workers", 8),
        pretrained=params.get("pretrained", True),
        verbose=params.get("verbose", True),
        seed=params.get("seed", 0),
        deterministic=params.get("deterministic", True),
        single_cls=params.get("single_cls", False),
        rect=params.get("rect", False),
        cos_lr=params.get("cos_lr", False),
        close_mosaic=params.get("close_mosaic", 10),
        amp=params.get("amp", True),
        fraction=params.get("fraction", 1.0),
        profile=params.get("profile", False),
        dropout=params.get("dropout", 0.0),
        val=params.get("val", True),
        plots=params.get("plots", True),
        freeze=params.get("freeze", []),
        conf=params.get("conf", 0.25),
        iou=params.get("iou", 0.7),
        anchors=params.get("anchors", 0),
        # 数据增强参数
        hsv_h=params.get("hsv_h", 0.015),
        hsv_s=params.get("hsv_s", 0.7),
        hsv_v=params.get("hsv_v", 0.4),
        degrees=params.get("degrees", 0.0),
        translate=params.get("translate", 0.1),
        scale=params.get("scale", 0.5),
        shear=params.get("shear", 0.0),
        perspective=params.get("perspective", 0.0),
        flipud=params.get("flipud", 0.0),
        fliplr=params.get("fliplr", 0.5),
        mosaic=params.get("mosaic", 1.0),
        mixup=params.get("mixup", 0.0),
        copy_paste=params.get("copy_paste", 0.0),
        # 类别不平衡处理
        class_weights=params.get("class_weights", []),
        cls_loss_gain=params.get("cls_loss_gain", 0.0),
        box_loss_gain=params.get("box_loss_gain", 0.0),
        dfl_loss_gain=params.get("dfl_loss_gain", 0.0),
        # 正则化
        label_smoothing=params.get("label_smoothing", 0.0),
        # EMA
        ema=params.get("ema", 0.0),
        # 梯度累积
        accumulate=params.get("accumulate", 1),
    )

    trainer = Trainer(config)

    # Find checkpoint if resuming
    resume = params.get("resume", False)
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
        "epochs": params.get("epochs", 100),
        "best_model": best_model,
        "results": results,
        "success": best_model is not None,
    }


@register_tool("auto-annotate")
def tool_auto_annotate(params: Dict[str, Any]) -> Dict[str, Any]:
    """Auto-annotate images using AI.

    Params:
        images: Path to images directory or single image
        output: Output directory
        dataset: Dataset YAML path (optional)
        classes: List of class names (e.g., ["smoking"])
        conf: Confidence threshold (default: 0.3)
        workers: Number of parallel workers (default: 10)
        seed: Random seed (default: 42)
        single: Whether to process single image (default: False)
        save_vis: Whether to save visualization images (default: True)

    Returns:
        Auto-annotation results
    """
    from .auto_annotator import auto_annotate_dataset

    images = params.get("images")
    if not images:
        raise ValueError("Parameter 'images' is required for auto-annotate tool")

    output = params.get("output", "./dataset")
    dataset = params.get("dataset")
    classes = params.get("classes", [])
    conf = params.get("conf", 0.3)
    workers = params.get("workers", 10)
    seed = params.get("seed", 42)
    save_vis = params.get("save_vis", True)

    logger.info(f"Auto-annotating images: {images}")

    # For single image mode, use annotate_image directly
    if params.get("single", False):
        from .auto_annotator import AutoAnnotator, AutoAnnotatorConfig
        import os

        api_key = params.get("api_key") or os.environ.get("SILICONFLOW_API_KEY", "")
        if not api_key:
            raise ValueError("API key required. Set SILICONFLOW_API_KEY in .env or use --api_key")

        config = AutoAnnotatorConfig(
            api_key=api_key,
            model=params.get("model", "Pro/moonshotai/Kimi-K2.5"),
            confidence_threshold=conf,
            timeout=params.get("timeout", 300),
        )
        annotator = AutoAnnotator(config)
        output_file = params.get("output_file", "labels.txt")

        annotations, class_names = annotator.annotate_image(
            images,
            classes=classes if classes else None,
            dataset_yaml=dataset,
        )
        annotator.save_annotations(annotations, class_names, output_file)

        # 保存可视化图片
        if save_vis:
            vis_path = Path(output_file).parent / f"{Path(output_file).stem}.jpg"
            annotator.draw_annotations_on_image(
                images,
                annotations,
                class_names,
                str(vis_path)
            )

        logger.info(f"Single image annotated: {len(annotations)} objects, saved to {output_file}")
        return {"images": images, "output": output_file, "annotated_count": len(annotations), "success": True}

    # Batch mode: use auto_annotate_dataset function
    auto_annotate_dataset(
        image_dir=images,
        output_dir=output,
        classes=classes if classes else None,
        dataset_yaml=dataset,
        api_key=params.get("api_key"),
        model=params.get("model", "Pro/moonshotai/Kimi-K2.5"),
        workers=workers,
        seed=seed,
        save_vis=save_vis,
    )
    return {
        "images": images,
        "output": output,
        "success": True,
    }


@register_tool("verify")
def tool_verify(params: Dict[str, Any]) -> Dict[str, Any]:
    """Verify annotations with interactive review.

    Params:
        images: Path to images directory (required)
        labels: Path to labels directory (required)
        classes: List of class names (required)
        mode: Verification mode (default: 'interactive')
            'interactive' - manual review with keyboard controls
            'auto' - auto-filter by confidence threshold
        conf: Confidence threshold for auto mode (default: 0.6)
        accept_threshold: Accept threshold for auto mode (default: 0.9)
        output_dir: Output directory for auto mode (default: None = in-place)

    Returns:
        Verification results
    """
    from .verify_annotator import AnnotationVerifier

    images = params.get("images")
    labels = params.get("labels")
    classes = params.get("classes", [])
    mode = params.get("mode", "interactive")
    conf = params.get("conf", 0.6)
    accept_threshold = params.get("accept_threshold", 0.9)
    output_dir = params.get("output_dir")

    if not images or not labels:
        raise ValueError("Parameters 'images' and 'labels' are required for verify tool")

    if not classes:
        raise ValueError("Parameter 'classes' is required for verify tool")

    logger.info(f"Verifying annotations: images={images}, labels={labels}, mode={mode}")

    verifier = AnnotationVerifier(images_dir=images, labels_dir=labels, class_names=classes)

    if mode == "auto":
        verifier.verify_auto(output_dir=output_dir, accept_threshold=accept_threshold)
        verified_count = len(verifier.image_files)
    else:
        # Interactive mode - this will block until user finishes
        verifier.verify_interactive()
        verified_count = len(verifier.image_files)

    return {
        "images": images,
        "labels": labels,
        "verified_count": verified_count,
        "success": True,
    }


@register_tool("error-analyze")
def tool_error_analyze(params: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze detection errors and categorize them.

    Args:
        model: Model path (required)
        data: Dataset YAML path (required)
        output_dir: Output directory (default: runs/error_analysis)
        conf_threshold: Confidence threshold (default: 0.25)
        iou_threshold: IoU threshold for matching (default: 0.5)

    Returns:
        Error analysis report
    """
    from ..eval.error_analyzer import run_error_analysis

    model = params.get("model")
    if not model:
        raise ValueError("Parameter 'model' is required for error-analyze tool")

    data = params.get("data")
    if not data:
        raise ValueError("Parameter 'data' is required for error-analyze tool")

    output_dir = params.get("output_dir", "runs/error_analysis")
    output_dir = _resolve_project_in_path(output_dir, params)
    if output_dir.startswith('./'):
        output_dir = str(Path(output_dir).resolve())
    conf_threshold = params.get("conf_threshold", 0.25)
    iou_threshold = params.get("iou_threshold", 0.5)

    logger.info(f"Running error analysis: model={model}, data={data}")

    report = run_error_analysis(
        model_path=model,
        data_yaml=data,
        output_dir=output_dir,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold
    )

    return {
        "model": model,
        "data": data,
        "output_dir": output_dir,
        "total_fp": report['summary']['total_fp'],
        "total_fn": report['summary']['total_fn'],
        "total_correct": report['summary']['total_correct'],
        "precision": report['summary']['precision'],
        "recall": report['summary']['recall'],
        "recommendations": report['recommendations'],
        "success": True,
    }


@register_tool("tune")
def tool_tune(params: Dict[str, Any]) -> Dict[str, Any]:
    """Run hyperparameter tuning using genetic algorithm.

    Args:
        model: Model path (default: yolo11n.pt)
        data: Dataset YAML path (required)
        epochs: Epochs per generation (default: 100)
        iterations: Number of generations (default: 10)
        output_dir: Output directory (default: runs/tune)
        patience: Early stopping patience (default: 50)
        metric: Metric to optimize (default: metrics/mAP50(B))
        direction: maximize or minimize (default: maximize)
        device: Device (default: '')
        space: Custom parameter space dict (optional)

    Returns:
        Tuning results with best hyperparameters
    """
    from ..tools.hyperparameter_tuner import HyperparameterTuner, TunerConfig

    model = params.get("model", "yolo11n.pt")
    data = params.get("data")

    if not data:
        raise ValueError("Parameter 'data' is required for tune tool")

    # Resolve project path
    output_dir = params.get("output_dir", "runs/tune")
    output_dir = _resolve_project_in_path(output_dir, params)
    if output_dir.startswith('./'):
        output_dir = str(Path(output_dir).resolve())

    config = TunerConfig(
        model=model,
        data=data,
        epochs=params.get("epochs", 100),
        iterations=params.get("iterations", 10),
        output_dir=output_dir,
        patience=params.get("patience", 50),
        metric=params.get("metric", "metrics/mAP50(B)"),
        direction=params.get("direction", "maximize"),
        device=str(params.get("device", "")) or "",
        space=params.get("space"),
    )

    logger.info(f"Starting hyperparameter tuning: model={model}, data={data}")
    logger.info(f"Epochs: {config.epochs}, Iterations: {config.iterations}")

    tuner = HyperparameterTuner(config)
    results = tuner.tune()

    logger.info("Hyperparameter tuning completed")

    return {
        "model": model,
        "data": data,
        "output_dir": output_dir,
        "best_weights": results.get("best", {}).get("best_weights"),
        "best_fitness": results.get("best", {}).get("fitness"),
        "generations": results.get("analysis", {}).get("generations", 0),
        "param_importance": results.get("analysis", {}).get("param_importance", {}),
        "top_runs": results.get("comparison", []),
        "success": bool(results.get("best")),
    }


@register_tool("pr-analyze")
def tool_pr_analyze(params: Dict[str, Any]) -> Dict[str, Any]:
    """Run PR/F1 curve analysis to find optimal confidence threshold.

    Args:
        model: Model path (required)
        data: Dataset YAML path (required)
        output_dir: Output directory (default: runs/pr_analysis)
        conf_thresholds: Custom confidence thresholds list (optional)
        iou_threshold: IoU threshold for matching (default: 0.5)
        num_thresholds: Number of auto-generated thresholds (default: 100)
        device: Device (default: '')

    Returns:
        PR analysis results with optimal threshold, F1, precision, recall, AUC-PR
    """
    from ..eval.pr_curve_analyzer import run_pr_analysis

    model = params.get("model")
    data = params.get("data")

    if not model:
        raise ValueError("Parameter 'model' is required for pr-analyze tool")
    if not data:
        raise ValueError("Parameter 'data' is required for pr-analyze tool")

    # Resolve project path
    output_dir = params.get("output_dir", "runs/pr_analysis")
    output_dir = _resolve_project_in_path(output_dir, params)
    if output_dir.startswith('./'):
        output_dir = str(Path(output_dir).resolve())

    conf_thresholds = params.get("conf_thresholds")
    if conf_thresholds is not None and not isinstance(conf_thresholds, list):
        conf_thresholds = None  # Will use auto-generation

    logger.info(f"Starting PR curve analysis: model={model}, data={data}")

    results = run_pr_analysis(
        model_path=model,
        data_yaml=data,
        output_dir=output_dir,
        conf_thresholds=conf_thresholds,
        iou_threshold=params.get("iou_threshold", 0.5),
        num_thresholds=params.get("num_thresholds", 100),
        device=params.get("device", ""),
    )

    logger.info("PR curve analysis completed")

    return {
        "model": model,
        "data": data,
        "output_dir": output_dir,
        "optimal_threshold": results.get("optimal", {}).get("threshold"),
        "optimal_f1": results.get("optimal", {}).get("f1"),
        "optimal_precision": results.get("optimal", {}).get("precision"),
        "optimal_recall": results.get("optimal", {}).get("recall"),
        "auc_pr": results.get("auc_pr"),
        "success": True,
    }


@register_tool("prune")
def tool_prune(params: Dict[str, Any]) -> Dict[str, Any]:
    """Run model pruning to compress the model.

    Args:
        model: Model path (required)
        data: Dataset YAML path (optional, for evaluation)
        output_dir: Output directory (default: runs/prune)
        method: Pruning method - l1, l2, bn_gamma (default: l1)
        amount: Pruning ratio 0.0-1.0 (default: 0.3)
        global_pruning: Use global pruning vs local (default: False)
        fine_tune: Fine-tune after pruning (default: False)
        fine_tune_epochs: Fine-tune epochs (default: 10)
        fine_tune_lr: Fine-tune learning rate (default: 0.0001)
        device: Device (default: '')

    Returns:
        Pruning results with pruned model path and metrics
    """
    from ..train.pruner import prune_model

    model = params.get("model")
    if not model:
        raise ValueError("Parameter 'model' is required for prune tool")

    # Resolve project path
    output_dir = params.get("output_dir", "runs/prune")
    output_dir = _resolve_project_in_path(output_dir, params)
    if output_dir.startswith('./'):
        output_dir = str(Path(output_dir).resolve())

    logger.info(f"Starting model pruning: model={model}, method={params.get('method', 'l1')}")

    results = prune_model(
        model=model,
        data=params.get("data", ""),
        output_dir=output_dir,
        method=params.get("method", "l1"),
        amount=params.get("amount", 0.3),
        global_pruning=params.get("global_pruning", False),
        fine_tune=params.get("fine_tune", False),
        fine_tune_epochs=params.get("fine_tune_epochs", 10),
        fine_tune_lr=params.get("fine_tune_lr", 0.0001),
        device=str(params.get("device", "")) or "0",
    )

    logger.info("Model pruning completed")

    return {
        "model": model,
        "output_dir": output_dir,
        "pruned_model": results.get("pruned_model"),
        "method": results.get("method"),
        "prune_ratio": results.get("actual_prune_ratio"),
        "pruned_channels": results.get("pruned_channels"),
        "total_channels": results.get("total_channels"),
        "mAP50": results.get("metrics", {}).get("mAP50"),
        "mAP50-95": results.get("metrics", {}).get("mAP50-95"),
        "success": bool(results.get("pruned_model")),
    }


@register_tool("verify-inference")
def tool_verify_inference(params: Dict[str, Any]) -> Dict[str, Any]:
    """Run inference on images and save annotated results.

    Params:
        model: Model path (required)
        images: Path to images directory or image file (required)
        data: Dataset YAML path for class names (optional)
        classes: List of class names (optional, overrides data)
        conf: Confidence threshold (default: 0.25)
        iou: NMS IoU threshold (default: 0.7)
        imgsz: Image size (default: 640)
        device: Device (default: 'cpu')
        output_dir: Output directory for annotated images (default: runs/verify_inference)
        save_txt: Save detection results to txt files (default: False)
        save_conf: Save confidence in txt (default: True)
        line_width: Line width for bounding boxes (default: 2)
        max_det: Maximum detections per image (default: 300)

    Returns:
        Inference results with annotated image paths
    """
    # TTA handling
    if params.get('tta'):
        from .tta_inference import TTAInference, TTAConfig
        scales = [float(s) for s in params.get('tta_scales', '0.8 1.0 1.2').split()]
        config = TTAConfig(
            model=params['model'],
            images=params['images'],
            output_dir=params.get('output_dir', './tta_results'),
            scales=scales,
            flip=params.get('tta_flip', True),
            conf=params.get('conf', 0.25),
            iou=params.get('iou', 0.7),
            wbf_iou=params.get('tta_wbf_iou', 0.5),
            device=params.get('device', 'cpu'),
            save_vis=True,
            save_txt=params.get('save_txt', False),
            save_conf=params.get('save_conf', True),
        )
        inference = TTAInference(config)
        return inference.run()

    from ultralytics import YOLO
    import cv2
    from pathlib import Path

    model_path = params.get("model")
    if not model_path:
        raise ValueError("Parameter 'model' is required for verify-inference tool")

    images = params.get("images")
    if not images:
        raise ValueError("Parameter 'images' is required for verify-inference tool")

    data = params.get("data")
    classes = params.get("classes", [])
    conf = params.get("conf", 0.25)
    iou = params.get("iou", 0.7)
    imgsz = params.get("imgsz", 640)
    device = params.get("device", "cpu")
    output_dir = params.get("output_dir", "runs/verify_inference")
    output_dir = _resolve_project_in_path(output_dir, params)
    if output_dir.startswith('./'):
        output_dir = str(Path(output_dir).resolve())
    save_txt = params.get("save_txt", False)
    save_conf = params.get("save_conf", True)
    line_width = params.get("line_width", 2)
    max_det = params.get("max_det", 300)

    # Load class names from data YAML if not provided
    if not classes and data:
        import yaml
        try:
            with open(data, 'r') as f:
                data_config = yaml.safe_load(f)
            names = data_config.get('names', {})
            if isinstance(names, dict):
                classes = [names[i] for i in sorted(names.keys())]
            elif isinstance(names, list):
                classes = names
        except Exception as e:
            logger.warning(f"Failed to load classes from {data}: {e}")

    # Load model
    logger.info(f"Loading model: {model_path}")
    model = YOLO(model_path)

    # Get class names from model if still not available
    if not classes and hasattr(model, 'names'):
        classes = list(model.names.values()) if isinstance(model.names, dict) else model.names

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Get image files
    images_path = Path(images)
    image_files = []
    if images_path.is_file():
        image_files = [images_path]
    elif images_path.is_dir():
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            image_files.extend(list(images_path.glob(ext)))
            image_files.extend(list(images_path.glob(ext.upper())))
        image_files = sorted(set(image_files))
    else:
        raise ValueError(f"Images path not found: {images}")

    logger.info(f"Found {len(image_files)} images")
    logger.info(f"Running inference with conf={conf}, iou={iou}, imgsz={imgsz}")

    # Color palette for classes
    np.random.seed(42)
    colors = {i: tuple(map(int, np.random.randint(0, 255, 3))) for i in range(len(classes) if classes else 100)}

    results_list = []
    annotated_count = 0

    for img_path in image_files:
        # Run inference
        results = model.predict(
            source=str(img_path),
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            device=device,
            max_det=max_det,
            verbose=False,
        )

        # Load image for drawing
        img = cv2.imread(str(img_path))
        if img is None:
            logger.warning(f"Failed to read image: {img_path}")
            continue

        h, w = img.shape[:2]

        # Process detections
        detections = []
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for box in boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                conf_score = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())

                x1, y1, x2, y2 = map(int, xyxy)
                class_name = classes[cls_id] if classes and cls_id < len(classes) else str(cls_id)

                # Draw bounding box
                color = colors.get(cls_id, (0, 255, 0))
                cv2.rectangle(img, (x1, y1), (x2, y2), color, line_width)

                # Draw label background
                label = f"{class_name} {conf_score:.2f}"
                (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                label_y = max(y1 - 10, label_h + 10)
                cv2.rectangle(img, (x1, label_y - label_h - 4), (x1 + label_w + 4, label_y + 4), color, -1)
                cv2.putText(img, label, (x1 + 2, label_y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                detections.append({
                    "class": class_name,
                    "confidence": conf_score,
                    "bbox": [x1, y1, x2, y2],
                })

        # Save annotated image
        output_img_path = output_path / img_path.name
        cv2.imwrite(str(output_img_path), img)
        annotated_count += 1

        # Save txt results if requested
        if save_txt and detections:
            txt_path = output_path / f"{img_path.stem}.txt"
            with open(txt_path, 'w') as f:
                for det in detections:
                    # YOLO format: class x_center y_center width height confidence
                    x1, y1, x2, y2 = det['bbox']
                    cx = ((x1 + x2) / 2) / w
                    cy = ((y1 + y2) / 2) / h
                    bw = (x2 - x1) / w
                    bh = (y2 - y1) / h
                    if save_conf:
                        f.write(f"{det['class']} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} {det['confidence']:.4f}\n")
                    else:
                        f.write(f"{det['class']} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

        results_list.append({
            "image": str(img_path),
            "output": str(output_img_path),
            "detections": len(detections),
            "results": detections,
        })

    logger.info(f"Annotated {annotated_count}/{len(image_files)} images")
    logger.info(f"Results saved to: {output_path}")

    return {
        "model": model_path,
        "images": images,
        "output_dir": str(output_path),
        "total_images": len(image_files),
        "annotated_count": annotated_count,
        "results": results_list,
        "success": annotated_count > 0,
    }


@register_tool("export")
def tool_export(params: Dict[str, Any]) -> Dict[str, Any]:
    """Export YOLO model.

    Params:
        model: Model path (required)
        format: Export format (default: onnx)
            Options: onnx, torchscript, engine, openvino, coreml, tflite, ncnn, mnn
        imgsz: Image size (default: 640)
        half: FP16 quantization (default: False)
        int8: INT8 quantization (default: False)
        dynamic: Dynamic input尺寸 (default: False)
        simplify: Simplify ONNX model (default: True)
        opset: ONNX opset version (default: 12)
        workspace: TensorRT workspace size in GiB (default: 4.0)
        nms: Add NMS post-processing (default: False)
        batch: Batch size (default: 1)
        device: Device (default: '0')
        keras: Keras format (default: False)
        optimize: Mobile optimization (default: False)
        fraction: INT8 calibration data fraction (default: 1.0)
        data: Dataset config for INT8 calibration (default: 'coco8.yaml')

    Returns:
        Export results
    """
    from ..export.exporter import ModelExporter, ExportConfig

    model = params.get("model")
    if not model:
        raise ValueError("Parameter 'model' is required for export tool")

    # Resolve ${project} in model path
    model = _resolve_project_in_path(model, params)

    logger.info(f"Exporting model: {model}, format: {params.get('format', 'onnx')}")

    config = ExportConfig(
        model=model,
        format=params.get("format", "onnx"),
        imgsz=params.get("imgsz", 640),
        half=params.get("half", False),
        int8=params.get("int8", False),
        dynamic=params.get("dynamic", False),
        simplify=params.get("simplify", True),
        opset=params.get("opset", 12),
        workspace=params.get("workspace", 4.0),
        nms=params.get("nms", False),
        batch=params.get("batch", 1),
        device=params.get("device", "0"),
        keras=params.get("keras", False),
        optimize=params.get("optimize", False),
        fraction=params.get("fraction", 1.0),
        data=params.get("data", "coco8.yaml"),
    )

    exporter = ModelExporter(model, config)
    export_path = exporter.export()

    return {
        "model": model,
        "format": config.format,
        "export_path": export_path,
        "success": Path(export_path).exists() if export_path else False,
    }


@register_tool("best-model-select")
def tool_best_model_select(params: Dict[str, Any]) -> Dict[str, Any]:
    """最佳模型选择"""
    from .best_model_selector import BestModelSelector, BestModelSelectorConfig

    model = params.get("model")
    data = params.get("data")

    if not model:
        raise ValueError("Parameter 'model' is required for best-model-select")
    if not data:
        raise ValueError("Parameter 'data' is required for best-model-select")

    config = BestModelSelectorConfig(
        model=model,
        data=data,
        metric=params.get("metric", "fitness"),
        output=params.get("output"),
        device=str(params.get("device", "0")),
    )

    selector = BestModelSelector(config)
    result = selector.select()

    return result


@register_tool("tta-inference")
def tool_tta_inference(params: Dict[str, Any]) -> Dict[str, Any]:
    """TTA 推理"""
    from .tta_inference import TTAInference, TTAConfig

    model = params.get("model")
    images = params.get("images")

    if not model:
        raise ValueError("Parameter 'model' is required for tta-inference")
    if not images:
        raise ValueError("Parameter 'images' is required for tta-inference")

    scales = params.get("scales", [0.8, 1.0, 1.2])
    if isinstance(scales, str):
        scales = [float(s) for s in scales.split()]

    config = TTAConfig(
        model=model,
        images=images,
        output_dir=params.get("output", "./tta_results"),
        scales=scales,
        flip=params.get("flip", True),
        conf=params.get("conf", 0.25),
        iou=params.get("iou", 0.7),
        wbf_iou=params.get("wbf_iou", 0.5),
        device=str(params.get("device", "cpu")),
        save_vis=params.get("save_vis", True),
        save_txt=params.get("save_txt", False),
        save_conf=params.get("save_conf", True),
    )

    inference = TTAInference(config)
    return inference.run()


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
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview pipeline without executing"
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

    # Load config
    executor = PipelineExecutor(args.config)
    executor.config.output_dir = args.output

    # Dry run - just preview
    if args.dry_run:
        print("\n" + "=" * 60)
        print("DRY RUN - Pipeline Preview")
        print("=" * 60)
        print(f"Pipeline: {executor.config.name}")
        print(f"Description: {executor.config.description}")
        print(f"Total Stages: {len(executor.config.stages)}")
        if executor.config.global_params:
            print(f"Global params: {executor.config.global_params}")
        print()
        for i, stage in enumerate(executor.config.stages):
            # Merge global_params with stage.params (global as defaults, stage overrides)
            merged_params = {**executor.config.global_params, **stage.params}
            # Resolve ${var} references in params for display
            resolved_params = _resolve_var_refs(merged_params, {**executor.config.global_params, **merged_params}, executor.config.global_params)
            print(f"  Stage {i+1}: {stage.name}")
            print(f"    Tool: {stage.tool}")
            print(f"    Merged params: {resolved_params}")
            print(f"    Enabled: {stage.enabled}")
            if stage.condition:
                print(f"    Condition: {stage.condition}")
            print()
        print("=" * 60)
        print("To execute this pipeline, run without --dry-run")
        print("=" * 60)
        return

    # Execute pipeline
    try:
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


def verify_inference_main():
    """CLI entry point for verify-inference tool."""
    parser = argparse.ArgumentParser(
        description="YOLO Verify Inference - Run inference and annotate images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run inference on a directory of images
  yolo-verify-inference --model best.pt --images ./test_images

  # Run with specific confidence threshold
  yolo-verify-inference --model best.pt --images ./test_images --conf 0.5

  # Run on a single image
  yolo-verify-inference --model best.pt --images test.jpg --output ./results

  # Run with class names from dataset YAML
  yolo-verify-inference --model best.pt --images ./test_images --data dataset.yaml
        """
    )

    parser.add_argument("--model", type=str, required=True, help="Path to trained model (.pt)")
    parser.add_argument("--images", type=str, required=True, help="Path to images directory or image file")
    parser.add_argument("--data", type=str, help="Dataset YAML for class names")
    parser.add_argument("--classes", type=str, nargs="+", help="List of class names (overrides --data)")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (default: 0.25)")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold (default: 0.7)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size (default: 640)")
    parser.add_argument("--device", type=str, default="cpu", help="Device (default: cpu)")
    parser.add_argument("--output", type=str, default="runs/verify_inference", help="Output directory")
    parser.add_argument("--save-txt", action="store_true", help="Save detection results to txt files")
    parser.add_argument("--save-conf", action="store_true", default=True, help="Save confidence in txt")
    parser.add_argument("--line-width", type=int, default=2, help="Bounding box line width (default: 2)")
    parser.add_argument("--max-det", type=int, default=300, help="Max detections per image (default: 300)")
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Logging level")
    parser.add_argument("--tta", action="store_true", help="启用 TTA 增强")
    parser.add_argument("--tta-scales", type=str, default="0.8 1.0 1.2", help="TTA 尺度列表 (空格分隔)")
    parser.add_argument("--tta-flip", action="store_true", default=True, help="TTA 启用水平翻转")
    parser.add_argument("--tta-wbf-iou", type=float, default=0.5, help="TTA WBF 融合 IoU 阈值")

    args = parser.parse_args()

    # Configure logging
    import logging as log_module
    log_level = getattr(log_module, args.log_level.upper(), log_module.INFO)
    log_module.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    params = {
        "model": args.model,
        "images": args.images,
        "data": args.data,
        "classes": args.classes,
        "conf": args.conf,
        "iou": args.iou,
        "imgsz": args.imgsz,
        "device": args.device,
        "output_dir": args.output,
        "save_txt": args.save_txt,
        "save_conf": args.save_conf,
        "line_width": args.line_width,
        "max_det": args.max_det,
        "tta": args.tta,
        "tta_scales": args.tta_scales,
        "tta_flip": args.tta_flip,
        "tta_wbf_iou": args.tta_wbf_iou,
    }

    result = tool_verify_inference(params)

    # Print summary
    print("\n" + "=" * 60)
    print("Verify Inference Results")
    print("=" * 60)
    print(f"Model: {result['model']}")
    print(f"Images: {result['total_images']}")
    print(f"Annotated: {result['annotated_count']}")
    print(f"Output: {result['output_dir']}")
    print(f"Success: {result['success']}")
    print("=" * 60)

    return 0 if result['success'] else 1


if __name__ == "__main__":
    main()
