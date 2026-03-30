"""
最佳模型选择工具 - 自动对比 best.pt 和 last.pt，按指定指标选择最佳模型
"""

import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from ultralytics import YOLO

logger = logging.getLogger("yolo_toolchain.best_model_selector")


@dataclass
class BestModelSelectorConfig:
    """最佳模型选择配置"""

    model: str  # 模型目录或 .pt 文件路径
    data: str  # 数据集 YAML
    metric: str = "fitness"  # 选择指标
    output: Optional[str] = None  # 输出路径（Pipeline params 用 output）
    device: str = "0"  # 评估设备


class BestModelSelector:
    """最佳模型选择器"""

    # 支持的指标映射到 YOLO val 结果的属性
    METRIC_MAP = {
        "mAP50": lambda r: r.box.map50,
        "mAP50-95": lambda r: r.box.map,
        "recall": lambda r: r.box.mr,
        "precision": lambda r: r.box.mp,
        "fitness": lambda r: r.box.fitness(),
    }

    def __init__(self, config: BestModelSelectorConfig):
        self.config = config
        self.results = {}

    def _resolve_weights_dir(self, model: str) -> Path:
        """解析模型路径，返回 weights 目录"""
        model_path = Path(model)
        if model_path.is_file():
            return model_path.parent
        elif model_path.is_dir():
            if (model_path / "weights").exists():
                return model_path / "weights"
            return model_path
        else:
            # 尝试作为目录路径
            return model_path

    def _get_checkpoint_path(self, weights_dir: Path, name: str) -> Optional[Path]:
        """获取检查点路径"""
        path = weights_dir / name
        if path.exists():
            return path
        # 尝试 glob
        matches = list(weights_dir.glob(f"*/weights/{name}"))
        if matches:
            return matches[0]
        return None

    def _evaluate_model(self, model_path: str) -> Dict[str, float]:
        """评估单个模型"""
        logger.info(f"评估模型: {model_path}")
        model = YOLO(model_path)
        results = model.val(data=self.config.data, verbose=False)

        metric_fn = self.METRIC_MAP.get(self.config.metric, self.METRIC_MAP["fitness"])
        metric_value = metric_fn(results)

        return {
            "mAP50": float(results.box.map50),
            "mAP50-95": float(results.box.map),
            "recall": float(results.box.mr),
            "precision": float(results.box.mp),
            "fitness": float(results.box.fitness()),
            "selected_metric": metric_value,
        }

    def select(self) -> Dict[str, Any]:
        """执行模型选择"""
        weights_dir = self._resolve_weights_dir(self.config.model)

        # 检查 best.pt 和 last.pt
        best_path = self._get_checkpoint_path(weights_dir, "best.pt")
        last_path = self._get_checkpoint_path(weights_dir, "last.pt")

        results = {}

        if best_path:
            results["best.pt"] = self._evaluate_model(str(best_path))
            results["best.pt"]["path"] = str(best_path)
        else:
            logger.warning("未找到 best.pt")

        if last_path:
            results["last.pt"] = self._evaluate_model(str(last_path))
            results["last.pt"]["path"] = str(last_path)
        else:
            logger.warning("未找到 last.pt")

        if not results:
            raise ValueError("未找到 best.pt 或 last.pt")

        # 按指标选择
        selected_model = None
        selected_value = float("-inf")

        for name, result in results.items():
            if result["selected_metric"] > selected_value:
                selected_value = result["selected_metric"]
                selected_model = name

        self.results = results

        output = {
            "selected": selected_model,
            "selected_path": results[selected_model]["path"],
            "selected_metric_value": float(selected_value),
            "all_results": results,
        }

        # 写入 output 文件
        if self.config.output:
            output_path = Path(self.config.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                f.write(output["selected_path"])
            logger.info(f"已写入选定模型路径到: {output_path}")

        return output


def main():
    parser = argparse.ArgumentParser(
        description="YOLO 最佳模型选择工具 - 自动对比 best.pt 和 last.pt"
    )
    parser.add_argument(
        "--model", type=str, required=True, help="模型目录（含 weights/ 文件夹）或直接指定 .pt 文件"
    )
    parser.add_argument("--data", type=str, required=True, help="数据集 YAML 配置文件")
    parser.add_argument(
        "--metric",
        type=str,
        default="fitness",
        choices=["mAP50", "mAP50-95", "recall", "precision", "fitness"],
        help="选择指标 (默认: fitness)",
    )
    parser.add_argument("--output", type=str, default=None, help="输出路径（写入选定模型路径）")
    parser.add_argument("--device", type=str, default="0", help="评估设备 (默认: 0)")

    args = parser.parse_args()

    # 配置日志
    import logging as log_module

    log_module.basicConfig(
        level=log_module.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    config = BestModelSelectorConfig(
        model=args.model,
        data=args.data,
        metric=args.metric,
        output=args.output,
        device=args.device,
    )

    selector = BestModelSelector(config)
    result = selector.select()

    # 打印结果
    print("\n" + "=" * 60)
    print("模型选择结果")
    print("=" * 60)
    print(f"模型目录: {config.model}\n")

    for name, res in result["all_results"].items():
        print(f"{name}:")
        print(f"  - mAP50: {res['mAP50']:.4f}")
        print(f"  - mAP50-95: {res['mAP50-95']:.4f}")
        print(f"  - Recall: {res['recall']:.4f}")
        print(f"  - Precision: {res['precision']:.4f}")
        print()

    print(f"按 [{config.metric}] 指标，最佳模型: {result['selected']}")
    print(f"输出路径: {result['selected_path']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
