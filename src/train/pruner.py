"""
Model Pruner Module

支持 YOLO 模型剪枝，包括：
- 幅度剪枝 (Magnitude Pruning)
- 网络瘦身 (Network Slimming) - 基于 BN gamma 值
- 剪枝后微调

参考: Network Slimming (Li et al., 2017) - 使用 BatchNorm gamma 值评估通道重要性
"""

import argparse
import logging
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict

import torch
import torch.nn as nn
from ultralytics import YOLO

logger = logging.getLogger("yolo_toolchain.pruner")


@dataclass
class PrunerConfig:
    """剪枝配置"""
    model: str = ""                      # 模型路径
    data: str = ""                       # 数据集 YAML (剪枝时需要验证)
    output_dir: str = "runs/prune"       # 输出目录
    method: str = "l1"                   # 剪枝方法: l1, l2, bn_gamma
    amount: float = 0.3                  # 剪枝比例 (0.0-1.0)
    global_pruning: bool = False         # 全局剪枝 vs 本地剪枝
    ignore_layers: List[str] = field(default_factory=lambda: [])  # 跳过的层
    fine_tune_epochs: int = 10           # 剪枝后微调轮数
    fine_tune_lr: float = 0.0001         # 微调学习率
    preserve_filenames: bool = True      # 保持文件名


class ModelPruner:
    """YOLO 模型剪枝器"""

    def __init__(self, config: PrunerConfig):
        self.config = config
        self.model = None
        self.pruned_model = None

    def _get_layer_name(self, layer: nn.Module) -> str:
        """获取层的名称"""
        name = []
        for m in self.model.model.modules():
            if m is layer:
                break
            name.append(str(id(m)))
        return ".".join(name)

    def _get_conv_layers(self) -> List[Tuple[str, nn.Conv2d]]:
        """获取所有卷积层"""
        conv_layers = []
        for name, module in self.model.model.named_modules():
            if isinstance(module, nn.Conv2d):
                conv_layers.append((name, module))
        return conv_layers

    def _get_bn_layers(self) -> List[Tuple[str, nn.BatchNorm2d]]:
        """获取所有 BatchNorm 层 (用于 network slimming)"""
        bn_layers = []
        for name, module in self.model.model.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                bn_layers.append((name, module))
        return bn_layers

    def _compute_l1_norm(self, conv: nn.Conv2d) -> torch.Tensor:
        """计算卷积核的 L1 范数"""
        # weight shape: (out_channels, in_channels, kH, kW)
        return torch.sum(torch.abs(conv.weight), dim=(1, 2, 3))

    def _compute_l2_norm(self, conv: nn.Conv2d) -> torch.Tensor:
        """计算卷积核的 L2 范数"""
        return torch.sqrt(torch.sum(conv.weight ** 2, dim=(1, 2, 3)))

    def _compute_bn_gamma(self, bn: nn.BatchNorm2d) -> torch.Tensor:
        """获取 BN 层的 gamma (scale) 值"""
        return torch.abs(bn.weight)

    def _get_channel_importance(self, method: str = "l1") -> Dict[str, torch.Tensor]:
        """计算各层通道重要性

        Returns:
            {layer_name: importance_scores}
        """
        importance = {}

        if method == "bn_gamma":
            # Network Slimming: 使用 BN gamma 值
            for name, bn in self._get_bn_layers():
                if name not in self.config.ignore_layers:
                    importance[name] = self._compute_bn_gamma(bn)
        else:
            # Magnitude pruning: 使用卷积核范数
            for name, conv in self._get_conv_layers():
                if name not in self.config.ignore_layers:
                    if method == "l1":
                        importance[name] = self._compute_l1_norm(conv)
                    elif method == "l2":
                        importance[name] = self._compute_l2_norm(conv)
                    else:
                        importance[name] = self._compute_l1_norm(conv)

        return importance

    def _compute_threshold(self, importance: Dict[str, torch.Tensor], amount: float) -> float:
        """计算全局阈值 (用于全局剪枝)"""
        all_importance = torch.cat([v.flatten() for v in importance.values()])
        # 计算要保留的比例
        keep_ratio = 1.0 - amount
        n_to_keep = int(len(all_importance) * keep_ratio)
        if n_to_keep < 1:
            n_to_keep = 1
        threshold, _ = torch.kthvalue(all_importance, n_to_keep)
        return threshold.item()

    def _get_channels_to_prune(
        self, importance: Dict[str, torch.Tensor], amount: float
    ) -> Dict[str, torch.Tensor]:
        """确定要剪枝的通道

        Returns:
            {layer_name: channels_to_keep_mask}
        """
        if self.config.global_pruning:
            # 全局阈值
            threshold = self._compute_threshold(importance, amount)
            channels_to_keep = {}
            for name, imp in importance.items():
                channels_to_keep[name] = (imp >= threshold).nonzero(as_tuple=True)[0]
        else:
            # 本地阈值 - 每个层独立计算
            channels_to_keep = {}
            for name, imp in importance.items():
                n_to_keep = max(1, int(len(imp) * (1 - amount)))
                if n_to_keep >= len(imp):
                    # 保留所有通道
                    channels_to_keep[name] = torch.arange(len(imp))
                else:
                    _, indices = torch.topk(imp, n_to_keep)
                    channels_to_keep[name] = indices.sort()[0]

        return channels_to_keep

    def prune(self) -> Dict[str, Any]:
        """执行模型剪枝

        Returns:
            剪枝结果信息
        """
        # 加载模型
        self.model = YOLO(self.config.model)
        logger.info(f"Loaded model: {self.config.model}")

        # 计算通道重要性
        importance = self._get_channel_importance(self.config.method)
        logger.info(f"Computed importance for {len(importance)} layers")

        # 确定要保留的通道
        channels_to_keep = self._get_channels_to_prune(importance, self.config.amount)

        # 统计剪枝信息
        total_channels = 0
        pruned_channels = 0
        for name, imp in importance.items():
            total_channels += len(imp)
            pruned_channels += len(imp) - len(channels_to_keep[name])

        prune_ratio = pruned_channels / total_channels if total_channels > 0 else 0
        logger.info(f"Pruning: {pruned_channels}/{total_channels} channels ({prune_ratio:.1%})")

        # 创建剪枝后的模型
        self.pruned_model = self._create_pruned_model(channels_to_keep)

        # 评估剪枝后模型
        metrics = self._evaluate_model()

        # 保存剪枝后模型
        save_path = self._save_pruned_model()

        return {
            "original_model": self.config.model,
            "pruned_model": save_path,
            "method": self.config.method,
            "prune_amount": self.config.amount,
            "total_channels": total_channels,
            "pruned_channels": pruned_channels,
            "actual_prune_ratio": prune_ratio,
            "metrics": metrics,
        }

    def _create_pruned_model(
        self, channels_to_keep: Dict[str, torch.Tensor]
    ) -> nn.Module:
        """创建剪枝后的模型"""
        import copy
        model = copy.deepcopy(self.model.model)

        # 逐层剪枝
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                # 获取该层要保留的通道
                if name in channels_to_keep:
                    keep_idx = channels_to_keep[name]
                    # 剪枝输出通道 (out_channels)
                    module.weight.data = module.weight.data[keep_idx].clone()
                    if module.bias is not None:
                        module.bias.data = module.bias.data[keep_idx].clone()
                    module.out_channels = len(keep_idx)

                    # 更新下一层的输入通道
                    parent_name = ".".join(name.split(".")[:-1])
                    for child_name, child_module in model.named_modules():
                        if child_name.startswith(parent_name):
                            if isinstance(child_module, nn.Conv2d) and child_name != name:
                                # 这是一个后续层，需要调整其 in_channels
                                pass  # 复杂逻辑需单独处理

            elif isinstance(module, nn.BatchNorm2d):
                if name in channels_to_keep:
                    keep_idx = channels_to_keep[name]
                    module.weight.data = module.weight.data[keep_idx].clone()
                    module.bias.data = module.bias.data[keep_idx].clone()
                    module.running_mean = module.running_mean[keep_idx].clone()
                    module.running_var = module.running_var[keep_idx].clone()
                    module.num_features = len(keep_idx)

        return model

    def _evaluate_model(self) -> Dict[str, float]:
        """评估剪枝后模型的精度"""
        if not self.pruned_model or not self.config.data:
            return {}

        try:
            # 使用剪枝后的模型进行验证
            self.model.model = self.pruned_model
            results = self.model.val(data=self.config.data, verbose=False)
            metrics = {
                "mAP50": float(results.box.map50) if hasattr(results.box, 'map50') else 0.0,
                "mAP50-95": float(results.box.map) if hasattr(results.box, 'map') else 0.0,
            }
            logger.info(f"Pruned model metrics: {metrics}")
            return metrics
        except Exception as e:
            logger.warning(f"Evaluation failed: {e}")
            return {}

    def _save_pruned_model(self) -> str:
        """保存剪枝后的模型"""
        output_path = Path(self.config.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        save_path = output_path / "pruned.pt"
        torch.save({
            "model": self.pruned_model.state_dict(),
            "method": self.config.method,
            "prune_amount": self.config.amount,
        }, save_path)

        logger.info(f"Saved pruned model to: {save_path}")
        return str(save_path)

    def fine_tune(self, pruned_model_path: str) -> Dict[str, Any]:
        """剪枝后微调

        Args:
            pruned_model_path: 剪枝后模型路径

        Returns:
            微调结果
        """
        logger.info(f"Starting fine-tuning for {self.config.fine_tune_epochs} epochs")

        model = YOLO(pruned_model_path)

        # 微调训练
        results = model.train(
            data=self.config.data,
            epochs=self.config.fine_tune_epochs,
            lr0=self.config.fine_tune_lr,
            device=self.config.device if hasattr(self.config, 'device') else '0',
            verbose=False,
        )

        return {
            "fine_tuned_model": results.save_dir,
            "epochs": self.config.fine_tune_epochs,
            "final_lr": self.config.fine_tune_lr,
        }


def prune_model(
    model: str,
    data: str = "",
    output_dir: str = "runs/prune",
    method: str = "l1",
    amount: float = 0.3,
    global_pruning: bool = False,
    fine_tune: bool = False,
    fine_tune_epochs: int = 10,
    fine_tune_lr: float = 0.0001,
    device: str = "0",
) -> Dict[str, Any]:
    """模型剪枝入口函数

    Args:
        model: 模型路径
        data: 数据集 YAML (验证用)
        output_dir: 输出目录
        method: 剪枝方法 (l1, l2, bn_gamma)
        amount: 剪枝比例 (0.0-1.0)
        global_pruning: 是否全局剪枝
        fine_tune: 是否剪枝后微调
        fine_tune_epochs: 微调轮数
        fine_tune_lr: 微调学习率
        device: 设备

    Returns:
        剪枝结果
    """
    config = PrunerConfig(
        model=model,
        data=data,
        output_dir=output_dir,
        method=method,
        amount=amount,
        global_pruning=global_pruning,
        fine_tune_epochs=fine_tune_epochs if fine_tune else 0,
        fine_tune_lr=fine_tune_lr,
    )

    pruner = ModelPruner(config)
    result = pruner.prune()

    if fine_tune and result.get("pruned_model"):
        ft_result = pruner.fine_tune(result["pruned_model"])
        result["fine_tune"] = ft_result

    return result


def main():
    """CLI 入口"""
    parser = argparse.ArgumentParser(description="YOLO 模型剪枝工具")
    parser.add_argument("--model", type=str, required=True, help="模型路径")
    parser.add_argument("--data", type=str, default="", help="数据集 YAML")
    parser.add_argument("--output-dir", type=str, default="runs/prune", help="输出目录")
    parser.add_argument("--method", type=str, default="l1",
                        choices=["l1", "l2", "bn_gamma"], help="剪枝方法")
    parser.add_argument("--amount", type=float, default=0.3, help="剪枝比例 (0.0-1.0)")
    parser.add_argument("--global-pruning", action="store_true", help="使用全局剪枝")
    parser.add_argument("--fine-tune", action="store_true", help="剪枝后微调")
    parser.add_argument("--fine-tune-epochs", type=int, default=10, help="微调轮数")
    parser.add_argument("--fine-tune-lr", type=float, default=0.0001, help="微调学习率")
    parser.add_argument("--device", type=str, default="0", help="设备 (如 '0', 'cpu', 'mps')")

    args = parser.parse_args()

    import logging as log_module
    log_module.basicConfig(
        level=log_module.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    )

    results = prune_model(
        model=args.model,
        data=args.data,
        output_dir=args.output_dir,
        method=args.method,
        amount=args.amount,
        global_pruning=args.global_pruning,
        fine_tune=args.fine_tune,
        fine_tune_epochs=args.fine_tune_epochs,
        fine_tune_lr=args.fine_tune_lr,
        device=args.device,
    )

    print("\n" + "=" * 60)
    print("模型剪枝结果")
    print("=" * 60)
    print(f"原始模型: {results['original_model']}")
    print(f"剪枝后模型: {results['pruned_model']}")
    print(f"剪枝方法: {results['method']}")
    print(f"剪枝比例: {results['actual_prune_ratio']:.1%}")
    print(f"剪枝通道: {results['pruned_channels']}/{results['total_channels']}")
    if results.get("metrics"):
        print(f"mAP50: {results['metrics'].get('mAP50', 0):.4f}")
        print(f"mAP50-95: {results['metrics'].get('mAP50-95', 0):.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
