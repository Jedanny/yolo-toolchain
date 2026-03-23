"""
训练模块 - 增量训练器
支持在已有模型基础上添加新类别
"""

import os
import argparse
import yaml
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Union
from dataclasses import dataclass, asdict
import json

from ultralytics import YOLO


@dataclass
class IncrementalTrainConfig:
    """增量训练配置"""
    # 模型配置
    model: str = "yolo11n.pt"  # 预训练或已训练模型
    data: str = "data.yaml"    # 更新后的数据集配置
    epochs: int = 50
    imgsz: int = 640
    batch: int = 16
    device: str = "0"

    # 冻结配置 - 增量训练建议冻结骨干网络
    freeze_backbone: bool = True
    freeze_layers: List[int] = None

    # 优化器配置
    optimizer: str = "auto"
    lr0: float = 0.001  # 增量训练使用较小的学习率
    lrf: float = 0.01
    momentum: float = 0.937
    weight_decay: float = 0.0005
    cos_lr: bool = False
    warmup_epochs: float = 3.0

    # 损失权重
    box: float = 7.5
    cls: float = 0.5
    dfl: float = 1.5

    # 数据增强 - 增量训练建议增强以缓解类别不平衡
    hsv_h: float = 0.015
    hsv_s: float = 0.7
    hsv_v: float = 0.4
    degrees: float = 5.0  # 稍微增加旋转
    translate: float = 0.1
    scale: float = 0.5
    shear: float = 2.0  # 稍微增加剪切
    perspective: float = 0.0
    flipud: float = 0.0
    fliplr: float = 0.5
    mosaic: float = 1.0
    mixup: float = 0.3
    cutmix: float = 0.1
    close_mosaic: int = 10

    # 其他配置
    patience: int = 50
    save: bool = True
    save_period: int = -1
    cache: Union[bool, str] = False
    pretrained: Union[bool, str] = True
    resume: bool = False
    amp: bool = True
    plots: bool = True
    val: bool = True
    close_mosaic_epochs: int = 10

    # 输出配置
    project: str = "runs/train"
    name: str = "incremental_train"
    exist_ok: bool = False
    workers: int = 8
    verbose: bool = True
    seed: int = 0

    def __post_init__(self):
        if self.freeze_layers is None and self.freeze_backbone:
            # 默认冻结前10层 (骨干网络)
            self.freeze_layers = list(range(10))

    def to_dict(self) -> Dict:
        result = {}
        for k, v in asdict(self).items():
            if v is not None:
                result[k] = v
        return result


class IncrementalTrainer:
    """增量训练器"""

    def __init__(self, config: Union[Dict, IncrementalTrainConfig, str] = None):
        if config is None:
            self.config = IncrementalTrainConfig()
        elif isinstance(config, dict):
            self.config = IncrementalTrainConfig(**config)
        elif isinstance(config, IncrementalTrainConfig):
            self.config = config
        elif isinstance(config, str):
            with open(config, 'r') as f:
                self.config = IncrementalTrainConfig(**yaml.safe_load(f))

        self.model = None
        self.original_classes = []
        self.new_classes = []

    def analyze_model_classes(self, model_path: str) -> List[str]:
        """分析模型已有的类别"""
        model = YOLO(model_path)
        if hasattr(model, 'names') and model.names:
            self.original_classes = list(model.names.values())
        return self.original_classes

    def prepare_incremental_data(
        self,
        original_data_yaml: str,
        new_classes: List[str],
        merged_output: str = "incremental_data.yaml"
    ) -> str:
        """准备增量训练数据配置"""
        # 读取原始配置
        with open(original_data_yaml, 'r') as f:
            original_config = yaml.safe_load(f)

        # 合并类别
        old_classes = list(original_config.get('names', {}).values())
        all_classes = old_classes + new_classes

        # 创建新的类别映射
        class_mapping = {i: name for i, name in enumerate(all_classes)}

        # 更新配置
        incremental_config = {
            'path': original_config.get('path', '.'),
            'train': original_config.get('train', 'images/train'),
            'val': original_config.get('val', 'images/val'),
            'test': original_config.get('test', 'images/test'),
            'nc': len(all_classes),
            'names': class_mapping
        }

        # 保存配置
        with open(merged_output, 'w') as f:
            for key, value in incremental_config.items():
                if isinstance(value, dict):
                    f.write(f"{key}:\n")
                    for k, v in value.items():
                        f.write(f"  {k}: {v}\n")
                elif isinstance(value, list):
                    f.write(f"{key}:\n")
                    for item in value:
                        f.write(f"  - {item}\n")
                else:
                    f.write(f"{key}: {value}\n")

        self.original_classes = old_classes
        self.new_classes = new_classes

        return merged_output

    def train(self, model_path: Optional[str] = None) -> dict:
        """执行增量训练"""
        if model_path:
            self.config.model = model_path

        if self.model is None:
            self.model = YOLO(self.config.model)

        # 准备训练参数
        train_params = {
            'data': self.config.data,
            'epochs': self.config.epochs,
            'imgsz': self.config.imgsz,
            'batch': self.config.batch,
            'device': self.config.device,
            'optimizer': self.config.optimizer,
            'lr0': self.config.lr0,
            'lrf': self.config.lrf,
            'momentum': self.config.momentum,
            'weight_decay': self.config.weight_decay,
            'cos_lr': self.config.cos_lr,
            'warmup_epochs': self.config.warmup_epochs,
            'box': self.config.box,
            'cls': self.config.cls,
            'dfl': self.config.dfl,
            'hsv_h': self.config.hsv_h,
            'hsv_s': self.config.hsv_s,
            'hsv_v': self.config.hsv_v,
            'degrees': self.config.degrees,
            'translate': self.config.translate,
            'scale': self.config.scale,
            'shear': self.config.shear,
            'perspective': self.config.perspective,
            'flipud': self.config.flipud,
            'fliplr': self.config.fliplr,
            'mosaic': self.config.mosaic,
            'mixup': self.config.mixup,
            'cutmix': self.config.cutmix,
            'close_mosaic': self.config.close_mosaic,
            'patience': self.config.patience,
            'save': self.config.save,
            'save_period': self.config.save_period,
            'cache': self.config.cache,
            'pretrained': self.config.pretrained,
            'resume': self.config.resume,
            'amp': self.config.amp,
            'plots': self.config.plots,
            'val': self.config.val,
            'project': self.config.project,
            'name': self.config.name,
            'exist_ok': self.config.exist_ok,
            'workers': self.config.workers,
            'verbose': self.config.verbose,
            'seed': self.config.seed,
        }

        # 添加冻结层
        if self.config.freeze_layers:
            train_params['freeze'] = self.config.freeze_layers

        print(f"开始增量训练...")
        print(f"原始类别: {self.original_classes}")
        print(f"新增类别: {self.new_classes}")
        print(f"冻结层: {self.config.freeze_layers}")

        results = self.model.train(**train_params)
        return results

    def validate_old_classes(
        self,
        model_path: str,
        data_yaml: str,
        old_class_names: List[str]
    ) -> Dict:
        """验证旧类别性能是否下降"""
        model = YOLO(model_path)
        metrics = model.val(data=data_yaml)

        # 获取各类别mAP
        per_class_maps = metrics.box.maps if hasattr(metrics.box, 'maps') else []

        results = {
            'overall': {
                'mAP50': metrics.box.map50,
                'mAP50-95': metrics.box.map
            },
            'per_class': {}
        }

        for i, class_name in enumerate(old_class_names):
            if i < len(per_class_maps):
                results['per_class'][class_name] = per_class_maps[i]

        return results


def main():
    parser = argparse.ArgumentParser(description='YOLO增量训练工具')
    parser.add_argument('--model', type=str, required=True, help='已有模型路径')
    parser.add_argument('--data', type=str, required=True, help='更新后的数据集配置')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--imgsz', type=int, default=640, help='图像尺寸')
    parser.add_argument('--batch', type=int, default=16, help='批次大小')
    parser.add_argument('--device', type=str, default='0', help='设备')
    parser.add_argument('--lr0', type=float, default=0.001, help='初始学习率')
    parser.add_argument('--freeze_backbone', action='store_true', default=True, help='冻结骨干网络')
    parser.add_argument('--no_freeze', action='store_true', help='不冻结任何层')
    parser.add_argument('--mosaic', type=float, default=1.0, help='Mosaic增强概率')
    parser.add_argument('--mixup', type=float, default=0.3, help='Mixup增强概率')
    parser.add_argument('--project', type=str, default='runs/train', help='输出项目目录')
    parser.add_argument('--name', type=str, default='incremental_train', help='实验名称')

    args = parser.parse_args()

    config = IncrementalTrainConfig(
        model=args.model,
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        lr0=args.lr0,
        freeze_backbone=not args.no_freeze,
        mosaic=args.mosaic,
        mixup=args.mixup,
        project=args.project,
        name=args.name,
    )

    trainer = IncrementalTrainer(config)

    # 分析原始模型类别
    trainer.analyze_model_classes(args.model)
    print(f"模型已有类别: {trainer.original_classes}")

    # 开始训练
    results = trainer.train()
    print("增量训练完成!")


if __name__ == '__main__':
    main()
