"""
训练模块 - 冻结训练策略
支持冻结骨干网络进行高效微调
"""

import os
import argparse
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Union
from dataclasses import dataclass, asdict

from ultralytics import YOLO


@dataclass
class FreezeTrainConfig:
    """冻结训练配置"""
    # 模型配置
    model: str = "yolo11n.pt"
    data: str = "data.yaml"
    epochs: int = 100
    imgsz: int = 640
    batch: int = 16
    device: str = "0"

    # 冻结配置
    freeze_layers: List[int] = None  # 默认为[0-9]冻结前10层

    # 优化器配置
    optimizer: str = "auto"
    lr0: float = 0.01
    lrf: float = 0.01
    momentum: float = 0.937
    weight_decay: float = 0.0005
    cos_lr: bool = False
    warmup_epochs: float = 3.0
    warmup_momentum: float = 0.8
    warmup_bias_lr: float = 0.1

    # 损失权重
    box: float = 7.5
    cls: float = 0.5
    dfl: float = 1.5

    # 数据增强
    hsv_h: float = 0.015
    hsv_s: float = 0.7
    hsv_v: float = 0.4
    degrees: float = 0.0
    translate: float = 0.1
    scale: float = 0.5
    shear: float = 0.0
    perspective: float = 0.0
    flipud: float = 0.0
    fliplr: float = 0.5
    mosaic: float = 1.0
    mixup: float = 0.3
    cutmix: float = 0.0
    close_mosaic: int = 10  # 最后N个epoch关闭mosaic

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

    # 输出配置
    project: str = "runs/train"
    name: str = "freeze_train"
    exist_ok: bool = False
    workers: int = 8
    verbose: bool = True
    seed: int = 0

    def __post_init__(self):
        if self.freeze_layers is None:
            # 默认冻结前10层 (骨干网络)
            self.freeze_layers = list(range(10))

    def to_dict(self) -> Dict:
        """转换为字典，跳过None值"""
        result = {}
        for k, v in asdict(self).items():
            if v is not None:
                result[k] = v
        return result

    def save_config(self, path: str):
        """保存配置到YAML文件"""
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)


class FreezeTrainer:
    """冻结训练器"""

    def __init__(self, config: Union[Dict, FreezeTrainConfig, str] = None):
        if config is None:
            self.config = FreezeTrainConfig()
        elif isinstance(config, dict):
            self.config = FreezeTrainConfig(**config)
        elif isinstance(config, FreezeTrainConfig):
            self.config = config
        elif isinstance(config, str):
            # 从文件加载
            with open(config, 'r') as f:
                self.config = FreezeTrainConfig(**yaml.safe_load(f))

        self.model = None

    def build_model(self) -> YOLO:
        """构建模型"""
        self.model = YOLO(self.config.model)
        return self.model

    def train(self, model_path: Optional[str] = None) -> dict:
        """执行冻结训练"""
        if model_path:
            self.config.model = model_path

        if self.model is None:
            self.build_model()

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
            'warmup_momentum': self.config.warmup_momentum,
            'warmup_bias_lr': self.config.warmup_bias_lr,
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

        print(f"开始冻结训练...")
        print(f"冻结层: {self.config.freeze_layers}")
        print(f"训练参数: {train_params}")

        results = self.model.train(**train_params)
        return results

    @staticmethod
    def quick_freeze_train(
        model: str = "yolo11n.pt",
        data: str = "data.yaml",
        epochs: int = 100,
        freeze_layers: List[int] = None,
        **kwargs
    ) -> dict:
        """快速冻结训练接口"""
        if freeze_layers is None:
            freeze_layers = list(range(10))

        config = FreezeTrainConfig(
            model=model,
            data=data,
            epochs=epochs,
            freeze_layers=freeze_layers,
            **kwargs
        )

        trainer = FreezeTrainer(config)
        return trainer.train()


def main():
    parser = argparse.ArgumentParser(description='YOLO冻结训练工具')
    parser.add_argument('--model', type=str, default='yolo11n.pt', help='模型路径')
    parser.add_argument('--data', type=str, required=True, help='数据集配置')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--imgsz', type=int, default=640, help='图像尺寸')
    parser.add_argument('--batch', type=int, default=16, help='批次大小')
    parser.add_argument('--device', type=str, default='0', help='设备')
    parser.add_argument('--freeze', type=int, nargs='+', default=[0,1,2,3,4,5,6,7,8,9],
                        help='冻结的层索引')
    parser.add_argument('--lr0', type=float, default=0.01, help='初始学习率')
    parser.add_argument('--mosaic', type=float, default=1.0, help='Mosaic增强概率')
    parser.add_argument('--mixup', type=float, default=0.3, help='Mixup增强概率')
    parser.add_argument('--project', type=str, default='runs/train', help='输出项目目录')
    parser.add_argument('--name', type=str, default='freeze_train', help='实验名称')
    parser.add_argument('--config', type=str, default=None, help='配置文件路径')

    args = parser.parse_args()

    if args.config:
        # 从配置文件加载
        trainer = FreezeTrainer(args.config)
    else:
        # 从命令行参数构建
        config = FreezeTrainConfig(
            model=args.model,
            data=args.data,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            freeze_layers=args.freeze,
            lr0=args.lr0,
            mosaic=args.mosaic,
            mixup=args.mixup,
            project=args.project,
            name=args.name,
        )
        trainer = FreezeTrainer(config)

    trainer.train()


if __name__ == '__main__':
    main()
