"""
YOLO 训练模块 - 支持断点续训
"""

import os
import argparse
import logging
import time
from pathlib import Path
from typing import List, Optional, Union, Dict
from dataclasses import dataclass, field
import yaml

logger = logging.getLogger("yolo_toolchain.trainer")


@dataclass
class TrainConfig:
    """训练配置"""
    model: str = "yolo11n.pt"
    data: str = ""
    epochs: int = 100
    imgsz: int = 640
    batch: int = 16
    device: str = "0"
    optimizer: str = "auto"
    lr0: float = 0.01
    lrf: float = 0.01
    momentum: float = 0.937
    weight_decay: float = 0.0005
    warmup_epochs: float = 3.0
    patience: int = 50
    save: bool = True
    save_period: int = -1
    cache: bool = False
    workers: int = 8
    project: str = "runs/train"
    name: str = "exp"
    exist_ok: bool = False
    pretrained: bool = True
    verbose: bool = True
    seed: int = 0
    deterministic: bool = True
    single_cls: bool = False
    rect: bool = False
    cos_lr: bool = False
    close_mosaic: int = 10
    resume: bool = False
    amp: bool = True
    fraction: float = 1.0
    profile: bool = False
    freeze: List[int] = field(default_factory=list)
    overlap_mask: bool = True
    mask_ratio: int = 4
    dropout: float = 0.0
    val: bool = True
    plots: bool = True


class Trainer:
    """YOLO 训练器"""

    def __init__(self, config: Union[TrainConfig, Dict, str]):
        if isinstance(config, dict):
            self.config = TrainConfig(**config)
        elif isinstance(config, str):
            with open(config, 'r') as f:
                yaml_config = yaml.safe_load(f)
                self.config = TrainConfig(**yaml_config)
        else:
            self.config = config

        self.model = None
        self.trainer = None

    def build_model(self):
        """构建模型"""
        from ultralytics import YOLO

        model_path = self.config.model
        if not Path(model_path).exists():
            logger.info(f"Model {model_path} not found, downloading...")
            self.model = YOLO(model_path)
        else:
            self.model = YOLO(model_path)

        return self.model

    def train(self, resume: bool = False) -> Dict:
        """训练模型

        Args:
            resume: 是否从上次中断处继续训练

        Returns:
            训练结果字典
        """
        if self.model is None:
            self.build_model()

        if resume:
            logger.info("Resuming training from last checkpoint...")
            self.config.resume = True

        params = self._build_params()

        start_time = time.time()
        logger.info(f"Starting training: epochs={self.config.epochs}, imgsz={self.config.imgsz}, batch={self.config.batch}")

        results = self.model.train(**params)

        elapsed = time.time() - start_time
        logger.info(f"Training completed in {elapsed:.1f}s")

        return results

    def _build_params(self) -> Dict:
        """构建训练参数"""
        params = {
            'data': self.config.data,
            'epochs': self.config.epochs,
            'imgsz': self.config.imgsz,
            'batch': self.config.batch,
            'device': self.config.device,
            'save': self.config.save,
            'save_period': self.config.save_period,
            'cache': self.config.cache,
            'workers': self.config.workers,
            'project': self.config.project,
            'name': self.config.name,
            'exist_ok': self.config.exist_ok,
            'pretrained': self.config.pretrained,
            'verbose': self.config.verbose,
            'seed': self.config.seed,
            'deterministic': self.config.deterministic,
            'single_cls': self.config.single_cls,
            'rect': self.config.rect,
            'cos_lr': self.config.cos_lr,
            'close_mosaic': self.config.close_mosaic,
            'resume': self.config.resume,
            'amp': self.config.amp,
            'fraction': self.config.fraction,
            'profile': self.config.profile,
            'overlap_mask': self.config.overlap_mask,
            'mask_ratio': self.config.mask_ratio,
            'dropout': self.config.dropout,
            'val': self.config.val,
            'plots': self.config.plots,
        }

        if self.config.optimizer != "auto":
            params['optimizer'] = self.config.optimizer
            params['lr0'] = self.config.lr0
            params['lrf'] = self.config.lrf
            params['momentum'] = self.config.momentum
            params['weight_decay'] = self.config.weight_decay

        if self.config.warmup_epochs > 0:
            params['warmup_epochs'] = self.config.warmup_epochs

        if self.config.patience > 0:
            params['patience'] = self.config.patience

        if self.config.freeze:
            params['freeze'] = self.config.freeze

        return params

    def find_last_checkpoint(self, project: str = None, name: str = None) -> Optional[str]:
        """查找最近的训练检查点

        Args:
            project: 项目目录
            name: 实验名称

        Returns:
            检查点路径，如果不存在返回 None
        """
        project = project or self.config.project
        name = name or self.config.name

        project_path = Path(project) / name
        checkpoint_path = project_path / "weights" / "last.pt"

        if checkpoint_path.exists():
            logger.info(f"Found checkpoint: {checkpoint_path}")
            return str(checkpoint_path)

        checkpoints = list(project_path.glob("*/weights/last.pt"))
        if checkpoints:
            latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
            logger.info(f"Found checkpoint: {latest}")
            return str(latest)

        return None

    def get_best_checkpoint(self, project: str = None, name: str = None) -> Optional[str]:
        """获取最佳模型检查点"""
        project = project or self.config.project
        name = name or self.config.name

        project_path = Path(project) / name
        checkpoint_path = project_path / "weights" / "best.pt"

        if checkpoint_path.exists():
            return str(checkpoint_path)

        checkpoints = list(project_path.glob("*/weights/best.pt"))
        if checkpoints:
            return str(checkpoints[0])

        return None


def train_with_resume(
    model: str = "yolo11n.pt",
    data: str = "",
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = 16,
    device: str = "0",
    project: str = "runs/train",
    name: str = "exp",
    resume: bool = False,
    **kwargs
):
    """带断点续训的训练函数"""
    config = TrainConfig(
        model=model,
        data=data,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=project,
        name=name,
        resume=resume,
        **kwargs
    )

    trainer = Trainer(config)

    if resume and not config.model.endswith('.pt'):
        checkpoint = trainer.find_last_checkpoint()
        if checkpoint:
            config.model = checkpoint
            logger.info(f"Resuming from: {checkpoint}")
        else:
            logger.warning("No checkpoint found, starting fresh training")

    return trainer.train(resume=resume)


def main():
    parser = argparse.ArgumentParser(description='YOLO 训练工具（支持断点续训）')
    parser.add_argument('--model', type=str, default='yolo11n.pt', help='模型路径或名称')
    parser.add_argument('--data', type=str, required=True, help='数据集配置文件')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--imgsz', type=int, default=640, help='图像尺寸')
    parser.add_argument('--batch', type=int, default=16, help='批次大小')
    parser.add_argument('--device', type=str, default='0', help='设备')
    parser.add_argument('--project', type=str, default='runs/train', help='项目目录')
    parser.add_argument('--name', type=str, default='exp', help='实验名称')
    parser.add_argument('--resume', action='store_true', help='从上次中断处继续训练')
    parser.add_argument('--config', type=str, help='YAML 配置文件路径')
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='日志级别')

    args = parser.parse_args()

    import logging as log_module
    log_level = getattr(log_module, args.log_level.upper(), log_module.INFO)
    log_module.basicConfig(
        level=log_level,
        format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    if args.config:
        with open(args.config, 'r') as f:
            yaml_config = yaml.safe_load(f)
        config = TrainConfig(**yaml_config)
    else:
        config = TrainConfig(
            model=args.model,
            data=args.data,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            project=args.project,
            name=args.name,
            resume=args.resume
        )

    trainer = Trainer(config)

    if args.resume and not config.model.endswith('.pt'):
        checkpoint = trainer.find_last_checkpoint()
        if checkpoint:
            config.model = checkpoint
            logger.info(f"Resuming from: {checkpoint}")
        else:
            logger.warning("No checkpoint found, starting fresh training")

    logger.info(f"Starting training: model={config.model}, data={config.data}, epochs={config.epochs}")
    results = trainer.train(resume=args.resume)

    best_model = trainer.get_best_checkpoint()
    if best_model:
        logger.info(f"Best model saved at: {best_model}")


if __name__ == '__main__':
    main()
