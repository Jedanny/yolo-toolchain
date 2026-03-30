from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class HardExampleMiningConfig:
    """难例挖掘配置"""
    model: str
    data: str
    output: str = "./hard_examples"
    strategy: str = "oversample"  # oversample / weighted / filter
    iou_threshold: float = 0.5
    conf_threshold: float = 0.25
    small_area_threshold: float = 0.01
    max_oversample: int = 5
    device: str = "cpu"


def compute_hardness_score(error_type: str, iou: float, confidence: float) -> float:
    """
    计算难例分数 (0-1, 越高越难)

    FP: 分数 = 1.0 - confidence  (置信度越高越"自信"的误检越难)
    FN: 分数 = 1.0 - iou         (IoU越低越难)
    小目标: 分数 = 0.8           (固定高分)
    """
    if error_type == "FP":
        return 1.0 - confidence
    elif error_type == "FN":
        return 1.0 - iou if iou < 1.0 else 0.0
    elif error_type == "small":
        return 0.8
    else:
        return 0.5


def get_oversample_ratio(hardness_score: float, max_oversample: int = 5) -> int:
    """根据难例分数返回过采样倍数"""
    if hardness_score <= 0.5:
        return 1
    elif hardness_score < 0.7:
        return 2
    elif hardness_score < 0.9:
        return 3
    else:
        return min(5, max_oversample)
