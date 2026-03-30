import pytest
from src.tools.hard_example_miner import HardExampleMiningConfig


def test_config_defaults():
    config = HardExampleMiningConfig(
        model="best.pt",
        data="dataset.yaml"
    )
    assert config.output == "./hard_examples"
    assert config.strategy == "oversample"
    assert config.iou_threshold == 0.5
    assert config.conf_threshold == 0.25
    assert config.small_area_threshold == 0.01
    assert config.max_oversample == 5


def test_hardness_score():
    from src.tools.hard_example_miner import compute_hardness_score

    # FP: 置信度越高越难
    assert compute_hardness_score("FP", 0.0, 0.9) == pytest.approx(0.1)  # 1.0 - 0.9
    assert compute_hardness_score("FP", 0.0, 0.5) == pytest.approx(0.5)

    # FN: IoU 越低越难
    assert compute_hardness_score("FN", 0.3, 0.0) == pytest.approx(0.7)  # 1.0 - 0.3
    assert compute_hardness_score("FN", 0.1, 0.0) == pytest.approx(0.9)

    # 小目标: 固定 0.8
    assert compute_hardness_score("small", 0.0, 0.0) == pytest.approx(0.8)


def test_oversample_ratio():
    from src.tools.hard_example_miner import get_oversample_ratio

    assert get_oversample_ratio(0.3) == 1  # < 0.5, 不过采样
    assert get_oversample_ratio(0.5) == 1
    assert get_oversample_ratio(0.6) == 2  # 0.5 - 0.7 → 2x
    assert get_oversample_ratio(0.75) == 3  # 0.7 - 0.9 → 3x
    assert get_oversample_ratio(0.95) == 5  # > 0.9 → 5x
    assert get_oversample_ratio(1.0) == 5  # 不超过 max_oversample