import pytest
from src.tools.hard_example_miner import HardExampleMiningConfig, HardExampleMiner


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


def test_hard_example_miner_init():
    config = HardExampleMiningConfig(
        model="best.pt",
        data="dataset.yaml",
        output="./test_output"
    )
    miner = HardExampleMiner(config)
    assert miner.config == config
    assert miner.fp_cases == []
    assert miner.fn_cases == []
    assert miner.small_cases == []


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


def test_get_variant_count():
    from src.tools.hard_example_miner import get_variant_count

    assert get_variant_count(0.3) == 0  # < 0.5, 0个新变体
    assert get_variant_count(0.6) == 2  # 0.5-0.7
    assert get_variant_count(0.8) == 3  # 0.7-0.9
    assert get_variant_count(0.95) == 4  # >0.9


def test_compute_iou_xyxy():
    from src.tools.hard_example_miner import compute_iou_xyxy
    import numpy as np

    # 完全不重叠
    box1 = [0, 0, 10, 10]
    box2 = [20, 20, 30, 30]
    assert compute_iou_xyxy(box1, box2) == pytest.approx(0.0)

    # 完全重叠
    box1 = [0, 0, 10, 10]
    box2 = [0, 0, 10, 10]
    assert compute_iou_xyxy(box1, box2) == pytest.approx(1.0)

    # 部分重叠
    box1 = [0, 0, 10, 10]
    box2 = [5, 5, 15, 15]
    iou = compute_iou_xyxy(box1, box2)
    assert 0.1 < iou < 0.3  # 约 0.14


def test_classify_errors():
    from src.tools.hard_example_miner import classify_errors

    # 模拟预测和真值
    predictions = [
        {"image": "img1.jpg", "boxes": [[0, 0, 10, 10, 0.9, 0]]},  # 正确检测
        {"image": "img2.jpg", "boxes": [[100, 100, 110, 110, 0.8, 0]]},  # FP
    ]
    ground_truths = {
        "img1.jpg": [[0, 0, 10, 10, 0]],  # 匹配
        "img2.jpg": [],  # 无真值
    }
    image_paths = {"img1.jpg": "/path/img1.jpg", "img2.jpg": "/path/img2.jpg"}

    fp_cases, fn_cases, correct = classify_errors(
        predictions, ground_truths, image_paths,
        iou_threshold=0.5, conf_threshold=0.25
    )

    assert len(fp_cases) == 1
    assert fp_cases[0]["image"] == "img2.jpg"


def test_augment_image():
    import tempfile
    import os
    from src.tools.hard_example_miner import augment_image
    import cv2
    import numpy as np

    # 创建临时图片
    with tempfile.TemporaryDirectory() as tmpdir:
        img_path = os.path.join(tmpdir, "test.jpg")
        # 先创建测试图片
        test_img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.imwrite(img_path, test_img)

        # 验证增强函数
        variants = augment_image(img_path, "FP", output_dir=tmpdir)
        assert len(variants) >= 1
        assert all(os.path.exists(v) for v in variants)