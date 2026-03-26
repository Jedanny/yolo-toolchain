"""
Unit tests for Label QC module.

Tests IoU computation, duplicate detection, and configuration handling.
"""

import numpy as np
import pytest
from src.tools.label_qc import (
    compute_iou,
    check_duplicate_boxes,
    LabelQCConfig,
    LabelQCChecker,
)


class TestComputeIoU:
    """Tests for compute_iou function."""

    def test_compute_iou_identical(self):
        """IoU of identical boxes = 1."""
        box = np.array([0.5, 0.5, 0.2, 0.2])  # [cx, cy, w, h]
        iou = compute_iou(box, box)
        assert iou == 1.0, f"Expected IoU=1.0 for identical boxes, got {iou}"

    def test_compute_iou_no_overlap(self):
        """IoU of non-overlapping boxes < 1e-6."""
        # Box at top-left corner
        box1 = np.array([0.1, 0.1, 0.1, 0.1])
        # Box at bottom-right corner (no overlap)
        box2 = np.array([0.9, 0.9, 0.1, 0.1])
        iou = compute_iou(box1, box2)
        assert iou < 1e-6, f"Expected IoU ~ 0 for non-overlapping boxes, got {iou}"

    def test_compute_iou_partial_overlap(self):
        """IoU of partially overlapping boxes is between 0 and 1."""
        box1 = np.array([0.3, 0.3, 0.4, 0.4])
        box2 = np.array([0.5, 0.5, 0.4, 0.4])
        iou = compute_iou(box1, box2)
        assert 0 < iou < 1, f"Expected 0 < IoU < 1 for partial overlap, got {iou}"


class TestCheckDuplicateBoxes:
    """Tests for check_duplicate_boxes function."""

    def test_check_duplicate_boxes(self):
        """Detects duplicates with high IoU."""
        bboxes = np.array([
            [0.5, 0.5, 0.2, 0.2],
            [0.51, 0.51, 0.2, 0.2],  # High overlap with first box
            [0.1, 0.1, 0.1, 0.1],    # No overlap with others
        ])
        duplicates = check_duplicate_boxes(bboxes, iou_threshold=0.8)
        # Should detect boxes 0 and 1 as duplicates (high IoU)
        assert len(duplicates) == 1, f"Expected 1 duplicate pair, got {len(duplicates)}"
        idx1, idx2, iou = duplicates[0]
        assert idx1 == 0 and idx2 == 1, f"Expected duplicate pair (0, 1), got ({idx1}, {idx2})"
        assert iou >= 0.8, f"Expected IoU >= 0.8, got {iou}"

    def test_check_duplicate_boxes_no_duplicates(self):
        """No false positives when boxes don't overlap."""
        bboxes = np.array([
            [0.2, 0.2, 0.1, 0.1],
            [0.5, 0.5, 0.1, 0.1],
            [0.8, 0.8, 0.1, 0.1],
        ])
        duplicates = check_duplicate_boxes(bboxes, iou_threshold=0.8)
        assert len(duplicates) == 0, f"Expected no duplicates, got {len(duplicates)}"


class TestLabelQCConfig:
    """Tests for LabelQCConfig dataclass."""

    def test_LabelQCConfig_default(self):
        """Default config values are correct."""
        config = LabelQCConfig()
        assert config.duplicate_iou_threshold == 0.8
        assert config.min_box_area == 100
        assert config.max_box_area_ratio == 0.9
        assert config.min_box_area_ratio == 0.001
        assert config.occlusion_variance_threshold == 50.0
        assert config.image_size == (640, 640)
        assert config.backup_before_fix is True
        assert config.report_format == 'text'

    def test_LabelQCConfig_custom(self):
        """Custom config values are applied correctly."""
        config = LabelQCConfig(
            duplicate_iou_threshold=0.9,
            min_box_area=200,
            max_box_area_ratio=0.95,
            min_box_area_ratio=0.002,
            occlusion_variance_threshold=100.0,
            image_size=(1024, 1024),
            backup_before_fix=False,
            report_format='json',
        )
        assert config.duplicate_iou_threshold == 0.9
        assert config.min_box_area == 200
        assert config.max_box_area_ratio == 0.95
        assert config.min_box_area_ratio == 0.002
        assert config.occlusion_variance_threshold == 100.0
        assert config.image_size == (1024, 1024)
        assert config.backup_before_fix is False
        assert config.report_format == 'json'


class TestLabelQCChecker:
    """Tests for LabelQCChecker class."""

    def test_LabelQCChecker_init_default(self):
        """Checker initializes with default config."""
        checker = LabelQCChecker()
        assert checker.config is not None
        assert isinstance(checker.config, LabelQCConfig)
        assert checker.config.duplicate_iou_threshold == 0.8

    def test_LabelQCChecker_init_custom_config(self):
        """Checker initializes with custom config."""
        config = LabelQCConfig(duplicate_iou_threshold=0.95, min_box_area=500)
        checker = LabelQCChecker(config)
        assert checker.config.duplicate_iou_threshold == 0.95
        assert checker.config.min_box_area == 500

    def test_LabelQCChecker_init_stats_initialized(self):
        """Checker initializes with empty statistics."""
        checker = LabelQCChecker()
        assert checker.stats['total_images'] == 0
        assert checker.stats['total_labels'] == 0
        assert checker.stats['duplicate_boxes'] == 0
        assert checker.stats['tiny_boxes'] == 0
        assert checker.stats['images_with_issues'] == 0
