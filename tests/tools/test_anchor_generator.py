"""
Unit tests for Anchor Generator module
"""

import numpy as np
import pytest
from pathlib import Path
import tempfile
import os

from src.tools.anchor_generator import (
    AnchorConfig,
    AnchorGenerator,
    compute_iou,
    iou_distance,
    kmeans_iou,
    silhouette_score,
)


class TestComputeIou:
    """Tests for compute_iou function"""

    def test_compute_iou_identical_boxes(self):
        """IoU of identical boxes = 1"""
        box = np.array([0.5, 0.5, 0.2, 0.2])
        boxes = np.array([[0.5, 0.5, 0.2, 0.2]])
        iou = compute_iou(box, boxes)
        assert np.isclose(iou[0], 1.0, atol=1e-6)

    def test_compute_iou_non_overlapping(self):
        """IoU of non-overlapping boxes < 1e-6"""
        box = np.array([0.1, 0.1, 0.1, 0.1])
        boxes = np.array([[0.9, 0.9, 0.1, 0.1]])  # Far away, no overlap
        iou = compute_iou(box, boxes)
        assert iou[0] < 1e-6


class TestIouDistance:
    """Tests for iou_distance function"""

    def test_iou_distance(self):
        """distance = 1 - iou"""
        box = np.array([0.5, 0.5, 0.2, 0.2])
        boxes = np.array([[0.5, 0.5, 0.2, 0.2]])
        iou = compute_iou(box, boxes)[0]
        dist = iou_distance(box, boxes)[0]
        assert np.isclose(dist, 1 - iou, atol=1e-6)


class TestKmeans:
    """Tests for kmeans_iou function"""

    def test_kmeans_basic(self):
        """k-means clustering returns k centroids"""
        np.random.seed(42)
        # Create 3 clusters of boxes
        cluster1 = np.random.rand(10, 4) * 0.2 + [0.2, 0.2, 0.1, 0.1]
        cluster2 = np.random.rand(10, 4) * 0.2 + [0.6, 0.6, 0.15, 0.15]
        cluster3 = np.random.rand(10, 4) * 0.2 + [0.8, 0.3, 0.12, 0.12]
        bboxes = np.vstack([cluster1, cluster2, cluster3])

        k = 3
        centroids, avg_iou = kmeans_iou(bboxes, n_clusters=k)

        assert centroids.shape == (k, 4)
        assert 0 <= avg_iou <= 1

    def test_kmeans_fewer_bboxes_than_k(self):
        """Handles edge case when fewer bboxes than k"""
        np.random.seed(42)
        bboxes = np.random.rand(3, 4)  # Only 3 boxes, ask for 5 clusters
        k = 5
        centroids, avg_iou = kmeans_iou(bboxes, n_clusters=k)

        # Should use min(n_samples, k) as effective k
        effective_k = min(len(bboxes), k)
        assert centroids.shape == (effective_k, 4)


class TestSilhouetteScore:
    """Tests for silhouette_score function"""

    def test_silhouette_good_clustering(self):
        """Good clustering has high score"""
        np.random.seed(42)
        # Create well-separated clusters
        cluster1 = np.random.rand(20, 4) * 0.1 + [0.2, 0.2, 0.1, 0.1]
        cluster2 = np.random.rand(20, 4) * 0.1 + [0.7, 0.7, 0.1, 0.1]

        bboxes = np.vstack([cluster1, cluster2])
        assignments = np.array([0] * 20 + [1] * 20)

        # Compute centroids
        centroids = np.array([
            cluster1.mean(axis=0),
            cluster2.mean(axis=0)
        ])

        score = silhouette_score(bboxes, assignments, centroids)
        assert score > 0  # Good clustering should have positive score

    def test_silhouette_single_cluster(self):
        """Single cluster score = 1"""
        np.random.seed(42)
        bboxes = np.random.rand(10, 4)
        assignments = np.zeros(10, dtype=int)
        centroids = np.array([bboxes.mean(axis=0)])

        score = silhouette_score(bboxes, assignments, centroids)
        # silhouette_score returns 1.0 for single cluster case
        assert score == 1.0


class TestAnchorGenerator:
    """Tests for AnchorGenerator class"""

    def test_default_init(self):
        """AnchorGenerator with defaults"""
        generator = AnchorGenerator()
        assert generator.config is not None
        assert generator.config.n_clusters_min == 5
        assert generator.config.n_clusters_max == 15
        assert generator.bboxes is None
        assert generator.anchors is None

    def test_custom_config(self):
        """AnchorGenerator with custom config"""
        config = AnchorConfig(
            n_clusters_min=3,
            n_clusters_max=10,
            scales=['P3', 'P4']
        )
        generator = AnchorGenerator(config)
        assert generator.config.n_clusters_min == 3
        assert generator.config.n_clusters_max == 10
        assert generator.config.scales == ['P3', 'P4']


class TestLabelParsing:
    """Tests for label parsing functionality"""

    def test_parse_valid_labels(self):
        """Parse YOLO format labels"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a temp label file
            label_file = Path(tmpdir) / "test.txt"
            # YOLO format: class x_center y_center width height
            content = "0 0.5 0.5 0.2 0.2\n1 0.3 0.3 0.1 0.1\n"
            label_file.write_text(content)

            generator = AnchorGenerator()
            bboxes = generator._parse_labels(label_file)

            assert len(bboxes) == 2
            np.testing.assert_array_almost_equal(bboxes[0], [0.5, 0.5, 0.2, 0.2])
            np.testing.assert_array_almost_equal(bboxes[1], [0.3, 0.3, 0.1, 0.1])

    def test_parse_invalid_lines(self):
        """Skip invalid lines"""
        with tempfile.TemporaryDirectory() as tmpdir:
            label_file = Path(tmpdir) / "test.txt"
            # Mix of valid and invalid lines
            content = """0 0.5 0.5 0.2 0.2
1 0.3 0.3  # invalid line - has only 3 values
2 0.4      # invalid line - has only 2 values
invalid line here
0 0.6 0.6 0.15 0.15
"""
            label_file.write_text(content)

            generator = AnchorGenerator()
            bboxes = generator._parse_labels(label_file)

            # Should only parse the 2 valid lines
            assert len(bboxes) == 2


class TestScaleAssignment:
    """Tests for scale assignment functionality"""

    def test_assign_small_to_p3(self):
        """Small bbox assigned to P3"""
        generator = AnchorGenerator(AnchorConfig(scales=['P3', 'P4', 'P5']))

        # Small bbox (normalized)
        small_bbox = np.array([[0.5, 0.5, 0.05, 0.05]])

        scale_bboxes = generator._assign_to_scale(small_bbox)

        # Small bbox should be in P3
        assert len(scale_bboxes['P3']) == 1
        np.testing.assert_array_almost_equal(scale_bboxes['P3'][0], small_bbox[0])

    def test_assign_large_to_p5(self):
        """Large bbox assigned to P5"""
        generator = AnchorGenerator(AnchorConfig(scales=['P3', 'P4', 'P5']))

        # Large bbox (normalized)
        large_bbox = np.array([[0.5, 0.5, 0.8, 0.8]])

        scale_bboxes = generator._assign_to_scale(large_bbox)

        # Large bbox should be in P5
        assert len(scale_bboxes['P5']) == 1
        np.testing.assert_array_almost_equal(scale_bboxes['P5'][0], large_bbox[0])
