"""
Label Quality Check Module for YOLO datasets.

Detects and auto-fixes label quality issues:
- Duplicate boxes (auto-fix: delete high-IoU duplicates)
- Tiny boxes (auto-fix: delete too small bboxes)
- Oversized boxes (report only)
- Occlusion (report only)
"""

import os
import json
import shutil
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Any
from datetime import datetime

import numpy as np
from PIL import Image


@dataclass
class LabelQCConfig:
    """Configuration for label quality checks."""

    # IoU threshold for duplicate detection (auto-fix)
    duplicate_iou_threshold: float = 0.8

    # Minimum box area in pixels (auto-fix tiny boxes)
    min_box_area: int = 100

    # Maximum box area as ratio of image area (report oversized)
    max_box_area_ratio: float = 0.9

    # Minimum box area as ratio of image area (report tiny)
    min_box_area_ratio: float = 0.001

    # Minimum variance for occlusion detection
    occlusion_variance_threshold: float = 50.0

    # Image size for normalization
    image_size: Tuple[int, int] = (640, 640)

    # Backup labels before fixing
    backup_before_fix: bool = True

    # Report format: 'text' or 'json'
    report_format: str = 'text'


def compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Compute IoU between two boxes in [cx, cy, w, h] format.

    Args:
        box1: First box [cx, cy, w, h]
        box2: Second box [cx, cy, w, h]

    Returns:
        IoU value between 0 and 1
    """
    # Convert from center format to corner format [x1, y1, x2, y2]
    def center_to_corners(box):
        cx, cy, w, h = box
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        return np.array([x1, y1, x2, y2])

    box1_corners = center_to_corners(box1)
    box2_corners = center_to_corners(box2)

    # Calculate intersection area
    x1 = max(box1_corners[0], box2_corners[0])
    y1 = max(box1_corners[1], box2_corners[1])
    x2 = min(box1_corners[2], box2_corners[2])
    y2 = min(box1_corners[3], box2_corners[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    # Calculate union area
    area1 = (box1_corners[2] - box1_corners[0]) * (box1_corners[3] - box1_corners[1])
    area2 = (box2_corners[2] - box2_corners[0]) * (box2_corners[3] - box2_corners[1])
    union = area1 + area2 - intersection

    if union == 0:
        return 0.0

    return intersection / union


def check_duplicate_boxes(bboxes: np.ndarray, iou_threshold: float = 0.8) -> List[Tuple[int, int, float]]:
    """
    Check for duplicate boxes based on IoU threshold.

    Args:
        bboxes: Array of boxes in [cx, cy, w, h] format, shape (N, 4)
        iou_threshold: IoU threshold above which boxes are considered duplicates

    Returns:
        List of (idx1, idx2, iou) tuples for duplicate pairs
    """
    duplicates = []
    n = len(bboxes)

    for i in range(n):
        for j in range(i + 1, n):
            iou = compute_iou(bboxes[i], bboxes[j])
            if iou >= iou_threshold:
                duplicates.append((i, j, iou))

    return duplicates


class LabelQCChecker:
    """
    Label quality checker for YOLO format datasets.

    Detects:
    - Duplicate boxes (high IoU overlap)
    - Tiny boxes (below minimum area)
    - Oversized boxes (too large relative to image)
    - Occlusion (high variance in box region)

    Auto-fixes: duplicates, tiny boxes
    Reports only: oversized boxes, occlusion
    """

    def __init__(self, config: Optional[LabelQCConfig] = None):
        """
        Initialize LabelQCChecker.

        Args:
            config: LabelQCConfig instance. Uses defaults if None.
        """
        self.config = config or LabelQCConfig()
        self.stats = {
            'total_images': 0,
            'total_labels': 0,
            'duplicate_boxes': 0,
            'tiny_boxes': 0,
            'oversized_boxes': 0,
            'occlusion_suspected': 0,
            'images_with_issues': 0,
            'duplicate_pairs': [],
            'tiny_box_indices': [],
            'oversized_box_indices': [],
            'occlusion_indices': [],
        }
        self.image_to_label_map: Dict[str, str] = {}

    def _reset_stats(self) -> None:
        """Reset statistics counters."""
        self.stats = {
            'total_images': 0,
            'total_labels': 0,
            'duplicate_boxes': 0,
            'tiny_boxes': 0,
            'oversized_boxes': 0,
            'occlusion_suspected': 0,
            'images_with_issues': 0,
            'duplicate_pairs': [],
            'tiny_box_indices': [],
            'oversized_box_indices': [],
            'occlusion_indices': [],
        }
        self.image_to_label_map = {}

    def _find_image_for_label(self, label_path: Path) -> Optional[Path]:
        """
        Find corresponding image file for a label file.

        Args:
            label_path: Path to label .txt file

        Returns:
            Path to image file or None if not found
        """
        # Get image extensions to try
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']

        # Common image directory names
        image_dirs = ['images', 'JPEGImages', 'data', 'img']

        label_path_obj = Path(label_path)
        label_stem = label_path_obj.stem

        # Try same directory
        for ext in extensions:
            img_path = label_path_obj.parent / f"{label_stem}{ext}"
            if img_path.exists():
                return img_path

        # Try parent directories for image subdirectory
        for img_dir in image_dirs:
            for parent in label_path_obj.parents:
                img_path = parent / img_dir / f"{label_stem}{ext}"
                if img_path.exists():
                    return img_path

        # Try subdirectories of parent
        for parent in label_path_obj.parents:
            for img_dir in image_dirs:
                img_path = parent / img_dir / f"{label_stem}{ext}"
                if img_path.exists():
                    return img_path

        return None

    def _check_occlusion(self, image_path: Path, bbox: np.ndarray) -> bool:
        """
        Simple variance-based occlusion detection.

        Args:
            image_path: Path to image file
            bbox: Box in [cx, cy, w, h] format (normalized 0-1)

        Returns:
            True if occlusion suspected
        """
        try:
            img = Image.open(image_path).convert('RGB')
            img_w, img_h = img.size

            # Convert normalized bbox to pixel coordinates
            cx, cy, w, h = bbox
            x1 = int((cx - w / 2) * img_w)
            y1 = int((cy - h / 2) * img_h)
            x2 = int((cx + w / 2) * img_w)
            y2 = int((cy + h / 2) * img_h)

            # Ensure valid crop coordinates
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img_w, x2), min(img_h, y2)

            if x2 <= x1 or y2 <= y1:
                return False

            # Crop and calculate variance
            crop = img.crop((x1, y1, x2, y2))
            crop_array = np.array(crop)

            # Calculate variance across channels
            variance = np.var(crop_array)

            # Low variance often indicates occlusion or flat surface
            return variance < self.config.occlusion_variance_threshold

        except Exception:
            return False

    def _check_label_file(self, label_path: Path) -> Dict[str, Any]:
        """
        Check a single label file for quality issues.

        Args:
            label_path: Path to label .txt file

        Returns:
            Dictionary with check results
        """
        result = {
            'path': str(label_path),
            'image_path': None,
            'issues': [],
            'duplicate_indices': [],
            'tiny_indices': [],
            'oversized_indices': [],
            'occlusion_indices': [],
        }

        # Find corresponding image
        image_path = self._find_image_for_label(label_path)
        result['image_path'] = str(image_path) if image_path else None

        if image_path is None:
            result['issues'].append('no_image_found')
            return result

        # Get image dimensions
        try:
            img = Image.open(image_path)
            img_w, img_h = img.size
        except Exception:
            result['issues'].append('image_load_error')
            return result

        # Read labels
        if not label_path.exists():
            result['issues'].append('label_file_missing')
            return result

        bboxes = []
        with open(label_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                cls = int(parts[0])
                cx, cy, w, h = map(float, parts[1:5])
                bboxes.append([cls, cx, cy, w, h])

        if not bboxes:
            return result

        bboxes_array = np.array([b[1:] for b in bboxes])  # Only bbox coords

        # Check for duplicates
        duplicate_pairs = check_duplicate_boxes(
            bboxes_array, self.config.duplicate_iou_threshold
        )

        # Track which indices are involved in duplicates
        duplicate_indices = set()
        for idx1, idx2, iou in duplicate_pairs:
            duplicate_indices.add(idx1)
            duplicate_indices.add(idx2)
            result['duplicate_indices'].append((idx1, idx2, iou))
            self.stats['duplicate_pairs'].append({
                'label_file': str(label_path),
                'idx1': idx1,
                'idx2': idx2,
                'iou': iou,
            })

        # Check for tiny boxes
        min_area_pixels = self.config.min_box_area
        min_area_ratio = self.config.min_box_area_ratio

        for idx, bbox in enumerate(bboxes_array):
            cx, cy, w, h = bbox
            box_area_pixels = w * h * img_w * img_h
            box_area_ratio = w * h

            if box_area_pixels < min_area_pixels or box_area_ratio < min_area_ratio:
                if idx not in duplicate_indices:  # Don't double-count duplicates
                    result['tiny_indices'].append(idx)
                    self.stats['tiny_box_indices'].append({
                        'label_file': str(label_path),
                        'idx': idx,
                        'area_pixels': box_area_pixels,
                    })

        # Check for oversized boxes
        max_area_ratio = self.config.max_box_area_ratio

        for idx, bbox in enumerate(bboxes_array):
            cx, cy, w, h = bbox
            box_area_ratio = w * h

            if box_area_ratio > max_area_ratio:
                result['oversized_indices'].append(idx)
                self.stats['oversized_box_indices'].append({
                    'label_file': str(label_path),
                    'idx': idx,
                    'area_ratio': box_area_ratio,
                })

        # Check for occlusion (on boxes that pass other checks)
        for idx, bbox in enumerate(bboxes_array):
            if idx in duplicate_indices or idx in result['tiny_indices']:
                continue
            if self._check_occlusion(image_path, bbox):
                result['occlusion_indices'].append(idx)
                self.stats['occlusion_indices'].append({
                    'label_file': str(label_path),
                    'idx': idx,
                })

        return result

    def _update_stats(self, result: Dict[str, Any]) -> None:
        """
        Update internal statistics from a check result.

        Args:
            result: Result from _check_label_file
        """
        self.stats['total_labels'] += len(result['duplicate_indices']) + \
            len(result['tiny_indices']) + \
            len(result['oversized_indices']) + \
            len(result['occlusion_indices'])

        has_issues = (
            result['duplicate_indices'] or
            result['tiny_indices'] or
            result['oversized_indices'] or
            result['occlusion_indices']
        )

        if has_issues:
            self.stats['images_with_issues'] += 1

        self.stats['duplicate_boxes'] += len(result['duplicate_indices'])
        self.stats['tiny_boxes'] += len(result['tiny_indices'])
        self.stats['oversized_boxes'] += len(result['oversized_indices'])
        self.stats['occlusion_suspected'] += len(result['occlusion_indices'])

    def _build_report(self) -> str:
        """
        Build text report from statistics.

        Returns:
            Formatted report string
        """
        lines = [
            "=" * 60,
            "Label Quality Check Report",
            "=" * 60,
            f"Timestamp: {datetime.now().isoformat()}",
            f"Total Images Checked: {self.stats['total_images']}",
            f"Total Labels: {self.stats['total_labels']}",
            "",
            "-" * 60,
            "Summary",
            "-" * 60,
            f"Images with Issues: {self.stats['images_with_issues']}",
            f"Duplicate Boxes: {self.stats['duplicate_boxes']} (auto-fix)",
            f"Tiny Boxes: {self.stats['tiny_boxes']} (auto-fix)",
            f"Oversized Boxes: {self.stats['oversized_boxes']} (report only)",
            f"Possible Occlusion: {self.stats['occlusion_suspected']} (report only)",
            "",
            "-" * 60,
            "Details - Duplicate Boxes (High IoU)",
            "-" * 60,
        ]

        if self.stats['duplicate_pairs']:
            for dup in self.stats['duplicate_pairs'][:20]:  # Limit to first 20
                lines.append(
                    f"  {dup['label_file']}: box[{dup['idx1']}] <-> box[{dup['idx2']}] IoU={dup['iou']:.3f}"
                )
            if len(self.stats['duplicate_pairs']) > 20:
                lines.append(f"  ... and {len(self.stats['duplicate_pairs']) - 20} more")
        else:
            lines.append("  None found")

        lines.extend([
            "",
            "-" * 60,
            "Details - Tiny Boxes",
            "-" * 60,
        ])

        if self.stats['tiny_box_indices']:
            for tiny in self.stats['tiny_box_indices'][:20]:
                lines.append(
                    f"  {tiny['label_file']}: box[{tiny['idx']}] area={tiny['area_pixels']:.1f}px"
                )
            if len(self.stats['tiny_box_indices']) > 20:
                lines.append(f"  ... and {len(self.stats['tiny_box_indices']) - 20} more")
        else:
            lines.append("  None found")

        lines.extend([
            "",
            "-" * 60,
            "Details - Oversized Boxes",
            "-" * 60,
        ])

        if self.stats['oversized_box_indices']:
            for over in self.stats['oversized_box_indices'][:20]:
                lines.append(
                    f"  {over['label_file']}: box[{over['idx']}] area_ratio={over['area_ratio']:.3f}"
                )
            if len(self.stats['oversized_box_indices']) > 20:
                lines.append(f"  ... and {len(self.stats['oversized_box_indices']) - 20} more")
        else:
            lines.append("  None found")

        lines.extend([
            "",
            "-" * 60,
            "Details - Possible Occlusion",
            "-" * 60,
        ])

        if self.stats['occlusion_indices']:
            for occ in self.stats['occlusion_indices'][:20]:
                lines.append(
                    f"  {occ['label_file']}: box[{occ['idx']}]"
                )
            if len(self.stats['occlusion_indices']) > 20:
                lines.append(f"  ... and {len(self.stats['occlusion_indices']) - 20} more")
        else:
            lines.append("  None found")

        lines.extend([
            "",
            "=" * 60,
            "Recommendations",
            "=" * 60,
            "- Duplicate boxes: Use apply_fixes() to auto-remove",
            "- Tiny boxes: Use apply_fixes() to auto-remove",
            "- Oversized/Occlusion: Manual review recommended",
            "=" * 60,
        ])

        return "\n".join(lines)

    def check(self, dataset_path: str) -> Dict[str, Any]:
        """
        Check all labels in a dataset for quality issues.

        Args:
            dataset_path: Path to dataset directory containing labels

        Returns:
            Dictionary with all check results
        """
        self._reset_stats()
        dataset_path_obj = Path(dataset_path)

        # Find all label files
        label_files = list(dataset_path_obj.rglob("*.txt"))

        # Filter out non-label files (e.g., dataset.yaml, classes.txt)
        label_files = [
            f for f in label_files
            if not any(x in f.name.lower() for x in ['dataset', 'classes', 'names'])
        ]

        results = []

        for label_file in label_files:
            self.stats['total_images'] += 1
            result = self._check_label_file(label_file)
            results.append(result)
            self._update_stats(result)

        return {
            'config': {
                'duplicate_iou_threshold': self.config.duplicate_iou_threshold,
                'min_box_area': self.config.min_box_area,
                'max_box_area_ratio': self.config.max_box_area_ratio,
                'min_box_area_ratio': self.config.min_box_area_ratio,
                'occlusion_variance_threshold': self.config.occlusion_variance_threshold,
            },
            'stats': self.stats,
            'results': results,
        }

    def _backup_labels(self, label_path: Path) -> Optional[Path]:
        """
        Create backup of label file before fixing.

        Args:
            label_path: Path to label file

        Returns:
            Path to backup file or None if failed
        """
        try:
            backup_dir = label_path.parent / '.label_qc_backup'
            backup_dir.mkdir(exist_ok=True)

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_name = f"{label_path.stem}_{timestamp}{label_path.suffix}"
            backup_path = backup_dir / backup_name

            shutil.copy2(label_path, backup_path)
            return backup_path
        except Exception:
            return None

    def _fix_duplicate_boxes(self, label_path: Path, duplicate_indices: List[int]) -> bool:
        """
        Remove duplicate boxes from label file.

        Args:
            label_path: Path to label file
            duplicate_indices: List of indices to remove

        Returns:
            True if successful
        """
        if not duplicate_indices:
            return True

        # Backup first
        if self.config.backup_before_fix:
            self._backup_labels(label_path)

        try:
            with open(label_path, 'r') as f:
                lines = f.readlines()

            # Build new content excluding duplicates
            kept_lines = []
            for idx, line in enumerate(lines):
                if idx not in duplicate_indices:
                    kept_lines.append(line)

            with open(label_path, 'w') as f:
                f.writelines(kept_lines)

            return True

        except Exception:
            return False

    def _fix_tiny_boxes(self, label_path: Path, tiny_indices: List[int]) -> bool:
        """
        Remove tiny boxes from label file.

        Args:
            label_path: Path to label file
            tiny_indices: List of indices to remove

        Returns:
            True if successful
        """
        if not tiny_indices:
            return True

        # Backup first
        if self.config.backup_before_fix:
            self._backup_labels(label_path)

        try:
            with open(label_path, 'r') as f:
                lines = f.readlines()

            # Build new content excluding tiny boxes
            kept_lines = []
            for idx, line in enumerate(lines):
                if idx not in tiny_indices:
                    kept_lines.append(line)

            with open(label_path, 'w') as f:
                f.writelines(kept_lines)

            return True

        except Exception:
            return False

    def apply_fixes(self, dataset_path: str) -> Dict[str, Any]:
        """
        Apply auto-fixes for duplicates and tiny boxes.

        Args:
            dataset_path: Path to dataset directory

        Returns:
            Summary of fixes applied
        """
        check_result = self.check(dataset_path)

        fix_summary = {
            'duplicates_fixed': 0,
            'tiny_boxes_fixed': 0,
            'files_modified': [],
            'errors': [],
        }

        for result in check_result['results']:
            label_path = Path(result['path'])

            if result['duplicate_indices']:
                # Get indices to remove (higher index first to avoid shifting issues)
                dup_indices = sorted(set(idx for pair in result['duplicate_indices'] for idx in [pair[0], pair[1]]), reverse=True)
                if self._fix_duplicate_boxes(label_path, dup_indices):
                    fix_summary['duplicates_fixed'] += len(dup_indices)
                    fix_summary['files_modified'].append(str(label_path))

            if result['tiny_indices']:
                tiny_indices = sorted(set(result['tiny_indices']), reverse=True)
                if self._fix_tiny_boxes(label_path, tiny_indices):
                    fix_summary['tiny_boxes_fixed'] += len(tiny_indices)
                    if str(label_path) not in fix_summary['files_modified']:
                        fix_summary['files_modified'].append(str(label_path))

        return fix_summary

    def save_report(self, output_path: str) -> None:
        """
        Save report to file.

        Args:
            output_path: Path to output file
        """
        if self.config.report_format == 'json':
            with open(output_path, 'w') as f:
                json.dump({
                    'config': self.config.__dict__,
                    'stats': self.stats,
                }, f, indent=2, default=str)
        else:
            with open(output_path, 'w') as f:
                f.write(self._build_report())


def main():
    """CLI entry point for label QC tool."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Label Quality Check Tool for YOLO datasets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check labels and show report
  label-qc --dataset /path/to/dataset

  # Check with custom thresholds
  label-qc --dataset /path/to/dataset --duplicate-iou 0.9 --min-area 200

  # Apply auto-fixes
  label-qc --dataset /path/to/dataset --apply-fixes

  # Save report to file
  label-qc --dataset /path/to/dataset --output report.txt
        """
    )

    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        help='Path to dataset directory containing labels'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output file path for report'
    )
    parser.add_argument(
        '--apply-fixes',
        action='store_true',
        help='Apply auto-fixes for duplicates and tiny boxes'
    )
    parser.add_argument(
        '--no-backup',
        action='store_true',
        help='Disable backup before fixing'
    )
    parser.add_argument(
        '--duplicate-iou',
        type=float,
        default=0.8,
        help='IoU threshold for duplicate detection (default: 0.8)'
    )
    parser.add_argument(
        '--min-area',
        type=int,
        default=100,
        help='Minimum box area in pixels (default: 100)'
    )
    parser.add_argument(
        '--max-area-ratio',
        type=float,
        default=0.9,
        help='Maximum box area as ratio of image (default: 0.9)'
    )
    parser.add_argument(
        '--min-area-ratio',
        type=float,
        default=0.001,
        help='Minimum box area as ratio of image (default: 0.001)'
    )
    parser.add_argument(
        '--occlusion-threshold',
        type=float,
        default=50.0,
        help='Variance threshold for occlusion detection (default: 50.0)'
    )
    parser.add_argument(
        '--format',
        type=str,
        choices=['text', 'json'],
        default='text',
        help='Report format (default: text)'
    )

    args = parser.parse_args()

    # Build config
    config = LabelQCConfig(
        duplicate_iou_threshold=args.duplicate_iou,
        min_box_area=args.min_area,
        max_box_area_ratio=args.max_area_ratio,
        min_box_area_ratio=args.min_area_ratio,
        occlusion_variance_threshold=args.occlusion_threshold,
        backup_before_fix=not args.no_backup,
        report_format=args.format,
    )

    # Run check
    checker = LabelQCChecker(config)
    results = checker.check(args.dataset)

    # Print report
    if args.format == 'json':
        print(json.dumps(results, indent=2, default=str))
    else:
        print(checker._build_report())

    # Apply fixes if requested
    if args.apply_fixes:
        print("\n" + "-" * 60)
        print("Applying fixes...")
        print("-" * 60)
        fix_results = checker.apply_fixes(args.dataset)
        print(f"Duplicates fixed: {fix_results['duplicates_fixed']}")
        print(f"Tiny boxes fixed: {fix_results['tiny_boxes_fixed']}")
        print(f"Files modified: {len(fix_results['files_modified'])}")

    # Save report if output specified
    if args.output:
        checker.save_report(args.output)
        print(f"\nReport saved to: {args.output}")


if __name__ == '__main__':
    main()
