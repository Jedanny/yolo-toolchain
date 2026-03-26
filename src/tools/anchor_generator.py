"""
Anchor 生成模块 - 从 YOLO 数据集自动生成最优 anchor boxes
使用 k-means 聚类 (IoU-based distance) + 轮廓系数自动确定最优 k
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict

import numpy as np
import yaml

logger = logging.getLogger("yolo_toolchain.anchor_generator")


@dataclass
class AnchorConfig:
    """Anchor 生成配置"""
    n_clusters_min: int = 5       # 最小聚类数
    n_clusters_max: int = 15      # 最大聚类数
    scales: List[str] = field(default_factory=lambda: ['P3', 'P4', 'P5'])
    min_bbox_area: float = 0.0001  # 忽略过小 bbox
    validate: bool = False         # 是否验证 anchors
    output_format: str = "yaml"    # 输出格式


def compute_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """
    计算单个边界框与一组边界框的 IoU

    Args:
        box: 单个边界框 [N, 4] (x_center, y_center, width, height) 归一化
        boxes: 一组边界框 [M, 4]

    Returns:
        IoU 值 [M]
    """
    # 转换为 x1, y1, x2, y2 格式
    box_x1 = box[0] - box[2] / 2
    box_y1 = box[1] - box[3] / 2
    box_x2 = box[0] + box[2] / 2
    box_y2 = box[1] + box[3] / 2

    boxes_x1 = boxes[:, 0] - boxes[:, 2] / 2
    boxes_y1 = boxes[:, 1] - boxes[:, 3] / 2
    boxes_x2 = boxes[:, 0] + boxes[:, 2] / 2
    boxes_y2 = boxes[:, 1] + boxes[:, 3] / 2

    # 计算交集区域
    inter_x1 = np.maximum(box_x1, boxes_x1)
    inter_y1 = np.maximum(box_y1, boxes_y1)
    inter_x2 = np.minimum(box_x2, boxes_x2)
    inter_y2 = np.minimum(box_y2, boxes_y2)

    inter_area = np.maximum(0, inter_x2 - inter_x1) * np.maximum(0, inter_y2 - inter_y1)

    # 计算各自的面积
    box_area = box[2] * box[3]
    boxes_area = boxes[:, 2] * boxes[:, 3]

    # 计算并集
    union_area = box_area + boxes_area - inter_area

    # 计算 IoU
    iou = np.where(union_area > 0, inter_area / union_area, 0)

    return iou


def iou_distance(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """
    基于 1 - IoU 的距离度量 (用于 k-means)

    Args:
        box: 单个边界框 [4] (x_center, y_center, width, height)
        boxes: 一组边界框 [N, 4]

    Returns:
        距离值 [N], 越小表示越相似
    """
    return 1 - compute_iou(box, boxes)


def kmeans_iou(bboxes: np.ndarray, n_clusters: int, max_iter: int = 300, tol: float = 1e-6) -> Tuple[np.ndarray, float]:
    """
    使用 IoU-based 距离的 k-means 聚类

    Args:
        bboxes: 边界框数组 [N, 4] (x_center, y_center, width, height) 归一化
        n_clusters: 聚类数
        max_iter: 最大迭代次数
        tol: 收敛阈值

    Returns:
        (centroids, avg_iou) - 聚类中心和平均 IoU
    """
    n_samples = len(bboxes)

    if n_clusters > n_samples:
        logger.warning(f"n_clusters ({n_clusters}) > 样本数 ({n_samples}), 使用样本数作为聚类数")
        n_clusters = n_samples

    # 随机选择初始聚类中心
    indices = np.random.choice(n_samples, n_clusters, replace=False)
    centroids = bboxes[indices].copy()

    for iteration in range(max_iter):
        # 分配每个边界框到最近的聚类中心
        distances = np.zeros((n_samples, n_clusters))
        for i, bbox in enumerate(bboxes):
            distances[i] = iou_distance(bbox, centroids)

        cluster_assignments = np.argmin(distances, axis=1)

        # 计算新的聚类中心
        new_centroids = np.zeros_like(centroids)
        for k in range(n_clusters):
            mask = cluster_assignments == k
            if np.sum(mask) > 0:
                new_centroids[k] = bboxes[mask].mean(axis=0)
            else:
                # 如果空簇，随机选择一个点作为中心
                new_centroids[k] = bboxes[np.random.choice(n_samples)]

        # 检查收敛
        centroid_shift = np.sqrt(((new_centroids - centroids) ** 2).sum(axis=1)).max()

        if centroid_shift < tol:
            logger.debug(f"K-means 收敛于第 {iteration + 1} 次迭代")
            centroids = new_centroids
            break

        centroids = new_centroids

    # 计算平均 IoU (聚类质量)
    total_iou = 0.0
    for i, bbox in enumerate(bboxes):
        cluster_id = cluster_assignments[i]
        iou = compute_iou(bbox, centroids[cluster_id:cluster_id + 1])[0]
        total_iou += iou

    avg_iou = total_iou / n_samples

    return centroids, avg_iou


def silhouette_score(bboxes: np.ndarray, cluster_assignments: np.ndarray, centroids: np.ndarray) -> float:
    """
    计算轮廓系数评估聚类质量

    Args:
        bboxes: 边界框数组 [N, 4]
        cluster_assignments: 聚类分配 [N]
        centroids: 聚类中心 [K, 4]

    Returns:
        轮廓系数 [-1, 1], 越大表示聚类质量越好
    """
    n_samples = len(bboxes)
    n_clusters = len(centroids)

    if n_clusters == 1 or n_clusters == n_samples:
        return 1.0

    # 计算每个样本的轮廓系数
    s = np.zeros(n_samples)

    for i in range(n_samples):
        cluster_id = cluster_assignments[i]

        # 计算 a_i: 同簇内其他点到该点的平均距离
        same_cluster_mask = cluster_assignments == cluster_id
        same_cluster_indices = np.where(same_cluster_mask)[0]
        same_cluster_indices = same_cluster_indices[same_cluster_indices != i]

        if len(same_cluster_indices) == 0:
            s[i] = 0
            continue

        # 使用 IoU-based 距离
        a_i_distances = []
        for j in same_cluster_indices:
            dist = iou_distance(bboxes[i], bboxes[j:j + 1])[0]
            a_i_distances.append(dist)
        a_i = np.mean(a_i_distances)

        # 计算 b_i: 到最近其他簇的最小平均距离
        b_i_distances = []
        for k in range(n_clusters):
            if k == cluster_id:
                continue
            other_cluster_mask = cluster_assignments == k
            other_cluster_indices = np.where(other_cluster_mask)[0]

            if len(other_cluster_indices) == 0:
                continue

            b_k_distances = []
            for j in other_cluster_indices:
                dist = iou_distance(bboxes[i], bboxes[j:j + 1])[0]
                b_k_distances.append(dist)
            b_i_distances.append(np.mean(b_k_distances))

        if len(b_i_distances) == 0:
            b_i = 0
        else:
            b_i = min(b_i_distances)

        # 轮廓系数
        if a_i + b_i > 0:
            s[i] = (b_i - a_i) / (a_i + b_i)
        else:
            s[i] = 0

    return np.mean(s)


class AnchorGenerator:
    """
    Anchor Box 生成器
    使用 k-means 聚类从 YOLO 数据集生成最优 anchor boxes
    """

    def __init__(self, config: Optional[AnchorConfig] = None):
        """
        初始化 Anchor 生成器

        Args:
            config: AnchorConfig 配置对象
        """
        self.config = config or AnchorConfig()
        self.bboxes: Optional[np.ndarray] = None
        self.anchors: Optional[Dict[str, np.ndarray]] = None
        self.dataset_info: Dict = {}

    def load_dataset(self, data_yaml: str) -> int:
        """
        加载 YOLO 数据集

        Args:
            data_yaml: 数据集配置文件路径 (data.yaml)

        Returns:
            加载的边界框数量
        """
        data_yaml = Path(data_yaml)

        if not data_yaml.exists():
            raise FileNotFoundError(f"数据集配置文件不存在: {data_yaml}")

        with open(data_yaml, 'r') as f:
            data_config = yaml.safe_load(f)

        # 获取数据集路径
        dataset_path = data_config.get('path', str(data_yaml.parent))
        dataset_path = Path(dataset_path)

        # 获取训练图像目录
        train_path = data_config.get('train', '')
        if not train_path:
            raise ValueError("数据集配置中未找到 'train' 路径")

        train_dir = dataset_path / train_path
        if not train_dir.exists():
            raise FileNotFoundError(f"训练集目录不存在: {train_dir}")

        # 解析所有标签文件
        all_bboxes = []
        label_files = list(train_dir.glob('**/*.txt'))

        logger.info(f"找到 {len(label_files)} 个标签文件")

        for label_file in label_files:
            bboxes = self._parse_labels(label_file)
            all_bboxes.extend(bboxes)

        if not all_bboxes:
            raise ValueError(f"未找到任何有效的边界框: {train_dir}")

        # 过滤过小的边界框
        filtered_bboxes = []
        for bbox in all_bboxes:
            area = bbox[2] * bbox[3]
            if area >= self.config.min_bbox_area:
                filtered_bboxes.append(bbox)

        self.bboxes = np.array(filtered_bboxes)

        logger.info(f"加载了 {len(self.bboxes)} 个有效边界框 (过滤了 {len(all_bboxes) - len(filtered_bboxes)} 个过小框)")

        # 保存数据集信息
        self.dataset_info = {
            'path': str(dataset_path),
            'train': str(train_dir),
            'n_images': len(label_files),
            'n_bboxes': len(self.bboxes)
        }

        return len(self.bboxes)

    def _parse_labels(self, label_file: Path) -> List[np.ndarray]:
        """
        解析单个标签文件

        Args:
            label_file: 标签文件路径

        Returns:
            边界框列表 [N, 4] (x_center, y_center, width, height) 归一化
        """
        bboxes = []

        if not label_file.exists():
            return bboxes

        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    # YOLO 格式: class x_center y_center width height
                    try:
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        bboxes.append([x_center, y_center, width, height])
                    except ValueError:
                        continue

        return bboxes

    def _assign_to_scale(self, bboxes: np.ndarray) -> Dict[str, np.ndarray]:
        """
        根据边界框尺寸分配到不同 scale

        YOLOv8 尺度定义:
        - P3/P4/P5: 用于小/中/大目标检测
        - P3 对应 stride=8, 特征图尺寸较大
        - P5 对应 stride=32, 特征图尺寸较小

        Args:
            bboxes: 边界框数组 [N, 4]

        Returns:
            各尺度的边界框字典
        """
        # 计算边界框面积 (用于确定尺度)
        areas = bboxes[:, 2] * bboxes[:, 3]

        # 转换为像素面积 (假设输入是归一化的，输出应该是针对原图尺寸)
        # 这里使用相对面积来划分尺度

        # 根据面积大小排序
        sorted_indices = np.argsort(areas)

        n = len(bboxes)
        scale_bboxes = {scale: [] for scale in self.config.scales}

        if n == 0:
            return scale_bboxes

        # 使用百分位数划分尺度
        # 小目标 (P3): 面积较小的 40%
        # 中目标 (P4): 面积中等的 40%
        # 大目标 (P5): 面积较大的 20%
        percentiles = [0, 40, 80, 100]

        for i, scale in enumerate(self.config.scales):
            low = np.percentile(areas, percentiles[i])
            high = np.percentile(areas, percentiles[i + 1])

            mask = (areas >= low) & (areas <= high)
            scale_bboxes[scale] = bboxes[mask]

        return scale_bboxes

    def generate(self) -> Dict[str, np.ndarray]:
        """
        生成最优 anchor boxes

        Returns:
            各尺度的 anchor boxes 字典
        """
        if self.bboxes is None:
            raise ValueError("请先调用 load_dataset 加载数据集")

        # 按尺度分配边界框
        scale_bboxes = self._assign_to_scale(self.bboxes)

        anchors = {}
        best_k_scores = {}

        for scale, scale_bboxes_data in scale_bboxes.items():
            if len(scale_bboxes_data) < self.config.n_clusters_min:
                logger.warning(f"Scale {scale} 边界框数量 ({len(scale_bboxes_data)}) 少于最小聚类数, 使用所有可用框")
                if len(scale_bboxes_data) > 0:
                    k_to_use = max(1, len(scale_bboxes_data))
                else:
                    k_to_use = self.config.n_clusters_min
            else:
                k_to_use = self.config.n_clusters_min

            # 寻找最优 k 值
            best_k = k_to_use
            best_score = -1
            best_avg_iou = 0

            search_k_range = range(
                max(2, len(scale_bboxes_data) // 10 if len(scale_bboxes_data) >= 10 else k_to_use),
                min(self.config.n_clusters_max, len(scale_bboxes_data))
            )

            if len(search_k_range) == 0 or len(scale_bboxes_data) < self.config.n_clusters_min:
                search_k_range = [k_to_use]

            logger.info(f"Scale {scale}: 搜索最优 k (范围: {list(search_k_range)})")

            for k in search_k_range:
                # 运行 k-means
                centroids, avg_iou = kmeans_iou(scale_bboxes_data, k)

                # 计算轮廓系数
                # 临时分配用于计算轮廓系数
                distances = np.zeros((len(scale_bboxes_data), k))
                for i, bbox in enumerate(scale_bboxes_data):
                    distances[i] = iou_distance(bbox, centroids)
                temp_assignments = np.argmin(distances, axis=1)

                silhouette = silhouette_score(scale_bboxes_data, temp_assignments, centroids)

                # 综合评分: 优先轮廓系数，次要平均 IoU
                # 轮廓系数范围 [-1, 1], IoU 范围 [0, 1]
                score = 0.7 * silhouette + 0.3 * avg_iou

                logger.debug(f"  k={k}: avg_iou={avg_iou:.4f}, silhouette={silhouette:.4f}, score={score:.4f}")

                if score > best_score:
                    best_score = score
                    best_k = k
                    best_avg_iou = avg_iou
                    best_centroids = centroids

            logger.info(f"Scale {scale}: 最优 k={best_k}, avg_iou={best_avg_iou:.4f}")

            # 调整 anchors 到合适的尺度 (乘以特征图 stride 后的相对原图的比例)
            # YOLOv8 anchors 使用相对于特征图的尺寸，需要转换到相对于原图
            # 这里直接使用归一化的值
            anchors[scale] = best_centroids
            best_k_scores[scale] = best_k

        self.anchors = anchors

        # 保存额外信息
        self.dataset_info['best_k'] = best_k_scores

        return anchors

    def save_anchors(self, output_path: str) -> str:
        """
        保存 anchor boxes 到文件

        Args:
            output_path: 输出文件路径

        Returns:
            保存的文件路径
        """
        if self.anchors is None:
            raise ValueError("请先调用 generate() 生成 anchors")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 转换为可序列化格式
        anchors_dict = {}
        for scale, anchors in self.anchors.items():
            # 转换为列表格式 (每行一个 anchor: width height)
            anchors_dict[scale] = anchors[:, 2:].tolist()  # 只保存 width, height

        # 构建输出数据
        output_data = {
            'anchors': anchors_dict,
            'metadata': {
                'n_bboxes': int(self.dataset_info.get('n_bboxes', 0)),
                'n_images': int(self.dataset_info.get('n_images', 0)),
                'best_k': {k: int(v) for k, v in self.dataset_info.get('best_k', {}).items()}
            }
        }

        if self.config.output_format == 'yaml':
            with open(output_path, 'w') as f:
                yaml.dump(output_data, f, default_flow_style=False)
        else:
            # 简单文本格式
            with open(output_path, 'w') as f:
                for scale, scale_anchors in anchors_dict.items():
                    f.write(f"# Scale {scale}\n")
                    for anchor in scale_anchors:
                        f.write(f"{anchor[0]:.6f} {anchor[1]:.6f}\n")

        logger.info(f"Anchors 已保存到: {output_path}")

        # 同时输出 YOLOv8 格式的 anchors (用于直接替换 yaml 文件)
        yolov8_format = []
        for scale in self.config.scales:
            if scale in self.anchors:
                scale_anchors = self.anchors[scale]
                for anchor in scale_anchors:
                    yolov8_format.extend([anchor[2], anchor[3]])

        logger.info(f"YOLOv8 格式 anchors: {yolov8_format}")

        return str(output_path)

    def validate_with_anchors(self, data_yaml: str) -> Dict[str, float]:
        """
        使用生成的 anchors 验证数据集

        Args:
            data_yaml: 数据集配置文件

        Returns:
            验证指标字典
        """
        if self.anchors is None:
            raise ValueError("请先调用 generate() 生成 anchors")

        # 重新加载数据集以验证
        if self.bboxes is None:
            self.load_dataset(data_yaml)

        # 计算整体平均 IoU
        all_anchor_iou = []
        for bbox in self.bboxes:
            max_iou = 0
            for scale_anchors in self.anchors.values():
                for anchor in scale_anchors:
                    iou = compute_iou(bbox, anchor)
                    max_iou = max(max_iou, iou)
            all_anchor_iou.append(max_iou)

        avg_iou = np.mean(all_anchor_iou)

        # 按尺度计算 IoU
        scale_bboxes = self._assign_to_scale(self.bboxes)
        scale_ious = {}

        for scale, scale_bbox_data in scale_bboxes.items():
            if len(scale_bbox_data) > 0 and scale in self.anchors:
                scale_iou_list = []
                for bbox in scale_bbox_data:
                    max_iou = 0
                    for anchor in self.anchors[scale]:
                        iou = compute_iou(bbox, anchor)
                        max_iou = max(max_iou, iou)
                    scale_iou_list.append(max_iou)
                scale_ious[scale] = np.mean(scale_iou_list)

        metrics = {
            'avg_iou': float(avg_iou),
            'scale_ious': {k: float(v) for k, v in scale_ious.items()}
        }

        logger.info(f"验证结果: avg_iou={avg_iou:.4f}")
        for scale, iou in scale_ious.items():
            logger.info(f"  {scale}: {iou:.4f}")

        return metrics


def main():
    """CLI 入口点"""
    import argparse

    parser = argparse.ArgumentParser(description='YOLO Anchor 生成工具')
    parser.add_argument('--data', type=str, required=True,
                        help='数据集配置文件 (data.yaml)')
    parser.add_argument('--output', type=str, default='anchors.yaml',
                        help='输出文件路径 (默认: anchors.yaml)')
    parser.add_argument('--min_k', type=int, default=5,
                        help='最小聚类数 (默认: 5)')
    parser.add_argument('--max_k', type=int, default=15,
                        help='最大聚类数 (默认: 15)')
    parser.add_argument('--scales', type=str, default='P3,P4,P5',
                        help='使用的尺度 (默认: P3,P4,P5)')
    parser.add_argument('--validate', action='store_true',
                        help='验证生成的 anchors')
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='日志级别')

    args = parser.parse_args()

    # 配置日志
    import logging as log_module
    log_level = getattr(log_module, args.log_level.upper(), log_module.INFO)
    log_module.basicConfig(
        level=log_level,
        format='%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 解析配置
    scales = [s.strip() for s in args.scales.split(',')]

    config = AnchorConfig(
        n_clusters_min=args.min_k,
        n_clusters_max=args.max_k,
        scales=scales,
        validate=args.validate
    )

    # 生成 anchors
    generator = AnchorGenerator(config)

    print(f"加载数据集: {args.data}")
    generator.load_dataset(args.data)

    print("开始生成 anchors...")
    anchors = generator.generate()

    print("保存 anchors...")
    output_path = generator.save_anchors(args.output)

    # 验证
    if args.validate:
        print("验证 anchors...")
        metrics = generator.validate_with_anchors(args.data)
        print(f"平均 IoU: {metrics['avg_iou']:.4f}")

    print(f"完成! Anchors 已保存到: {output_path}")


if __name__ == '__main__':
    main()
