"""
导出模块 - 模型导出工具
支持多种格式导出和推理优化
"""

import os
import argparse
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Union
from dataclasses import dataclass, asdict

from ultralytics import YOLO


@dataclass
class ExportConfig:
    """导出配置"""
    model: str = "yolo11n.pt"
    format: str = "onnx"  # onnx, torchscript, engine, openvino, coreml, etc.
    imgsz: Union[int, tuple] = 640
    half: bool = False  # FP16量化
    int8: bool = False  # INT8量化
    dynamic: bool = False  # 动态输入尺寸
    simplify: bool = True  # 简化ONNX模型
    opset: int = 12  # ONNX opset版本
    workspace: float = 4.0  # TensorRT工作空间大小(GiB)
    nms: bool = False  # 添加NMS
    batch: int = 1  # 批处理大小
    device: str = "0"  # 设备
    keras: bool = False  # Keras格式
    optimize: bool = False  # 移动端优化
    fraction: float = 1.0  # INT8校准数据比例
    data: str = "coco8.yaml"  # 数据集配置（用于INT8校准）

    def to_dict(self) -> Dict:
        result = {}
        for k, v in asdict(self).items():
            if v is not None and v != '' and v != []:
                result[k] = v
        return result


class ModelExporter:
    """模型导出器"""

    SUPPORTED_FORMATS = {
        'onnx': {'ext': '.onnx', 'description': 'ONNX Runtime'},
        'torchscript': {'ext': '.torchscript', 'description': 'TorchScript'},
        'engine': {'ext': '.engine', 'description': 'TensorRT'},
        'openvino': {'ext': '_openvino_model/', 'description': 'Intel OpenVINO'},
        'coreml': {'ext': '.mlpackage', 'description': 'Apple CoreML'},
        'saved_model': {'ext': '_saved_model/', 'description': 'TensorFlow SavedModel'},
        'pb': {'ext': '.pb', 'description': 'TensorFlow GraphDef'},
        'tflite': {'ext': '.tflite', 'description': 'TensorFlow Lite'},
        'edgetpu': {'ext': '_edgetpu.tflite', 'description': 'TensorFlow EdgeTPU'},
        'tfjs': {'ext': '_web_model/', 'description': 'TensorFlow.js'},
        'paddle': {'ext': '_paddle_model/', 'description': 'PaddlePaddle'},
        'mnn': {'ext': '.mnn', 'description': 'MNN'},
        'ncnn': {'ext': '_ncnn_model/', 'description': 'NCNN'},
    }

    def __init__(self, model_path: str, config: Union[ExportConfig, Dict, str] = None):
        self.model_path = model_path
        self.model = YOLO(model_path)

        if config is None:
            self.config = ExportConfig(model=model_path)
        elif isinstance(config, dict):
            self.config = ExportConfig(**config)
        elif isinstance(config, ExportConfig):
            self.config = config
        elif isinstance(config, str):
            # 从文件加载
            import yaml
            with open(config, 'r') as f:
                self.config = ExportConfig(**yaml.safe_load(f))

    def export(self, format: Optional[str] = None, **kwargs) -> str:
        """执行模型导出"""
        if format:
            self.config.format = format

        # 更新配置
        for k, v in kwargs.items():
            if hasattr(self.config, k):
                setattr(self.config, k, v)

        print(f"开始导出模型: {self.model_path}")
        print(f"目标格式: {self.config.format}")
        print(f"配置: {self.config.to_dict()}")

        # 执行导出
        export_path = self.model.export(**self.config.to_dict())

        print(f"导出完成: {export_path}")
        return export_path

    def export_multi_format(self, formats: List[str] = None) -> Dict[str, str]:
        """导出为多种格式"""
        if formats is None:
            formats = ['onnx', 'torchscript', 'openvino']

        results = {}
        for fmt in formats:
            print(f"\n{'='*50}")
            print(f"导出格式: {fmt}")
            try:
                path = self.export(format=fmt)
                results[fmt] = path
            except Exception as e:
                print(f"导出失败: {e}")
                results[fmt] = None

        return results

    @staticmethod
    def benchmark_export(model_path: str, format: str = 'onnx', imgsz: int = 640) -> Dict:
        """基准测试导出的模型"""
        from ultralytics.utils.benchmarks import benchmark

        results = benchmark(
            model=model_path,
            data='coco8.yaml',
            imgsz=imgsz,
            half=True,
            device=0
        )
        return results

    @staticmethod
    def compare_models(models: List[str], data: str = 'coco8.yaml') -> Dict:
        """比较多个模型的性能"""
        results = {}

        for model_path in models:
            print(f"\n评估模型: {model_path}")
            model = YOLO(model_path)

            # 验证
            metrics = model.val(data=data)

            results[model_path] = {
                'mAP50': float(metrics.box.map50),
                'mAP50-95': float(metrics.box.map),
                'speed': metrics.speed
            }

        return results


class InferenceOptimizer:
    """推理优化工具"""

    @staticmethod
    def optimize_for_tensorrt(
        model_path: str,
        output_path: str = None,
        imgsz: int = 640,
        half: bool = True,
        int8: bool = False,
        workspace: float = 4.0
    ) -> str:
        """优化为TensorRT格式"""
        model = YOLO(model_path)

        export_params = {
            'format': 'engine',
            'imgsz': imgsz,
            'half': half,
            'workspace': workspace,
        }

        if int8:
            export_params['int8'] = True
            export_params['data'] = 'coco8.yaml'

        export_path = model.export(**export_params)
        return export_path

    @staticmethod
    def optimize_for_openvino(
        model_path: str,
        output_path: str = None,
        imgsz: int = 640,
        half: bool = False
    ) -> str:
        """优化为OpenVINO格式"""
        model = YOLO(model_path)

        export_params = {
            'format': 'openvino',
            'imgsz': imgsz,
            'half': half,
        }

        export_path = model.export(**export_params)
        return export_path

    @staticmethod
    def optimize_for_edge(
        model_path: str,
        output_path: str = None,
        imgsz: int = 320
    ) -> str:
        """优化为边缘设备格式 (TensorFlow Lite)"""
        model = YOLO(model_path)

        export_path = model.export(
            format='tflite',
            imgsz=imgsz,
            int8=True,
            half=False
        )
        return export_path

    @staticmethod
    def batch_inference(
        model_path: str,
        source: str,
        batch_size: int = 8,
        imgsz: int = 640,
        device: str = "0"
    ):
        """批量推理"""
        model = YOLO(model_path)

        results = model.predict(
            source=source,
            batch=batch_size,
            imgsz=imgsz,
            device=device,
            stream=True  # 使用流式处理提高效率
        )

        return results


def main():
    parser = argparse.ArgumentParser(description='YOLO模型导出工具')
    parser.add_argument('--model', type=str, required=True, help='模型路径')
    parser.add_argument('--format', type=str, default='onnx',
                        choices=['onnx', 'torchscript', 'engine', 'openvino',
                                'coreml', 'tflite', 'ncnn', 'mnn'],
                        help='导出格式')
    parser.add_argument('--imgsz', type=int, default=640, help='图像尺寸')
    parser.add_argument('--half', action='store_true', help='FP16量化')
    parser.add_argument('--int8', action='store_true', help='INT8量化')
    parser.add_argument('--dynamic', action='store_true', help='动态输入尺寸')
    parser.add_argument('--workspace', type=float, default=4.0, help='TensorRT工作空间')
    parser.add_argument('--batch', type=int, default=1, help='批处理大小')
    parser.add_argument('--device', type=str, default='0', help='设备')
    parser.add_argument('--config', type=str, help='配置文件路径')
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='日志级别')

    args = parser.parse_args()

    import logging as log_module
    log_level = getattr(log_module, args.log_level.upper(), log_module.INFO)
    log_module.basicConfig(
        level=log_level,
        format='%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    exporter = ModelExporter(args.model)

    print(f"导出模型: {args.model}")
    print(f"目标格式: {args.format}")

    if args.format == 'engine' and args.int8:
        print("提示: INT8量化需要提供数据集配置")

    export_path = exporter.export(
        format=args.format,
        imgsz=args.imgsz,
        half=args.half,
        int8=args.int8,
        dynamic=args.dynamic,
        workspace=args.workspace,
        batch=args.batch,
        device=args.device
    )

    print(f"\n导出成功: {export_path}")


if __name__ == '__main__':
    main()
