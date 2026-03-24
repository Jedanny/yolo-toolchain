"""
模型下载模块 - 从 Hugging Face 下载 YOLO 预训练模型
"""

import argparse
import logging
import shutil
from pathlib import Path
from typing import Dict

logger = logging.getLogger("yolo_toolchain.downloader")

YOLO_PRETRAINED_MODELS: Dict[str, str] = {
    "yolo11n": "ultralytics/YOLO11",
    "yolo11s": "ultralytics/YOLO11",
    "yolo11m": "ultralytics/YOLO11",
    "yolo11l": "ultralytics/YOLO11",
    "yolo11x": "ultralytics/YOLO11",
    "yolov8n": "ultralytics/yolov8",
    "yolov8s": "ultralytics/yolov8",
    "yolov8m": "ultralytics/yolov8",
    "yolov8l": "ultralytics/yolov8",
    "yolov8x": "ultralytics/yolov8",
    "yolov9n": "ultralytics/yolov9",
    "yolov9s": "ultralytics/yolov9",
    "yolov9m": "ultralytics/yolov9",
    "yolov9c": "ultralytics/yolov9",
    "yolov9e": "ultralytics/yolov9",
    "yolov10n": "ultralytics/yolov10",
    "yolov10s": "ultralytics/yolov10",
    "yolov10m": "ultralytics/yolov10",
    "yolov10l": "ultralytics/yolov10",
    "yolov10x": "ultralytics/yolov10",
}


def download_yolo_model(
    model_name: str,
    output_dir: str = "models",
    file_name: str = None,
    hub_repo: str = None
) -> str:
    """从 Hugging Face 下载 YOLO 预训练模型

    Args:
        model_name: 模型名称 (如 yolo11n, yolov8m, yolov9c)
        output_dir: 输出目录
        file_name: 自定义文件名（可选）
        hub_repo: Hugging Face 仓库（可选，优先级高于内置映射）

    Returns:
        下载的模型文件路径
    """
    from huggingface_hub import hf_hub_download

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if hub_repo is None:
        if model_name not in YOLO_PRETRAINED_MODELS:
            raise ValueError(
                f"Unknown model: {model_name}. Available: {list(YOLO_PRETRAINED_MODELS.keys())}"
            )
        repo_id = YOLO_PRETRAINED_MODELS[model_name]
    else:
        repo_id = hub_repo

    model_file = f"{model_name}.pt"
    if file_name:
        save_file = file_name
    else:
        save_file = f"{model_name}.pt"

    save_path = output_path / save_file

    if save_path.exists():
        logger.info(f"Model already exists: {save_path}")
        return str(save_path)

    logger.info(f"Downloading {model_name} from Hugging Face: {repo_id}")
    downloaded_path = hf_hub_download(
        repo_id=repo_id,
        filename=model_file,
        local_dir=str(output_path),
        local_dir_use_symlinks=False
    )

    if Path(downloaded_path).parent != output_path:
        shutil.move(downloaded_path, save_path)

    logger.info(f"Model saved to: {save_path}")
    return str(save_path)


def list_available_models() -> Dict[str, str]:
    """列出所有可用的预训练模型

    Returns:
        模型名称到仓库的映射字典
    """
    return YOLO_PRETRAINED_MODELS.copy()


def main():
    parser = argparse.ArgumentParser(description='下载 YOLO 预训练模型')
    parser.add_argument('--model', type=str,
                        help='模型名称 (yolo11n, yolov8m, yolov9c 等)')
    parser.add_argument('--output', type=str, default='models',
                        help='输出目录 (默认 models)')
    parser.add_argument('--list', action='store_true',
                        help='列出所有可用模型')
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])

    args = parser.parse_args()

    import logging as log_module
    log_level = getattr(log_module, args.log_level.upper(), log_module.INFO)
    log_module.basicConfig(
        level=log_level,
        format='%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    if args.list:
        print("\n可用模型:")
        print(f"{'模型':<15} {'仓库'}")
        print("-" * 50)
        for name, repo in YOLO_PRETRAINED_MODELS.items():
            print(f"{name:<15} {repo}")
        print()
        return

    if not args.model:
        parser.error("--model is required unless --list is specified")

    download_yolo_model(args.model, args.output)


if __name__ == '__main__':
    main()
