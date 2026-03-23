"""
数据自动标注模块 - 使用视觉大模型自动标注 YOLO 格式目标检测数据
支持 SiliconFlow Pro/moonshotai/Kimi-K2.5 模型

配置方式（按优先级从高到低）：
1. 命令行参数 --api_key 和 --model
2. 环境变量 SILICONFLOW_API_KEY 和 SILICONFLOW_MODEL
3. .env 文件中的配置
"""

import os
import base64
import json
import argparse
import re
import time
import logging
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple
from dataclasses import dataclass
import requests
from PIL import Image
import yaml
import io

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger("yolo_toolchain.auto_annotator")


@dataclass
class AutoAnnotatorConfig:
    """自动标注配置"""
    api_key: str = ""
    base_url: str = "https://api.siliconflow.cn/v1"
    model: str = "Pro/moonshotai/Kimi-K2.5"
    max_tokens: int = 4096
    temperature: float = 0.1
    image_size: int = 1024
    confidence_threshold: float = 0.25
    max_retries: int = 3
    retry_delay: float = 2.0
    batch_size: int = 1


class SiliconFlowClient:
    """SiliconFlow API 客户端"""

    def __init__(self, api_key: str, base_url: str = "https://api.siliconflow.cn/v1"):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.endpoint = f"{self.base_url}/chat/completions"

    def chat(self, messages: List[Dict], model: str, **kwargs) -> Dict:
        """发送聊天请求"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": model,
            "messages": messages,
            **kwargs
        }

        response = requests.post(self.endpoint, headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        return response.json()


def encode_image_base64(image_path: str, max_size: int = 1024) -> str:
    """将图片编码为 base64 字符串"""
    img = Image.open(image_path)

    width, height = img.size
    if max(width, height) > max_size:
        scale = max_size / max(width, height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        img = img.resize((new_width, new_height), Image.LANCZOS)

    img_bytes = img.convert("RGB").tobytes()
    return base64.b64encode(img_bytes).decode("utf-8")


def encode_image_to_data_url(image_path: str, max_size: int = 1024) -> str:
    """将图片转换为 data URL 格式"""
    img = Image.open(image_path)

    width, height = img.size
    if max(width, height) > max_size:
        scale = max_size / max(width, height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        img = img.resize((new_width, new_height), Image.LANCZOS)

    buffer = io.BytesIO()
    img.convert("RGB").save(buffer, format="JPEG", quality=85)
    img_bytes = buffer.getvalue()
    b64_str = base64.b64encode(img_bytes).decode("utf-8")

    return f"data:image/jpeg;base64,{b64_str}"


def load_classes_from_yaml(yaml_path: str) -> List[str]:
    """从 YOLO 数据集配置文件中加载类别列表"""
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    names = config.get('names', {})
    if isinstance(names, dict):
        classes = [names[i] for i in sorted(names.keys())]
    elif isinstance(names, list):
        classes = names
    else:
        classes = []
    return classes


def build_annotation_prompt(classes: List[str] = None, custom_note: str = None) -> str:
    """根据类别构建标注提示词"""
    base_prompt = """你是一个专业的目标检测标注助手。请分析这张图片，用 YOLO 格式标注所有可见的目标物体。

## 输出格式要求
严格按照以下 JSON 格式输出，不要添加任何解释：
```json
{{
  "width": 图片宽度像素值,
  "height": 图片高度像素值,
  "classes": ["类别1", "类别2", ...],
  "annotations": [
    {{"class_id": 0, "bbox": [center_x, center_y, width, height], "confidence": 0.95}},
    ...
  ]
}}
```

## 标注规则
1. bbox 坐标为归一化值 (0-1)：center_x, center_y, width, height
2. class_id 对应 classes 数组的索引
3. confidence 为置信度 (0-1)，请为每个检测提供准确的置信度分数
4. 只标注确定的目标，忽略不确定的区域
5. 标注所有可见的物体，包括重叠和部分遮挡的
6. 置信度低于 0.25 的目标请忽略，不要返回"""

    if classes:
        classes_str = ', '.join(classes)
        base_prompt += f"""

## 指定类别（必须严格使用）
可选类别列表：[{classes_str}]
- 只标注列表中的类别
- 不要标注不在列表中的物体
- 如果图中没有列表中的类别，返回空标注"""
    else:
        base_prompt += """

## 通用类别参考（如未指定类别）
person, car, truck, bus, motorcycle, bicycle, dog, cat, bird, chair, table, cup, bottle, book, phone, laptop"""

    if custom_note:
        base_prompt += f"\n\n## 额外要求\n{custom_note}"

    base_prompt += "\n\n请直接输出 JSON，不要有其他内容："
    return base_prompt


class AutoAnnotator:
    """图片自动标注器"""

    def __init__(self, config: Union[AutoAnnotatorConfig, Dict, str]):
        if isinstance(config, dict):
            self.config = AutoAnnotatorConfig(**config)
        elif isinstance(config, str):
            with open(config, 'r') as f:
                self.config = AutoAnnotatorConfig(**yaml.safe_load(f))
        else:
            self.config = config

        api_key = self.config.api_key or os.environ.get("SILICONFLOW_API_KEY", "")
        if not api_key:
            raise ValueError("API key is required. Set SILICONFLOW_API_KEY environment variable or provide in config.")

        self.client = SiliconFlowClient(api_key, self.config.base_url)
        self.class_names = []

    def annotate_image(self, image_path: str, classes: List[str] = None, dataset_yaml: str = None) -> Tuple[List, List]:
        """标注单张图片

        Args:
            image_path: 图片路径
            classes: 类别列表（优先于 dataset_yaml）
            dataset_yaml: YOLO 数据集配置文件路径

        Returns:
            (annotations, class_names): 标注列表和类别名称列表
        """
        img = Image.open(image_path)
        width, height = img.size

        if classes is None and dataset_yaml:
            classes = load_classes_from_yaml(dataset_yaml)

        prompt = build_annotation_prompt(classes=classes)

        image_data = encode_image_to_data_url(image_path, self.config.image_size)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_data}}
                ]
            }
        ]

        retry_count = 0
        last_error = None

        while retry_count < self.config.max_retries:
            try:
                response = self.client.chat(
                    messages=messages,
                    model=self.config.model,
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature
                )

                content = response["choices"][0]["message"]["content"]
                result = self._parse_json_response(content)

                if result and "annotations" in result:
                    self.class_names = result.get("classes", [])
                    annotations = self._filter_by_confidence(result["annotations"])
                    return annotations, self.class_names

                retry_count += 1
                last_error = "Invalid response format"

            except requests.exceptions.RequestException as e:
                retry_count += 1
                last_error = str(e)
                time.sleep(self.config.retry_delay)
            except (KeyError, IndexError, json.JSONDecodeError) as e:
                retry_count += 1
                last_error = f"Parse error: {e}"
                time.sleep(self.config.retry_delay)

        raise RuntimeError(f"Failed to annotate after {self.config.max_retries} retries. Last error: {last_error}")

    def _parse_json_response(self, content: str) -> Optional[Dict]:
        """解析 JSON 响应"""
        content = content.strip()

        json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
        if json_match:
            content = json_match.group(1)

        if content.startswith("```"):
            content = content.split("```", 2)[1]
            if content.startswith("json"):
                content = content[4:]

        content = content.strip()

        try:
            return json.loads(content)
        except json.JSONDecodeError:
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group(0))
                except json.JSONDecodeError:
                    pass
        return None

    def _filter_by_confidence(self, annotations: List[Dict]) -> List[Dict]:
        """根据置信度阈值过滤标注

        Args:
            annotations: 原始标注列表

        Returns:
            过滤后的标注列表
        """
        threshold = self.config.confidence_threshold
        filtered = []
        for ann in annotations:
            confidence = ann.get('confidence', 1.0)
            if confidence >= threshold:
                filtered.append(ann)
        if filtered:
            logger.debug(f"Kept {len(filtered)}/{len(annotations)} annotations (conf >= {threshold})")
        return filtered

    def annotate_batch(self, image_paths: List[str], classes: List[str] = None) -> Dict[str, Tuple[List, List]]:
        """批量标注图片

        Returns:
            Dict[image_path, (annotations, class_names)]
        """
        results = {}

        for i, image_path in enumerate(image_paths):
            logger.info(f"[{i+1}/{len(image_paths)}] Processing: {image_path}")
            try:
                annotations, class_names = self.annotate_image(image_path, classes)
                results[image_path] = (annotations, class_names)
            except Exception as e:
                logger.error(f"Error processing {image_path}: {e}")
                results[image_path] = ([], [])

        return results

    def save_annotations(self, annotations: List, class_names: List[str], output_path: str):
        """保存标注为 YOLO 格式"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            for ann in annotations:
                class_id = ann.get('class_id', 0)
                bbox = ann.get('bbox', [])
                confidence = ann.get('confidence', 1.0)

                if len(bbox) >= 4:
                    cx, cy, w, h = bbox
                    f.write(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

    def create_dataset_yaml(self, output_path: str, class_names: List[str]):
        """创建 YOLO 数据集配置文件"""
        yaml_content = {
            'path': str(Path(output_path).parent.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'nc': len(class_names),
            'names': {i: name for i, name in enumerate(class_names)}
        }

        with open(output_path, 'w') as f:
            yaml.dump(yaml_content, f, default_flow_style=False, allow_unicode=True)


def auto_annotate_dataset(
    image_dir: str,
    output_dir: str,
    classes: List[str] = None,
    dataset_yaml: str = None,
    api_key: str = None,
    model: str = "Pro/moonshotai/Kimi-K2.5",
    train_split: float = 0.8,
    val_split: float = 0.1,
    test_split: float = 0.1
):
    """自动标注整个数据集

    Args:
        image_dir: 输入图片目录
        output_dir: 输出目录
        classes: 类别列表，如果为 None 则从 dataset_yaml 加载
        dataset_yaml: YOLO 数据集配置文件路径
        api_key: SiliconFlow API Key
        model: 使用的模型
        train_split: 训练集比例
        val_split: 验证集比例
        test_split: 测试集比例
    """
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)

    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        image_files.extend(list(image_dir.glob(ext)))
        image_files.extend(list(image_dir.glob(ext.upper())))

    if not image_files:
        raise ValueError(f"No images found in {image_dir}")

    logger.info(f"Found {len(image_files)} images")

    config = AutoAnnotatorConfig(
        api_key=api_key or os.environ.get("SILICONFLOW_API_KEY", ""),
        model=model
    )
    annotator = AutoAnnotator(config)

    train_count = int(len(image_files) * train_split)
    val_count = int(len(image_files) * val_split)

    splits = {
        'train': image_files[:train_count],
        'val': image_files[train_count:train_count + val_count],
        'test': image_files[train_count + val_count:]
    }

    all_classes = set()

    for split_name, split_images in splits.items():
        images_out = output_dir / 'images' / split_name
        labels_out = output_dir / 'labels' / split_name
        images_out.mkdir(parents=True, exist_ok=True)
        labels_out.mkdir(parents=True, exist_ok=True)

        for img_path in split_images:
            logger.info(f"[{split_name}] Processing {img_path.name}...")

            try:
                annotations, class_names = annotator.annotate_image(
                    str(img_path),
                    classes=classes,
                    dataset_yaml=dataset_yaml
                )
                all_classes.update(class_names)

                import shutil
                shutil.copy(img_path, images_out / img_path.name)

                label_path = labels_out / f"{img_path.stem}.txt"
                annotator.save_annotations(annotations, class_names, str(label_path))
                logger.info(f"  Saved {len(annotations)} annotations to {label_path}")

            except Exception as e:
                logger.error(f"Error processing {img_path.name}: {e}")
                continue

    final_classes = list(all_classes) if classes is None else classes
    dataset_yaml_path = output_dir / 'dataset.yaml'
    annotator.create_dataset_yaml(str(dataset_yaml_path), final_classes)
    logger.info(f"Dataset created at: {output_dir}")
    logger.info(f"Classes: {final_classes}")


def main():
    parser = argparse.ArgumentParser(description='YOLO 图片自动标注工具 (SiliconFlow Kimi-K2.5)')
    parser.add_argument('--images', type=str, required=True, help='图片目录或单个图片路径')
    parser.add_argument('--output', type=str, required=True, help='输出目录')
    parser.add_argument('--classes', type=str, nargs='+', help='类别列表，如: person car dog')
    parser.add_argument('--dataset', type=str, help='YOLO 数据集配置文件路径（用于自动加载类别）')
    parser.add_argument('--api_key', type=str, default=None, help='SiliconFlow API Key（默认从 .env 读取）')
    parser.add_argument('--model', type=str, default=None, help='模型名称（默认从 .env 读取）')
    parser.add_argument('--conf', type=float, default=0.25, help='置信度阈值（默认 0.25）')
    parser.add_argument('--config', type=str, help='配置文件路径')
    parser.add_argument('--single', action='store_true', help='单图片模式')
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='日志级别（默认 INFO）')

    args = parser.parse_args()

    import logging as log_module
    log_level = getattr(log_module, args.log_level.upper(), log_module.INFO)
    log_module.basicConfig(
        level=log_level,
        format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    api_key = args.api_key or os.environ.get("SILICONFLOW_API_KEY", "")
    model = args.model or os.environ.get("SILICONFLOW_MODEL", "Pro/moonshotai/Kimi-K2.5")

    if not api_key:
        raise ValueError("API key required. Set SILICONFLOW_API_KEY in .env or use --api_key")

    logger.info(f"Starting auto annotation with model: {model}, confidence threshold: {args.conf}")

    if args.config:
        config = AutoAnnotatorConfig(api_key=api_key, model=model, confidence_threshold=args.conf)
        annotator = AutoAnnotator(args.config)
    else:
        config = AutoAnnotatorConfig(api_key=api_key, model=model, confidence_threshold=args.conf)
        annotator = AutoAnnotator(config)

    if args.single:
        img_path = args.images
        logger.info(f"Annotating: {img_path}")
        annotations, class_names = annotator.annotate_image(
            img_path,
            classes=args.classes,
            dataset_yaml=args.dataset
        )
        logger.info(f"Found {len(annotations)} objects")
        logger.info(f"Classes: {class_names}")

        output_path = Path(args.output)
        annotator.save_annotations(annotations, class_names, str(output_path))
        logger.info(f"Saved to: {output_path}")
    else:
        auto_annotate_dataset(
            image_dir=args.images,
            output_dir=args.output,
            classes=args.classes,
            dataset_yaml=args.dataset,
            api_key=api_key,
            model=model
        )


if __name__ == '__main__':
    main()
