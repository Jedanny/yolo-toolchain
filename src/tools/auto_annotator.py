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
    import cv2
except ImportError:
    cv2 = None

try:
    import numpy as np
except ImportError:
    np = None

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
    max_tokens: int = 2048
    temperature: float = 0.1
    image_size: int = 768
    confidence_threshold: float = 0.25
    max_retries: int = 3
    retry_delay: float = 2.0
    timeout: int = 300
    batch_size: int = 1


class SiliconFlowClient:
    """SiliconFlow API 客户端"""

    def __init__(self, api_key: str, base_url: str = "https://api.siliconflow.cn/v1", timeout: int = 300):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.endpoint = f"{self.base_url}/chat/completions"
        self.timeout = timeout

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

        response = requests.post(self.endpoint, headers=headers, json=payload, timeout=self.timeout)
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


def encode_image_to_data_url(image_path: str, max_size: int = 1024) -> Tuple[str, int, int]:
    """将图片转换为 data URL 格式，返回 (data_url, actual_width, actual_height)

    Args:
        image_path: 图片路径
        max_size: 最大尺寸（默认 1024）

    Returns:
        (data_url, actual_width, actual_height): 数据URL和实际发送的图片尺寸
    """
    img = Image.open(image_path)

    orig_width, orig_height = img.size
    actual_width, actual_height = orig_width, orig_height

    if max(orig_width, orig_height) > max_size:
        scale = max_size / max(orig_width, orig_height)
        actual_width = int(orig_width * scale)
        actual_height = int(orig_height * scale)
        img = img.resize((actual_width, actual_height), Image.LANCZOS)

    buffer = io.BytesIO()
    img.convert("RGB").save(buffer, format="JPEG", quality=75)
    img_bytes = buffer.getvalue()
    b64_str = base64.b64encode(img_bytes).decode("utf-8")

    return f"data:image/jpeg;base64,{b64_str}", actual_width, actual_height


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


def load_class_descriptions_from_yaml(yaml_path: str) -> Dict[str, str]:
    """从 YOLO 数据集配置文件中加载类别详细描述

    支持格式：
    class_descriptions:
      0: 描述文字
      1: 描述文字
    或者
    class_descriptions:
      "类名": 描述文字
    """
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)

    desc_dict = {}
    class_descriptions = config.get('class_descriptions', {})

    if not class_descriptions:
        return desc_dict

    names = config.get('names', {})

    # 支持两种格式：数字索引 或 类名
    if isinstance(class_descriptions, dict):
        for key, desc in class_descriptions.items():
            if isinstance(key, int):
                # 数字索引，转换为类名
                if isinstance(names, dict) and key in names:
                    desc_dict[names[key]] = desc
            else:
                # 类名本身就是 key
                desc_dict[key] = desc
    elif isinstance(class_descriptions, list):
        # 列表格式，顺序对应 names
        for i, desc in enumerate(class_descriptions):
            if isinstance(names, dict) and i in names:
                desc_dict[names[i]] = desc
            elif isinstance(names, list) and i < len(names):
                desc_dict[names[i]] = desc

    return desc_dict


def build_annotation_prompt(
    classes: List[str] = None,
    class_descriptions: Dict[str, str] = None,
    custom_note: str = None
) -> str:
    """根据类别构建标注提示词

    Args:
        classes: 类别名称列表
        class_descriptions: 类别详细描述字典 {"类名": "描述", ...}
        custom_note: 额外说明
    """
    if classes:
        if class_descriptions:
            # 构建带详细描述的类别列表
            classes_detail = []
            for cls in classes:
                if cls in class_descriptions:
                    classes_detail.append(f"- {cls}：{class_descriptions[cls]}")
                else:
                    classes_detail.append(f"- {cls}")
            classes_str = '\n'.join(classes_detail)
            base_prompt = f"""你是一个精确的目标检测标注专家。请仔细分析图片，准确地标注目标。

类别及详细说明：
{classes_str}

【重要-坐标格式】
坐标必须归一化到 0-1 范围，使用 [center_x, center_y, width, height] 格式：
- center_x, center_y 是目标中心点（相对于图片宽高的比例）
- width, height 是目标宽高（相对于图片宽高的比例）
- 正确：bbox=[0.5, 0.5, 0.2, 0.3] 表示中心在50%位置，宽高为20%和30%
- 错误：bbox=[480, 360, 100, 150] 这是像素值，必须归一化！

【重要-检测原则】
- 只标注列表中明确列出的类别，不要推测或假设
- 严格区分相似物体：例如"黄色工装"只标注穿黄色/橙色工装的工人，不标注穿其他颜色衣服的人
- 目标框要紧凑，紧密包围目标边缘，不要留太多空白
- 漏标比误标更可取：不确定的目标不要标注
- 互相遮挡的目标只标注清晰可见的部分

【重要-置信度说明】
- confidence 表示你的确信程度（0-1）
- ≥0.9：非常有把握，边界清晰
- 0.7-0.9：有把握，但边界可能有轻微模糊
- 0.5-0.7：较有把握，存在一定不确定性
- 0.25-0.5：不确定，只标注明确属于该类别的目标
- <0.25：不标注

JSON格式：
{{"width":W,"height":H,"classes":["类1","类2"],"annotations":[{{"class_id":0,"bbox":[cx,cy,w,h],"confidence":0.9}}]}}

直接输出JSON："""
        else:
            classes_str = ', '.join(classes)
            base_prompt = f"""你是一个精确的目标检测标注专家。请仔细分析图片，准确地标注目标。

类别列表：{classes_str}

【重要-坐标格式】
坐标必须归一化到 0-1 范围，使用 [center_x, center_y, width, height] 格式：
- 正确：bbox=[0.5, 0.5, 0.2, 0.3]
- 错误：bbox=[480, 360, 100, 150] 必须归一化！

【重要-检测原则】
- 只标注列表中明确列出的类别
- 严格区分相似物体
- 目标框要紧凑
- 漏标比误标更可取

【重要-置信度】
- ≥0.9：非常有把握
- 0.7-0.9：有把握
- 0.5-0.7：较有把握
- 0.25-0.5：不确定
- <0.25：不标注

JSON格式：
{{"width":W,"height":H,"classes":["类1","类2"],"annotations":[{{"class_id":0,"bbox":[cx,cy,w,h],"confidence":0.9}}]}}

直接输出JSON："""
    else:
        base_prompt = """你是一个精确的目标检测标注专家。

【重要-坐标格式】
坐标必须归一化到 0-1 范围：[center_x, center_y, width, height]
- 正确：bbox=[0.5, 0.5, 0.2, 0.3]
- 错误：bbox=[480, 360, 100, 150] 必须归一化！

【检测原则】
- 只标注明确属于类别的目标
- 漏标比误标更可取
- 目标框要紧凑

【置信度】
- ≥0.9：非常有把握
- 0.7-0.9：有把握
- 0.5-0.7：较有把握
- 0.25-0.5：不确定
- <0.25：不标注

JSON格式：
{{"width":W,"height":H,"classes":["类别"],"annotations":[{{"class_id":0,"bbox":[cx,cy,w,h],"confidence":0.9}}]}}

直接输出JSON："""

    if custom_note:
        base_prompt += f"\n额外要求：{custom_note}"

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

        # 解析配置优先级: 参数 > .env > 兜底默认值
        api_key = self.config.api_key or os.environ.get("SILICONFLOW_API_KEY", "")
        if not api_key:
            raise ValueError("API key is required. Set SILICONFLOW_API_KEY environment variable or provide in config.")

        # 模型优先级: config.model > SILICONFLOW_MODEL > "Pro/moonshotai/Kimi-K2.5"
        if not self.config.model or self.config.model == "Pro/moonshotai/Kimi-K2.5":
            self.config.model = os.environ.get("SILICONFLOW_MODEL", "Pro/moonshotai/Kimi-K2.5")

        self.client = SiliconFlowClient(api_key, self.config.base_url, self.config.timeout)
        self.class_names = []

    def annotate_image(
        self,
        image_path: str,
        classes: List[str] = None,
        dataset_yaml: str = None,
        class_descriptions: Dict[str, str] = None
    ) -> Tuple[List, List]:
        """标注单张图片

        Args:
            image_path: 图片路径
            classes: 类别列表（优先于 dataset_yaml）
            dataset_yaml: YOLO 数据集配置文件路径
            class_descriptions: 类别详细描述 {"类名": "描述", ...}

        Returns:
            (annotations, class_names): 标注列表和类别名称列表
        """
        img = Image.open(image_path)
        width, height = img.size

        if classes is None and dataset_yaml:
            classes = load_classes_from_yaml(dataset_yaml)

        # 自动从 dataset_yaml 加载 class_descriptions（除非已显式传入）
        if class_descriptions is None and dataset_yaml:
            class_descriptions = load_class_descriptions_from_yaml(dataset_yaml)

        prompt = build_annotation_prompt(classes=classes, class_descriptions=class_descriptions)

        # 获取实际 resize 后的图片尺寸（用于坐标映射）
        image_data, actual_width, actual_height = encode_image_to_data_url(image_path, self.config.image_size)

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
            api_start = time.time()
            try:
                response = self.client.chat(
                    messages=messages,
                    model=self.config.model,
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature
                )
                api_time = time.time() - api_start

                content = response["choices"][0]["message"]["content"]
                logger.debug(f"Raw response: {content[:500]}...")
                result = self._parse_json_response(content)

                if result and "annotations" in result:
                    # IMPORTANT: Use OUR class order, not API's returned order
                    # API might return classes in different order than what we sent
                    self.class_names = classes if classes else result.get("classes", [])
                    annotations = self._filter_by_confidence(result["annotations"])
                    # Store ACTUAL resized image dimensions (not API-reported) for coordinate scaling
                    # This is critical because the API might misreport dimensions
                    for ann in annotations:
                        ann['_ann_width'] = actual_width
                        ann['_ann_height'] = actual_height
                        # 将 class_id 转换为 class 名称（使用我们的 class_names 顺序）
                        class_id = ann.get('class_id', 0)
                        if 0 <= class_id < len(self.class_names):
                            ann['class'] = self.class_names[class_id]
                    logger.info(f"API: {api_time:.1f}s, annotations: {len(annotations)}, img_size: {actual_width}x{actual_height}")
                    return annotations, self.class_names

                retry_count += 1
                if result:
                    if isinstance(result, dict):
                        last_error = f"Missing 'annotations' key. Got keys: {list(result.keys())}"
                    else:
                        last_error = f"Expected dict but got {type(result).__name__}: {str(result)[:200]}"
                    logger.warning(f"Invalid response structure ({retry_count}/{self.config.max_retries}): {last_error}")
                else:
                    last_error = "Invalid response format"
                    logger.warning(f"Failed to parse JSON ({retry_count}/{self.config.max_retries})")

            except requests.exceptions.RequestException as e:
                retry_count += 1
                last_error = str(e)
                logger.warning(f"API call failed ({retry_count}/{self.config.max_retries}): {e}")
                time.sleep(self.config.retry_delay)
            except (KeyError, IndexError, json.JSONDecodeError) as e:
                retry_count += 1
                last_error = f"Parse error: {e}"
                logger.warning(f"Parse failed ({retry_count}/{self.config.max_retries}): {e}")
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
            try:
                annotations, class_names = self.annotate_image(image_path, classes)
                results[image_path] = (annotations, class_names)
            except Exception as e:
                logger.error(f"[{i+1}/{len(image_paths)}] Failed: {Path(image_path).name} - {e}")
                results[image_path] = ([], [])

        return results

    def save_annotations(self, annotations: List, class_names: List[str], output_path: str):
        """保存标注为 YOLO 格式

        Args:
            annotations: 标注列表，每个 annotation 包含 'class'（类别名）或 'class_id'
            class_names: 类别名称列表，用于将类别名转换为 class_id
            output_path: 输出文件路径
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 构建类别名到 id 的映射（更高效的查找）
        class_name_to_id = {name: i for i, name in enumerate(class_names)}

        with open(output_path, 'w') as f:
            for ann in annotations:
                # 优先使用 'class' 名称，否则使用 'class_id'
                class_name = ann.get('class')
                if class_name is not None and class_name in class_name_to_id:
                    class_id = class_name_to_id[class_name]
                else:
                    class_id = ann.get('class_id', 0)

                # 支持多种 bbox 格式: 'bbox', 'bboxes', 'coordinates'
                bbox = ann.get('bbox') or ann.get('bboxes') or ann.get('coordinates', [])

                if len(bbox) >= 4:
                    cx, cy, w, h = bbox

                    # API 返回归一化坐标 [cx, cy, w, h] (0-1)，直接保存
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

    def save_predefined_classes(self, output_path: str, class_names: List[str]):
        """保存 labelImg 所需类别文件"""
        output_path = Path(output_path)
        with open(output_path, 'w') as f:
            for name in class_names:
                f.write(f"{name}\n")

    def draw_annotations_on_image(
        self,
        image_path: str,
        annotations: List,
        class_names: List[str],
        output_path: str = None,
        ann_width: int = None,
        ann_height: int = None
    ) -> str:
        """在图片上绘制标注并保存

        Args:
            image_path: 原图路径
            annotations: 标注列表
            class_names: 类别名称列表
            output_path: 输出路径，默认在原图目录创建 vis/ 子目录
            ann_width: API 返回的图像宽度（如果不同于原图）
            ann_height: API 返回的图像高度（如果不同于原图）

        Returns:
            保存的图片路径
        """
        if cv2 is None:
            logger.warning("opencv-python not installed, skipping visualization")
            return None

        img = cv2.imread(str(image_path))
        if img is None:
            logger.warning(f"Failed to read image: {image_path}")
            return None

        h, w = img.shape[:2]

        if not annotations:
            logger.warning(f"No annotations to draw!")
            return None

        # 如果 annotation 包含图像尺寸信息，使用它来计算缩放比例
        # API 处理的是缩放后的图片，需要将 bbox 坐标映射回原图
        scale_x = w / ann_width if ann_width else 1.0
        scale_y = h / ann_height if ann_height else 1.0

        # 颜色调色板
        colors = [
            (255, 0, 0),      # 红色
            (0, 255, 0),      # 绿色
            (0, 0, 255),      # 蓝色
            (255, 255, 0),    # 黄色
            (255, 0, 255),    # 紫色
            (0, 255, 255),    # 青色
            (255, 128, 0),    # 橙色
            (128, 0, 255),    # 紫红
        ]

        for ann in annotations:
            # 支持 class 名称或 class_id
            class_name = ann.get('class')
            if class_name is not None and class_name in class_names:
                class_id = class_names.index(class_name)
            else:
                class_id = ann.get('class_id', 0)

            # 支持多种 bbox 格式: 'bbox', 'bboxes', 'coordinates'
            bbox = ann.get('bbox')
            if bbox is None:
                bbox = ann.get('bboxes')
            if bbox is None:
                bbox = ann.get('coordinates', [])
            confidence = ann.get('confidence', 1.0)

            if not isinstance(bbox, list) or len(bbox) < 4:
                logger.warning(f"Skipping ann with invalid bbox: {bbox}")
                continue

            cx, cy, bw, bh = bbox
            color = colors[class_id % len(colors)]

            # API 返回归一化坐标 [cx, cy, w, h] (0-1)
            # 乘以 ann_size 再映射到原图
            if ann_width and ann_height:
                cx_px = int(cx * ann_width * scale_x)
                cy_px = int(cy * ann_height * scale_y)
                bw_px = int(bw * ann_width * scale_x)
                bh_px = int(bh * ann_height * scale_y)
            else:
                cx_px = int(cx * w)
                cy_px = int(cy * h)
                bw_px = int(bw * w)
                bh_px = int(bh * h)

            x1 = cx_px - bw_px // 2
            y1 = cy_px - bh_px // 2
            x2 = cx_px + bw_px // 2
            y2 = cy_px + bh_px // 2

            # 绘制边界框
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            # 绘制标签
            display_name = class_name if class_name else (class_names[class_id] if class_id < len(class_names) else f"class_{class_id}")
            label = f"{display_name}:{confidence:.2f}"

            # 标签背景
            (label_w, label_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(
                img,
                (x1, y1 - label_h - baseline - 5),
                (x1 + label_w, y1),
                color,
                -1
            )
            # 标签文字
            cv2.putText(
                img,
                label,
                (x1, y1 - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )

        # 确定输出路径
        if output_path is None:
            vis_dir = Path(image_path).parent / "vis"
            vis_dir.mkdir(parents=True, exist_ok=True)
            output_path = vis_dir / Path(image_path).name

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 保存图片
        cv2.imwrite(str(output_path), img)
        logger.debug(f"Saved visualization: {output_path}")

        return str(output_path)


def auto_annotate_dataset(
    image_dir: str,
    output_dir: str,
    classes: List[str] = None,
    dataset_yaml: str = None,
    api_key: str = None,
    model: str = None,
    train_split: float = 0.7,
    val_split: float = 0.2,
    test_split: float = 0.1,
    seed: int = 42,
    workers: int = 10,
    save_vis: bool = True
):
    """自动标注整个数据集

    Args:
        image_dir: 输入图片目录
        output_dir: 输出目录
        classes: 类别列表，如果为 None 则从 dataset_yaml 加载
        dataset_yaml: YOLO 数据集配置文件路径
        api_key: SiliconFlow API Key
        model: 使用的模型（优先级: 参数 > SILICONFLOW_MODEL > Pro/moonshotai/Kimi-K2.5）
        train_split: 训练集比例（默认 0.7）
        val_split: 验证集比例（默认 0.2）
        test_split: 测试集比例（默认 0.1）
        seed: 随机种子（默认 42）
        workers: 并发线程数（默认 10）
        save_vis: 是否保存标注可视化图片（默认 True）
    """
    # 模型优先级: 参数 > SILICONFLOW_MODEL > 兜底值
    resolved_model = model or os.environ.get("SILICONFLOW_MODEL", "Pro/moonshotai/Kimi-K2.5")

    import time
    import random
    from concurrent.futures import ThreadPoolExecutor, as_completed

    image_dir = Path(image_dir)
    output_dir = Path(output_dir)

    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        image_files.extend(list(image_dir.glob(ext)))
        image_files.extend(list(image_dir.glob(ext.upper())))

    if not image_files:
        raise ValueError(f"No images found in {image_dir}")

    random.seed(seed)
    random.shuffle(image_files)
    logger.info(f"Found {len(image_files)} images, seed={seed}, using {workers} workers")
    start_time = time.time()

    config = AutoAnnotatorConfig(
        api_key=api_key or os.environ.get("SILICONFLOW_API_KEY", ""),
        model=resolved_model
    )

    train_count = int(len(image_files) * train_split)
    val_count = int(len(image_files) * val_split)

    splits = {
        'train': image_files[:train_count],
        'val': image_files[train_count:train_count + val_count],
        'test': image_files[train_count + val_count:]
    }

    # 优先使用传入的 classes 参数（保持顺序）
    # 如果未提供，从 dataset_yaml 加载（保持顺序）
    # 只有两者都没有时才从 API 返回结果收集
    final_classes = None
    if classes is not None:
        final_classes = list(classes)
    elif dataset_yaml is not None:
        final_classes = load_classes_from_yaml(dataset_yaml)

    all_classes = set()  # 仅用于收集（不保证顺序），最终结果用 final_classes
    total = len(image_files)
    success_count = 0
    fail_count = 0
    lock_count = __import__('threading').Lock()

    for split_name, split_images in splits.items():
        images_out = output_dir / 'images' / split_name
        labels_out = output_dir / 'labels' / split_name
        images_out.mkdir(parents=True, exist_ok=True)
        labels_out.mkdir(parents=True, exist_ok=True)

    def process_image(img_path, split_name):
        nonlocal success_count, fail_count, all_classes
        annotator = AutoAnnotator(AutoAnnotatorConfig(
            api_key=api_key or os.environ.get("SILICONFLOW_API_KEY", ""),
            model=resolved_model
        ))
        img_start = time.time()
        try:
            annotations, class_names = annotator.annotate_image(
                str(img_path),
                classes=classes,
                dataset_yaml=dataset_yaml
            )
            with lock_count:
                all_classes.update(class_names)

            import shutil
            images_out = output_dir / 'images' / split_name
            labels_out = output_dir / 'labels' / split_name
            shutil.copy(img_path, images_out / img_path.name)

            label_path = labels_out / f"{img_path.stem}.txt"
            annotator.save_annotations(annotations, class_names, str(label_path))

            # 保存可视化图片
            if save_vis:
                vis_images_out = output_dir / 'vis' / split_name
                vis_output_path = vis_images_out / img_path.name
                # Extract API image dimensions from first annotation (if available)
                ann_width = annotations[0].get('_ann_width') if annotations else None
                ann_height = annotations[0].get('_ann_height') if annotations else None
                annotator.draw_annotations_on_image(
                    str(img_path),
                    annotations,
                    class_names,
                    str(vis_output_path),
                    ann_width=ann_width,
                    ann_height=ann_height
                )

            img_time = time.time() - img_start
            with lock_count:
                success_count += 1
                if success_count % 10 == 0 or success_count == total:
                    logger.info(f"Progress: {success_count}/{total} ({success_count} ok, {fail_count} failed)")
            return True, img_path, img_time
        except Exception as e:
            img_time = time.time() - img_start
            with lock_count:
                fail_count += 1
            logger.warning(f"Failed: {img_path.name} ({img_time:.1f}s) - {e}")
            return False, img_path, img_time

    for split_name, split_images in splits.items():
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(process_image, img_path, split_name): img_path for img_path in split_images}
            for future in as_completed(futures):
                pass

    total_time = time.time() - start_time
    avg_time = total_time / total if total > 0 else 0

    # final_classes 已在前面根据 classes 或 dataset_yaml 设置好
    # 如果两者都没有，则从收集的结果中取（顺序不确定）
    if final_classes is None:
        final_classes = list(all_classes)
    annotator = AutoAnnotator(AutoAnnotatorConfig(
        api_key=api_key or os.environ.get("SILICONFLOW_API_KEY", ""),
        model=resolved_model
    ))
    dataset_yaml_path = output_dir / 'dataset.yaml'
    annotator.create_dataset_yaml(str(dataset_yaml_path), final_classes)
    labels_classes_path = output_dir / 'labels' / 'train' / 'classes.txt'
    labels_classes_path.parent.mkdir(parents=True, exist_ok=True)
    annotator.save_predefined_classes(str(labels_classes_path), final_classes)
    logger.info(f"classes.txt saved to {labels_classes_path}")
    logger.info(f"Done! {success_count}/{total} annotated, total: {total_time:.1f}s, avg: {avg_time:.1f}s/img, output: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='YOLO 图片自动标注工具 (SiliconFlow Kimi-K2.5)')
    parser.add_argument('--images', type=str, required=True, help='图片目录或单个图片路径')
    parser.add_argument('--output', type=str, required=True, help='输出目录')
    parser.add_argument('--classes', type=str, nargs='+', help='类别列表，如: person car dog')
    parser.add_argument('--class-desc', type=str, nargs='+', action='append',
                       help='类别详细描述，格式: 类名:描述，如: --class-desc 整洁工地:地面无垃圾 --class-desc 黄色工装:穿黄色安全服')
    parser.add_argument('--dataset', type=str, help='YOLO 数据集配置文件路径（用于自动加载类别）')
    parser.add_argument('--api_key', type=str, default=None, help='SiliconFlow API Key（默认从 .env 读取）')
    parser.add_argument('--model', type=str, default=None, help='模型名称（默认从 .env 读取）')
    parser.add_argument('--conf', type=float, default=0.25, help='置信度阈值（默认 0.25）')
    parser.add_argument('--timeout', type=int, default=300, help='API 超时秒数（默认 300）')
    parser.add_argument('--config', type=str, help='配置文件路径')
    parser.add_argument('--single', action='store_true', help='单图片模式')
    parser.add_argument('--workers', type=int, default=10, help='并发线程数（默认 10）')
    parser.add_argument('--seed', type=int, default=42, help='随机种子（默认 42）')
    parser.add_argument('--save-vis', action='store_true', default=True, help='保存标注可视化图片（默认开启）')
    parser.add_argument('--no-save-vis', action='store_true', help='不保存标注可视化图片')
    parser.add_argument('--labelimg', action='store_true', help='标注完成后用 labelImg 预览')
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

    logger.info(f"Starting auto annotation with model: {model}, confidence: {args.conf}, timeout: {args.timeout}s")

    if args.config:
        config = AutoAnnotatorConfig(api_key=api_key, model=model, confidence_threshold=args.conf, timeout=args.timeout)
        annotator = AutoAnnotator(args.config)
    else:
        config = AutoAnnotatorConfig(api_key=api_key, model=model, confidence_threshold=args.conf, timeout=args.timeout)
        annotator = AutoAnnotator(config)

    # Parse class descriptions
    class_descriptions = {}
    if args.class_desc:
        for item in args.class_desc:
            for desc in item:
                if ':' in desc:
                    cls_name, cls_desc = desc.split(':', 1)
                    class_descriptions[cls_name.strip()] = cls_desc.strip()
        logger.info(f"Class descriptions: {class_descriptions}")

    if args.single:
        import time
        img_path = args.images
        start_time = time.time()
        logger.info(f"Annotating: {img_path}")
        annotations, class_names = annotator.annotate_image(
            img_path,
            classes=args.classes,
            dataset_yaml=args.dataset,
            class_descriptions=class_descriptions if class_descriptions else None
        )
        elapsed = time.time() - start_time
        logger.info(f"Done! {len(annotations)} objects, {elapsed:.1f}s total")

        output_path = Path(args.output)
        annotator.save_annotations(annotations, class_names, str(output_path))

        # 保存可视化图片
        if not args.no_save_vis:
            vis_path = output_path.with_suffix('.jpg')
            ann_width = annotations[0].get('_ann_width') if annotations else None
            ann_height = annotations[0].get('_ann_height') if annotations else None
            annotator.draw_annotations_on_image(
                img_path,
                annotations,
                class_names,
                str(vis_path),
                ann_width=ann_width,
                ann_height=ann_height
            )
            logger.info(f"Visualization saved: {vis_path}")
    else:
        # 批量模式：--no-save-vis 禁用可视化
        save_vis = not args.no_save_vis
        if save_vis:
            logger.info("Will save visualization images to: {output_dir}/vis/")

        auto_annotate_dataset(
            image_dir=args.images,
            output_dir=args.output,
            classes=args.classes,
            dataset_yaml=args.dataset,
            api_key=api_key,
            model=model,
            workers=args.workers,
            seed=args.seed,
            save_vis=save_vis
        )

        if args.labelimg:
            output_path = Path(args.output)
            images_dir = output_path / 'images' / 'train'
            labels_dir = output_path / 'labels' / 'train'
            classes_file = labels_dir / 'classes.txt'

            if not images_dir.exists():
                logger.warning(f"images/train dir not found: {images_dir}")
            elif not classes_file.exists():
                logger.warning(f"classes.txt not found: {classes_file}")
            else:
                import subprocess
                logger.info(f"Launching labelImg for preview...")
                logger.info(f"Images: {images_dir}")
                logger.info(f"Labels: {labels_dir}")
                try:
                    subprocess.Popen([
                        'labelImg',
                        str(images_dir),
                        str(classes_file)
                    ])
                    logger.info("labelImg launched. Close it to continue.")
                except FileNotFoundError:
                    logger.error("labelImg not found. Install with: pip install labelImg")
                except Exception as e:
                    logger.error(f"Failed to launch labelImg: {e}")


if __name__ == '__main__':
    main()
