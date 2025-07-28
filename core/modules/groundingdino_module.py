import numpy as np
import torch
from typing import List
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

from core.entity.image_parse_data import BBox, ImageParseUnit, ImageParseResult
from core.modules.base import BaseModule
from core.modules.model_config import ModelLoader
from core.modules.module_factory import ModuleFactory

from groundingdino.util.inference import load_model, predict

@ModelLoader.register_loader("groundingdino")
def load_model_groundingdino(cfg, device):
    return load_model(cfg["config_path"], cfg["checkpoint_path"]).to(device)

def preprocess(image: np.ndarray, device="cuda"):
    if image.shape[-1] == 4:
        image = image[..., :3]  # 丢弃 alpha 通道
    image_pil = Image.fromarray(image)

    transform = Compose([
        Resize([800], max_size=1333),  # GroundingDINO 默认使用统一尺寸
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225]),
    ])

    image_tensor = transform(image_pil).to(device)  # [3, H, W]
    return image_tensor

def normalize_box_to_xyxy_abs(box: torch.Tensor, image_size: tuple):
    cx, cy, bw, bh = box.tolist()
    x1 = (cx - 0.5 * bw) * image_size[1]  # width
    y1 = (cy - 0.5 * bh) * image_size[0]  # height
    x2 = (cx + 0.5 * bw) * image_size[1]
    y2 = (cy + 0.5 * bh) * image_size[0]
    return [x1, y1, x2, y2]


class GroundingDinoModule(BaseModule):
    def __init__(self, model, device="cuda"):
        self.model = model
        self.device = device

    def parse(self, image: np.ndarray, prompts: List[str], box_threshold=0.3, text_threshold=0.25, **kwargs) -> ImageParseResult:
        if prompts is None or len(prompts) == 0:
            return None

        image_tensor = preprocess(image, device=self.device)

        # 统一 prompts 为字符串
        if isinstance(prompts, list):
            prompt_str = ", ".join(prompts)
        elif isinstance(prompts, str):
            prompt_str = prompts.strip()
        else:
            raise ValueError("prompts 必须是 str 或 List[str] 类型")

        # 调用 Grounding DINO 的推理函数
        boxes, logits, phrases = predict(
            model=self.model,
            image=image_tensor,
            caption=prompt_str,
            box_threshold=box_threshold,
            text_threshold=text_threshold
        )
        print(boxes, logits, phrases)
        # 构造解析结果
        result = ImageParseResult(image=image)
        for box, score, label in zip(boxes, logits, phrases):
            xyxy = normalize_box_to_xyxy_abs(box, image.shape[:2])
            bbox = BBox(*xyxy)
            result.units.append(
                ImageParseUnit(
                    image=image,
                    source_module="groundingdino",
                    score=float(score),
                    bbox=bbox,
                    type="text_target",
                    label=label,
                )
            )
        return result


@ModuleFactory.register_module("groundingdino")
def build_module_groundingdino():
    model = ModelLoader().get_model("groundingdino")
    return GroundingDinoModule(model)
