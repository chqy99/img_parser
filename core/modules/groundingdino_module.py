import numpy as np
import torch
from typing import List
from PIL import Image

from core.entity.image_parse_data import BBox, ImageParseUnit, ImageParseResult
from core.modules.base import BaseModule
from core.modules.model_config import ModelLoader
from core.modules.module_factory import ModuleFactory

from groundingdino.util.inference import load_model, predict

@ModelLoader.register_loader("groundingdino")
def load_model_groundingdino(cfg, device):
    return load_model(cfg["config_path"], cfg["checkpoint_path"]).to(device)

class GroundingDinoModule(BaseModule):
    def __init__(self, model, device="cuda"):
        self.model = model
        self.device = device

    def parse(self, image: np.ndarray, prompts: List[str], box_threshold=0.3, text_threshold=0.25, **kwargs) -> ImageParseResult:
        if prompts is None or len(prompts) == 0:
            return None

        # 将 numpy image 转换为 PIL.Image
        pil_image = Image.fromarray(image)

        # 构造逗号分隔的 prompt 字符串
        prompt_str = ", ".join(prompts)

        # 调用 Grounding DINO 的推理函数
        boxes, logits, phrases = predict(
            model=self.model,
            image=pil_image,
            caption=prompt_str,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=self.device,
        )

        # 构造解析结果
        result = ImageParseResult(image=image)
        for box, score, label in zip(boxes, logits, phrases):
            xyxy = box.cpu().numpy().tolist()
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
