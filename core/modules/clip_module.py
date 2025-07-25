import torch
from transformers import AutoProcessor, AutoModelForZeroShotImageClassification
import numpy as np
from typing import List
from PIL import Image
from core.entity.image_parse_data import ImageParseUnit
from core.modules.base import EnricherModule
from core.modules.model_config import ModelLoader
from core.modules.module_factory import ModuleFactory


@ModelLoader.register_loader("clip")
def load_model_clip(cfg, device):
    processor = AutoProcessor.from_pretrained(cfg["processor"])
    model = AutoModelForZeroShotImageClassification.from_pretrained(cfg["model"]).to(
        device
    )
    return {
        "processor": processor,
        "model": model,
        "label_texts": cfg.get("label_texts", []),
    }


class ClipModule(EnricherModule):
    def __init__(self, model, processor, label_texts, device="cuda"):
        self.model = model
        self.processor = processor
        self.label_texts = label_texts
        self.device = device

        with torch.no_grad():
            self.label_features = self._encode_texts(self.label_texts)

    def parse(
        self,
        objects: List[ImageParseUnit],
        filter: str = "bbox",  # 可选：bbox / mask / image
        **kwargs
    ) -> List[ImageParseUnit]:
        for obj in objects:
            # --- 选择区域图像 ---
            if filter == "mask":
                image = obj.get_mask_image()
                if image is None:
                    image = obj.image
            elif filter == "image":
                image = obj.image
            else:  # 默认 bbox
                image = (
                    obj.bbox_image
                    if obj.bbox_image is not None
                    else obj.get_bbox_image()
                )

            label, score = self._classify(image)
            obj.enrich_label(source_module="clip", score=score, label=label)
        return objects

    def _classify(self, image: np.ndarray):
        image = Image.fromarray(image)
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            similarity = torch.matmul(image_features, self.label_features.T)
            best_idx = similarity.argmax().item()
            best_score = similarity[0, best_idx].item()
            return self.label_texts[best_idx], round(float(best_score), 4)

    def _encode_texts(self, texts: List[str]) -> torch.Tensor:
        text_inputs = self.processor(text=texts, return_tensors="pt", padding=True)
        text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
        text_features = self.model.get_text_features(**text_inputs)
        return text_features / text_features.norm(dim=-1, keepdim=True)


@ModuleFactory.register_module("clip")
def build_module_clip():
    model_bundle = ModelLoader().get_model("clip")
    return ClipModule(
        model=model_bundle["model"],
        processor=model_bundle["processor"],
        label_texts=model_bundle["label_texts"],
    )
