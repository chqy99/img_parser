from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
import torch
import numpy as np
from typing import List

from core.entity.image_parse_data import BBox, ImageParseUnit, ImageParseResult
from core.modules.base import BaseModule
from core.modules.model_config import ModelLoader
from core.modules.module_factory import ModuleFactory


@ModelLoader.register_loader("sam2")
def load_model_sam2(cfg, device):
    return build_sam2(cfg["model_cfg"], cfg["checkpoint"]).to(device)


class SamModule(BaseModule):
    def __init__(self, model):
        self.model = model
        self.mask_generator = SAM2AutomaticMaskGenerator(
            self.model, box_nms_thresh=0.5, crop_nms_thresh=0.5, multimask_output=False
        )
        self.predictor = SAM2ImagePredictor(self.model)

    def parse(self, image: np.ndarray, **kwargs) -> ImageParseResult:
        res = self.mask_generator.generate(image)
        parse_res = ImageParseResult(image=image)
        for item in res:
            mask = item["segmentation"]
            # bbox_input = item["bbox"], 不使用解析的 bbox
            bbox = BBox.mask_to_bbox(mask)
            score = item["stability_score"]
            parse_res.units.append(
                ImageParseUnit(
                    image=image,
                    source_module="sam2",
                    score=score,
                    bbox=bbox,
                    mask=mask,
                    type="instance",
                )
            )
        return parse_res

    def parse_with_prompts(
        self, image: np.ndarray, prompts=None, **kwargs
    ) -> ImageParseUnit:
        self.predictor.set_image(image)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            if prompts is not None:
                masks, scores, _ = self.predictor.predict(**prompts)
            else:
                masks, scores, _ = self.predictor.predict()

        max_index = np.argmax(scores)
        bbox = BBox.mask_to_bbox(masks[max_index])
        return ImageParseUnit(
            image, "sam2", scores[max_index], bbox, masks[max_index], type="instance"
        )


@ModuleFactory.register_module("sam2")
def build_module_sam2():
    model = ModelLoader().get_model("sam2")
    return SamModule(model)
