import base64
import numpy as np
from typing import Dict, Any, List
from core.entity.image_parse_data import ImageParseResult, ImageParseUnit
import core.modules
from core.modules.module_factory import ModuleFactory
from core.pipeline.semantic_parser import SemanticParser
from core.pipeline.custom_omni_parser import CustomOmniParser
from core.services.image_cache_service import image_cache_service

class ParserManager:
    def __init__(self):
        # 可扩展 pipeline 注册
        self.pipelines = {
            "semantic": SemanticParser(),
            "omni": CustomOmniParser(),
        }
        # 支持的单模型
        self.modules = ["yolo", "paddleocr", "clip", "sam2", "groundingdino"]

    def parse_image(self, uid: str, mode: str = "semantic", prompts: List[str] = None) -> dict:
        """
        支持多种解析模式：pipeline 或单模型。
        mode: "semantic" | "omni" | "yolo" | "paddleocr" | "clip" | "sam2"
        prompt: 可选，部分模型支持
        """
        image = image_cache_service.get_image_by_uid(uid)
        # pipeline 解析
        if mode in self.pipelines:
            parser = self.pipelines[mode]
            result: ImageParseResult = parser.parse(image, prompts=prompts)
            return result.to_dict(image_filter=[], unit_image_filter=["bbox_image", "mask_image", "mask"])
        # 单模型解析
        elif mode in self.modules:
            module = ModuleFactory.get_module(mode)
            result = module.parse(image, prompts=prompts)
            # 兼容返回 ImageParseResult 或 ImageParseUnit
            if isinstance(result, ImageParseResult):
                return result.to_dict(image_filter=[], unit_image_filter=["bbox_image", "mask_image", "mask"])
            elif isinstance(result, ImageParseUnit):
                return result.to_dict(image_filter=["bbox_image", "mask_image", "mask"])
            else:
                return {"error": "Unknown result type"}
        else:
            return {"error": f"不支持的解析模式: {mode}"}

    def parse_image_keyinfo(self, uid: str, mode: str = "semantic", prompts: List[str] = None) -> List[dict]:
        """
        仅返回 key info: [ {"bbox": [...], "text": "...", "label": "..."}, ... ]
        """
        image = image_cache_service.get_image_by_uid(uid)
        if image is None:
            return []

        def _extract(unit) -> dict:
            # 适配 ImageParseUnit 或 dict
            return {
                "bbox": unit.bbox,
                "text": unit.text,
                "label": unit.label
            }

        # pipeline
        if mode in self.pipelines:
            parser = self.pipelines[mode]
            result = parser.parse(image, prompts=prompts)
            return [_extract(u) for u in result.units]

        # 单模型
        if mode in self.modules:
            module = ModuleFactory.get_module(mode)
            result = module.parse(image, prompts=prompts)
            if isinstance(result, ImageParseResult):
                return [_extract(u) for u in result.units]
            elif isinstance(result, ImageParseUnit):
                return [_extract(result)]
        return []

    def sam2_predict_with_prompts(self, uid: str, prompts: Dict[str, Any]) -> dict:
        """
        SAM2 单点/多点掩码预测，prompts 结构见 sam2_module。
        """
        image = image_cache_service.get_image_by_uid(uid)
        sam2_module = ModuleFactory.get_module("sam2")
        unit: ImageParseUnit = sam2_module.parse_with_prompts(image, prompts=prompts)
        return unit.to_dict(image_filter=["bbox_image", "mask_image", "mask"])
