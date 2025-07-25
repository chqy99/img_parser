import base64
import numpy as np
from typing import Optional, Dict, Any
from core.entity.image_parse_data import ImageParseResult, ImageParseUnit, base64_to_np, np_to_base64
from core.modules.module_factory import ModuleFactory
from core.pipeline.semantic_parser import SemanticParser
from core.pipeline.custom_omni_parser import CustomOmniParser

class ParserManager:
    def __init__(self):
        # 可扩展 pipeline 注册
        self.pipelines = {
            "semantic": SemanticParser(),
            "omni": CustomOmniParser(),
        }
        # 支持的单模型
        self.modules = ["yolo", "paddleocr", "clip", "sam2"]

    def upload_image(self, image_base64: str) -> str:
        """
        上传图片，返回 uid。
        """
        image = base64_to_np(image_base64)
        # 生成唯一 uid
        from core.entity.image_parse_data import IDGenerator
        uid = IDGenerator.instance().next_id("img")
        # 可扩展：保存图片到本地或数据库
        # ...
        return uid

    def parse_image(self, image_base64: str, mode: str = "semantic", prompt: Optional[str] = None) -> dict:
        """
        支持多种解析模式：pipeline 或单模型。
        mode: "semantic" | "omni" | "yolo" | "paddleocr" | "clip" | "sam2"
        prompt: 可选，部分模型支持
        """
        image = base64_to_np(image_base64)
        # pipeline 解析
        if mode in self.pipelines:
            parser = self.pipelines[mode]
            result: ImageParseResult = parser.parse(image, prompt=prompt)
            return result.to_dict(image_filter=["image", "bboxs_image", "masks", "masks_image"], unit_image_filter=["bbox_image", "mask_image", "mask"])
        # 单模型解析
        elif mode in self.modules:
            module = ModuleFactory.get_module(mode)
            if prompt is not None:
                result = module.parse(image, prompt=prompt)
            else:
                result = module.parse(image)
            # 兼容返回 ImageParseResult 或 ImageParseUnit
            if isinstance(result, ImageParseResult):
                return result.to_dict(image_filter=["image", "bboxs_image", "masks", "masks_image"], unit_image_filter=["bbox_image", "mask_image", "mask"])
            elif isinstance(result, ImageParseUnit):
                return result.to_dict(image_filter=["bbox_image", "mask_image", "mask", "image"])
            else:
                return {"error": "Unknown result type"}
        else:
            return {"error": f"不支持的解析模式: {mode}"}

    def parse_with_prompts(self, image_base64: str, prompts: Dict[str, Any]) -> dict:
        """
        SAM2 单点/多点掩码预测，prompts 结构见 sam2_module。
        """
        image = base64_to_np(image_base64)
        sam2_module = ModuleFactory.get_module("sam2")
        unit: ImageParseUnit = sam2_module.parse_with_prompts(image, prompts=prompts)
        return unit.to_dict(image_filter=["bbox_image", "mask_image", "mask", "image"])

