# core/pipeline/custom_omni_parser.py

import numpy as np
from typing import List
from core.pipeline.base import PipelineParser
from core.entity.image_parse_data import ImageParseResult, ImageParseUnit

from core.modules.yolo_module import YoloModule
from core.modules.paddleocr_module import PaddleOCRModule
from core.modules.florence2_module import Florence2Module


class CustomOmniParser(PipelineParser):
    def __init__(self):
        super().__init__(module_names=["yolo", "paddleocr", "florence2_icon"])

    def parse(self, image: np.ndarray, **kwargs) -> ImageParseResult:

        # 1. YOLO 检测
        yolo_module: YoloModule = self.get_module("yolo")
        yolo_result = yolo_module.parse(image, **kwargs)
        yolo_units = yolo_result.units

        # 2. OCR 检测
        ocr_module: PaddleOCRModule = self.get_module("paddleocr")
        ocr_result = ocr_module.parse(image, **kwargs)
        ocr_units = ocr_result.units

        # 3. Florence2 区域语义丰富（对所有 YOLO 区域）
        florence_module: Florence2Module = self.get_module("florence2_icon")
        florence_units = florence_module.parse(yolo_units, image_filter="bbox", **kwargs)

        # 4. 合并所有解析单元
        all_units = []
        # YOLO 区域（已被 Florence2 enrich）
        all_units.extend(florence_units)
        # OCR 区域
        all_units.extend(ocr_units)

        # 5. 构建最终结果
        result = ImageParseResult(image=image, units=all_units)
        return result
