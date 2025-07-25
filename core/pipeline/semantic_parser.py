# core/pipeline/semantic_parser.py

import numpy as np
from typing import List
from core.pipeline.base import PipelineParser
from core.entity.image_parse_data import ImageParseResult
from core.imgtools.statistics_utils import StatisticsUtils


from core.modules.sam2_module import SamModule
from core.modules.paddleocr_module import PaddleOCRModule
from core.modules.florence2_module import Florence2Module
from core.modules.yolo_module import YoloModule


class SemanticParser(PipelineParser):
    def __init__(self):
        super().__init__(module_names=["sam2", "paddleocr", "florence2"])

    def parse(self, image: np.ndarray, **kwargs) -> ImageParseResult:
        # 1. SAM2 实例分割
        sam2_module: SamModule = self.get_module("sam2")
        sam2_result = sam2_module.parse(image, **kwargs)
        sam2_units = sam2_result.units

        # 2. OCR 检测
        ocr_module: PaddleOCRModule = self.get_module("paddleocr")
        ocr_result = ocr_module.parse(image, **kwargs)
        ocr_units = ocr_result.units

        # 3. Florence2 区域语义丰富（对 SAM2 区域）
        florence_module: Florence2Module = self.get_module("florence2")
        florence_units = florence_module.parse(sam2_units, filter="mask", **kwargs)

        # 4. 合并所有解析单元
        all_units = []
        all_units.extend(florence_units)  # 语义丰富后的 SAM2 区域
        all_units.extend(ocr_units)       # OCR 区域

        # 5. 构建最终结果
        result = ImageParseResult(image=image, units=all_units)
        return result
