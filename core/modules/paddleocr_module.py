from paddleocr import PaddleOCR
import numpy as np
from typing import List
from core.entity.image_parse_data import BBox, ImageParseUnit, ImageParseResult
from core.modules.base import BaseModule
from core.modules.model_config import ModelLoader
from core.modules.module_factory import ModuleFactory


@ModelLoader.register_loader("paddleocr")
def load_model_paddleocr(cfg, device):
    # PaddleOCR 的 device 参数要传字符串 "gpu" 或 "cpu"
    device = "gpu" if device == "cuda" else device
    model = PaddleOCR(paddlex_config=cfg.get("paddlex_config", None), device="gpu")
    import paddle

    print("是否支持 GPU：", paddle.device.is_compiled_with_cuda())
    print("推理时使用设备:", paddle.device.get_device())
    return model


class PaddleOCRModule(BaseModule):
    def __init__(self, model):
        self.reader = model

    def parse(self, image: np.ndarray, **kwargs) -> ImageParseResult:
        result = self.reader.predict(image)
        res = ImageParseResult(image)
        box, text, score = (
            result[0]["rec_polys"],
            result[0]["rec_texts"],
            result[0]["rec_scores"],
        )

        score_threshold = kwargs.get("score_threshold", 0.8)  # 用户可指定分数阈值

        for i in range(len(box)):
            poly = box[i]
            x_coords = poly[:, 0]
            y_coords = poly[:, 1]
            bbox = BBox(
                x1=float(min(x_coords)),
                y1=float(min(y_coords)),
                x2=float(max(x_coords)),
                y2=float(max(y_coords)),
            )

            if score_threshold is not None and score[i] < score_threshold:
                continue

            res.units.append(
                ImageParseUnit(
                    image=image,
                    source_module="paddleocr",
                    score=score[i],
                    bbox=bbox,
                    type="ocr",
                    text=text[i],
                )
            )
        return res


@ModuleFactory.register_module("paddleocr")
def build_module_paddleocr():
    model = ModelLoader().get_model("paddleocr")
    return PaddleOCRModule(model)
