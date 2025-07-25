from ultralytics import YOLO
import numpy as np
from typing import List

from core.entity.image_parse_data import BBox, ImageParseUnit, ImageParseResult
from core.modules.base import BaseModule
from core.modules.model_config import ModelLoader
from core.modules.module_factory import ModuleFactory


@ModelLoader.register_loader("yolo")
def load_model_yolo(cfg, device):
    ckpt = cfg.get("checkpoint")
    model = YOLO(ckpt).to(device)
    return model


class YoloModule(BaseModule):
    def __init__(self, model):
        self.model = model

    def parse(self, image: np.ndarray, **kwargs) -> ImageParseResult:
        # 确保输入是 RGB 3 通道
        if image.ndim == 3 and image.shape[2] == 4:
            image = image[:, :, :3]  # 去掉 alpha 通道
        elif image.ndim == 2:  # 灰度图 -> RGB
            image = np.stack([image] * 3, axis=-1)

        result = self.model.predict(image, verbose=False)[0]

        boxes = result.boxes
        names = result.names  # {0: 'icon'}

        parse_units: List[ImageParseUnit] = []

        max_area = kwargs.get("max_area", None)  # 用户可指定面积上限

        for i in range(len(boxes)):
            xyxy = boxes.xyxy[i].cpu().numpy().tolist()  # [x1, y1, x2, y2]
            conf = float(boxes.conf[i].item())
            cls_id = int(boxes.cls[i].item())
            label = names.get(cls_id, str(cls_id))  # 类别名字符串

            bbox = BBox(*xyxy)  # 假设 BBox 接受 4 个 float 值
            if max_area is not None and bbox.area() > max_area:
                continue  # 跳过过大的框

            unit = ImageParseUnit(
                image=image,
                source_module="yolo",
                score=conf,
                bbox=bbox,
                type="region",
                label=label,
            )
            parse_units.append(unit)

        return ImageParseResult(image=image, units=parse_units)


@ModuleFactory.register_module("yolo")
def build_module_yolo():
    model = ModelLoader().get_model("yolo")
    return YoloModule(model)
