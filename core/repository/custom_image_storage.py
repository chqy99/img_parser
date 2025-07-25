import os
import warnings
import numpy as np
from typing import List, Optional, Union
from PIL import Image

from core.entity.image_parse_data import ImageParseUnit, ImageParseResult

class ImageStorageHandler:
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)
        print(f"[ImageStorageManager] Initialized with base_dir: {self.base_dir}")

    def save_images(self, obj: Union[ImageParseUnit, ImageParseResult], image_filter: Optional[List[str]] = []):
        uid = getattr(obj, "uid", None)
        if not uid:
            warnings.warn(f"[ImageStorageManager] Missing 'uid' for object, skip saving.")
            return

        storage_dict = getattr(obj, "storage_dict", {})

        for field in image_filter:
            # 优先尝试调用 get_{field} 方法
            getter = getattr(obj, f"get_{field}", None)
            if callable(getter):
                img = getter()
            else:
                img = getattr(obj, field, None)

            if img is None:
                continue

            # 获取路径
            path = storage_dict.get(f"{field}_path")
            if not path:
                path = self._default_image_path(uid, field)
                print(f"[ImageStorageManager] No explicit path for '{field}', using default: {path}")
                storage_dict[f"{field}_path"] = path

            try:
                if isinstance(img, np.ndarray):
                    img = Image.fromarray(img)
                img.save(path)
                print(f"[ImageStorageManager] Saved '{field}' to {path}")
            except Exception as e:
                warnings.warn(f"[ImageStorageManager] Failed to save {field} for {uid}: {e}")

        obj.storage_dict = storage_dict

    def load_images(self, obj: Union[ImageParseUnit, ImageParseResult], image_filter: Optional[List[str]] = None):
        uid = getattr(obj, "uid", None)
        if not uid:
            warnings.warn(f"[ImageStorageManager] Missing 'uid' for object, skip loading.")
            return

        image_fields = image_filter if image_filter is not None else self._get_image_fields(obj)
        storage_dict = getattr(obj, "storage_dict", {})

        for field in image_fields:
            path = storage_dict.get(f"{field}_path")
            if not path:
                path = self._default_image_path(uid, field)
                print(f"[ImageStorageManager] No explicit path for '{field}', using default: {path}")
                storage_dict[f"{field}_path"] = path

            if not os.path.exists(path):
                warnings.warn(f"[ImageStorageManager] Path not found for '{field}': {path}")
                continue

            try:
                img = Image.open(path).convert("RGB")
                if isinstance(img, Image.Image):
                    img = np.array(img)
                setattr(obj, field, img)
                print(f"[ImageStorageManager] Loaded '{field}' from {path}")
            except Exception as e:
                warnings.warn(f"[ImageStorageManager] Failed to load {field} from {path}: {e}")

        obj.storage_dict = storage_dict

    def get_image_by_uid(self, uid: str, field: str) -> Optional[Image.Image]:
        """
        根据 uid 和字段获取图像，如果不存在则返回 None。
        """
        path = self._default_image_path(uid, field)
        if not os.path.exists(path):
            warnings.warn(f"[ImageStorageManager] Image path not found: {path}")
            return None
        try:
            img = Image.open(path).convert("RGB")
            if isinstance(img, Image.Image):
                img = np.array(img)
            return img
        except Exception as e:
            warnings.warn(f"[ImageStorageManager] Failed to load image from {path}: {e}")
            return None

    def save_image_by_bytes(self, uid: str, image_type: str, image_bytes: bytes):
        """
        直接保存 base64/bytes 图片到指定路径。
        """
        from PIL import Image
        import io
        import base64
        path = self._default_image_path(uid, image_type)
        try:
            if isinstance(image_bytes, str):
                image_bytes = base64.b64decode(image_bytes)
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            img.save(path)
        except Exception as e:
            warnings.warn(f"[ImageStorageManager] Failed to save image by bytes: {e}")

    def delete_image(self, uid: str, image_type: str):
        path = self._default_image_path(uid, image_type)
        if os.path.exists(path):
            try:
                os.remove(path)
            except Exception as e:
                warnings.warn(f"[ImageStorageManager] Failed to delete image: {e}")

    def _get_image_fields(self, obj: Union[ImageParseUnit, ImageParseResult]) -> List[str]:
        candidate_fields = [
            "image", "bbox_image", "mask", "mask_image",
            "bboxs_image", "masks", "masks_image"
        ]
        return [field for field in candidate_fields if hasattr(obj, field)]

    def _default_image_path(self, uid: str, field: str) -> str:
        """
        Generate a default image path based on uid and field.
        """
        return os.path.join(self.base_dir, f"{uid}_{field}.png")