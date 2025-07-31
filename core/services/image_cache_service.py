# core/services/image_cache_service.py
from threading import Lock
from collections import OrderedDict
import time
from core.entity.image_parse_data import IDGenerator, base64_to_np


class ImageCachePool:
    def __init__(self, max_size=100, ttl=300):
        self.max_size = max_size
        self.ttl = ttl
        self.pool = OrderedDict()
        self.lock = Lock()

    def add_image(self, image) -> str:
        uid = IDGenerator.instance().next_id()
        with self.lock:
            self.pool[uid] = (time.time(), image)
            self.pool.move_to_end(uid)
            self._evict_if_needed()
        return uid

    def get_image(self, uid):
        with self.lock:
            item = self.pool.get(uid)
            if not item:
                return None
            ts, image = item
            if time.time() - ts > self.ttl:
                self.pool.pop(uid, None)
                return None
            return image

    def _evict_if_needed(self):
        while len(self.pool) > self.max_size:
            self.pool.popitem(last=False)


class ImageCacheService:
    def __init__(self):
        self.pool = ImageCachePool(max_size=100, ttl=300)

    def upload_image(self, image_base64: str) -> str:
        image = base64_to_np(image_base64)
        return self.pool.add_image(image)

    def get_image_by_uid(self, uid: str):
        return self.pool.get_image(uid)


# ✅ 全局单例：其他 service 直接 import 使用
image_cache_service = ImageCacheService()
