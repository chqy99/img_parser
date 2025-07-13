import yaml
import numpy as np
from typing import Optional, List, Dict, Any
from core.imgdata.image_data import ImageParseUnit, ImageParseResult
from core.memory.embedding_handler import EmbeddingHandler
from core.memory.custom_chromadb import CustomChromaDB

class ImageMemory:
    def __init__(self, collection_name: str, config_path: Optional[str] = None, embedding_handler: Optional[EmbeddingHandler] = None):
        """
        collection_name: 必须指定，向量库集合名
        config_path: 只处理 base_dir 和 vector_db_dir
        embedding_handler: 默认不变
        """
        if config_path is None:
            # 获取当前文件所在目录，拼接 configs/memory_config.yaml
            import os
            current_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(current_dir, "..", "..", "configs", "memory_config.yaml")
            config_path = os.path.normpath(config_path)
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        self.base_dir = cfg.get("base_dir", "./img_memory")
        self.vector_db_dir = cfg.get("vector_db_dir", "./.chroma_db")
        self.collection_name = collection_name
        self.embedding_handler = embedding_handler or EmbeddingHandler()
        self.db = CustomChromaDB(persist_directory=self.vector_db_dir, collection_name=self.collection_name)

    def save_unit(self, unit: ImageParseUnit, type: str = "bbox"):
        # 保存图片
        unit.save_image(self.base_dir, image_filter=[f"{type}_image"])
        # 按 type 选取主键 embedding
        if type == "mask":
            img = unit.mask_image if unit.mask_image is not None else unit.get_mask_image()
        else:
            img = unit.bbox_image if unit.bbox_image is not None else unit.get_bbox_image()
        if img is None:
            raise ValueError(f"No valid {type}_image for embedding.")
        embedding = self.embedding_handler.get_embedding(img)
        metadata = unit.to_vector_record()
        self.db.add(f"{unit.get_uid()}_{type}", embedding, metadata)

    def save_units(self, units: List[ImageParseUnit], type: str = "bbox"):
        ids, embeddings, metadatas = [], [], []
        for u in units:
            u.save_image(self.base_dir, image_filter=[f"{type}_image"])
            if type == "mask":
                img = u.mask_image if u.mask_image is not None else u.get_mask_image()
            else:
                img = u.bbox_image if u.bbox_image is not None else u.get_bbox_image()
            if img is None:
                continue
            ids.append(f"{u.get_uid()}_{type}")
            embeddings.append(self.embedding_handler.get_embedding(img))
            metadatas.append(u.to_vector_record())
        if ids:
            self.db.add(ids, embeddings, metadatas)

    def save_result(self, result: ImageParseResult, type: str = "bbox"):
        # # 保存所有 unit 的图片
        # self.save_units(result.units, type=type)
        # 保存 result 级图片（如有）
        result.save_image(self.base_dir, image_filter=[f"{type}s_image"])
        # 按 type 选取主键 embedding
        if type == "mask":
            img = result.masks_image if result.masks_image is not None else result.get_masks_image()
        else:
            img = result.bboxs_image if result.bboxs_image is not None else result.get_bboxs_image()
        if img is None:
            raise ValueError(f"No valid result {type}s_image for embedding.")
        embedding = self.embedding_handler.get_embedding(img)
        metadata = result.to_vector_record()
        # id 加 type 后缀
        self.db.add(f"{result.get_uid()}_{type}", embedding, metadata)

    def save_raw_image(self, result: ImageParseResult):
        result.save_image(self.base_dir, image_filter=["image"])
        embedding_img = self.embedding_handler.get_embedding(result.image)
        metadata_img = result.to_vector_record()
        # 原图 embedding 单独插入，id 不加 type 后缀
        self.db.add(result.get_uid(), embedding_img, metadata_img)

    def query_result(self, img: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        embedding = self.embedding_handler.get_embedding(img)
        return self.db.query(embedding, top_k=top_k)
