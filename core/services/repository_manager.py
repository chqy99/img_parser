# 存储服务，增删改查
import yaml
import numpy as np
from typing import Optional, List, Dict, Any
from core.entity.image_parse_data import ImageParseUnit, ImageParseResult
from core.repository.embedding_handler import EmbeddingHandler
from core.repository.custom_chromadb import CustomChromaDB
from core.repository.custom_sql import SQLHandler
from core.repository.custom_image_storage import ImageStorageHandler


class RepositoryManager:
    def __init__(
        self,
        collection_name: str,
        config_path: Optional[str] = None,
        embedding_handler: Optional[EmbeddingHandler] = None,
        result_filter: Optional[List[str]] = ['image', 'bboxs_image', 'masks_image'],
        unit_filter: Optional[List[str]] = ['mask'],
    ):
        """
        collection_name: 必须指定，向量库集合名
        config_path: 给定 base_dir, vector_db_dir, sql_path 等配置文件路径
        embedding_handler: 可以传入自定义的 EmbeddingHandler 实例，只保留 image 的 embedding
        result_filter: 结果级图片保存过滤器，默认 ['image', 'bboxs_image', 'masks_image']
        unit_filter: 单元级图片保存过滤器，默认 ['mask']
        """
        if config_path is None:
            # 获取当前文件所在目录，拼接 configs/memory_config.yaml
            import os

            current_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(
                current_dir, "..", "..", "configs", "memory_config.yaml"
            )
            config_path = os.path.normpath(config_path)
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        base_dir = cfg.get("base_dir", "./img_memory")
        vector_db_dir = cfg.get("vector_db_dir", "./.chroma_db")
        sql_path = cfg.get("sql_path", "./image.sqlite")


        self.embedding_handler = embedding_handler or EmbeddingHandler()
        self.collection_name = collection_name
        self.custom_chromadb = CustomChromaDB(
            persist_directory=vector_db_dir, collection_name=self.collection_name
        )

        self.image_storage_handler = ImageStorageHandler(base_dir)
        self.sql_handler = SQLHandler(sql_path)

        self.result_filter = result_filter
        self.unit_filter = unit_filter


    def save_result(self, result: ImageParseResult):
        # 1. 保存图像（包含 result 和 unit）
        self.image_storage_handler.save_images(result, self.result_filter)
        for unit in result.units:
            self.image_storage_handler.save_images(unit, self.unit_filter)

        # 2. 提取 image 的嵌入并保存至向量数据库
        if result.image is not None:
            embedding = self.embedding_handler.get_image_embedding(result.image)
            self.custom_chromadb.add(
                ids=result.uid,
                embeddings=embedding,
                metadatas={"uid": result.uid}  # metadata 只会包含 uid
            )

        # 3. 保存结构化信息到 SQLite
        self.sql_handler.save_result(result)

    def get_result(self, uid: str, as_object: bool = False) -> ImageParseResult:
        # 1. 从 SQLite 获取结果对象
        result_obj = self.sql_handler.fetch_result(uid)
        if not as_object:
            return result_obj
        if not result_obj:
            return None
        # 2. 作为对象时，加载图像数据
        self.image_storage_handler.load_images(result_obj, self.result_filter)

        return result_obj

    def query_result(self, query_text: str, query_image: np.ndarray, topk: int = 1, as_object: bool = False) -> List[str]:
        # Step 1: Text-based fuzzy search
        uids_text = self.sql_handler.fuzzy_query(query_text, topk=topk * 3) if query_text else []
        print(uids_text)

        if query_image is None:
            return [self.get_result(uid, as_object) for uid in uids_text[:topk]]

        # Step 2: Embedding search
        emb = self.embedding_handler.get_image_embedding(query_image)
        uids_embed = self.custom_chromadb.query(emb, topk=topk * 3)

        # Step 3: Union of candidates
        candidate_uids = list(set(uids_text + uids_embed))
        print(candidate_uids)

        # Step 4: Precise image match ranking
        scored = [(uid, self.image_match_score(uid, query_image)) for uid in candidate_uids]
        scored.sort(key=lambda x: x[1], reverse=True)
        print(f"Scored results: {scored}")

        return [self.get_result(uid, as_object) for uid, score in scored[:topk]]


    def image_match_score(self, uid: str, query_image: np.ndarray) -> float:
        """
        对比 query_image 与已存储图像中与 uid 对应的 base_image 区域内容是否一致。
        忽略 base_image 中为 0 的区域，只对非零区域进行像素精确比对。
        """
        # 获取参考图像
        base_image = self.image_storage_handler.get_image_by_uid(uid, "masks_image")
        if base_image is None:
            base_image = self.image_storage_handler.get_image_by_uid(uid, "bboxs_image")
        if base_image is None:
            base_image = self.image_storage_handler.get_image_by_uid(uid, "image")
        if base_image is None:
            return 0.0

        # 尺寸检查
        if base_image.shape != query_image.shape:
            return 0.0  # 不同尺寸无法比较

        # 创建掩码：仅比较 base_image 中非 0 的区域
        if len(base_image.shape) == 3:
            base_mask = np.any(base_image != 0, axis=-1)  # 彩色图像：只要有一个通道非 0 就算有效
        else:
            base_mask = base_image != 0  # 灰度图像

        # 比较值相同的区域（仅在 mask 内）
        if len(base_image.shape) == 3:
            match = np.all(base_image == query_image, axis=-1)
        else:
            match = base_image == query_image

        # 有效区域 + 匹配区域
        valid_points = np.count_nonzero(base_mask)
        if valid_points == 0:
            return 0.0  # 避免除以 0

        match_points = np.count_nonzero(match & base_mask)
        score = match_points / valid_points
        return float(score)

    def update_result(self, uid: str, update_fields: dict):
        """
        更新解析结果（结构化信息、图片、embedding）。
        update_fields: 可能包含结构化字段、图片（base64）、embedding等。
        """
        # 1. 更新结构化信息
        self.sql_handler.update_result(uid, update_fields)
        # 2. 如有图片更新
        images = update_fields.get("images", {})
        for img_type, img_bytes in images.items():
            self.image_storage_handler.save_image_by_bytes(uid, img_type, img_bytes)
        # 3. 如有 embedding 更新
        embedding = update_fields.get("embedding", None)
        if embedding is not None:
            self.custom_chromadb.add(ids=uid, embeddings=embedding, metadatas={"uid": uid})

    def delete_result(self, uid: str):
        """
        删除解析结果（结构化、图片、embedding）。
        """
        # 1. 删除结构化信息
        self.sql_handler.delete_result(uid)
        # 2. 删除所有相关图片
        for t in ["image", "bboxs_image", "masks_image", "mask", "bbox_image", "mask_image"]:
            self.image_storage_handler.delete_image(uid, t)
        # 3. 删除向量库 embedding
        self.custom_chromadb.delete(uid)

    def list_results(self, filters: dict = None, as_object: bool = False):
        """
        批量查询，支持条件过滤。
        """
        return self.sql_handler.list_results(filters, as_object)
