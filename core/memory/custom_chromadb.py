import ast
import chromadb
from typing import List, Optional, Dict, Any
from core.memory.metadata_utils import MetadataUtils


class CustomChromaDB:
    def __init__(
        self,
        persist_directory: str,
        collection_name: str,
    ):
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(collection_name)

    def add(self, ids, embeddings, metadatas):
        """
        支持单条和批量插入：
        - 单条：id(str), embedding(list), metadata(dict)
        - 批量：ids(list), embeddings(list of list), metadatas(list of dict)
        """
        if isinstance(ids, str):
            ids = [ids]
            embeddings = [embeddings]
            metadatas = [metadatas]
        # 预处理 metadatas，保证所有值为 str/int/float/bool
        metadatas = [MetadataUtils.preprocess_metadata(m) for m in metadatas]
        self.collection.add(
            ids=ids, embeddings=embeddings,
            metadatas=metadatas
        )

    def get(self, id: str) -> Dict[str, Any]:
        results = self.collection.get(ids=[id])
        # chromadb get 返回 dict，包含 ids, embeddings, metadatas
        if results and "metadatas" in results and results["metadatas"]:
            return MetadataUtils.deserialize_metadata(results["metadatas"][0])
        return {}

    def delete(self, id: str):
        self.collection.delete(ids=[id])

    def count(self) -> int:
        # chromadb 暂无 count，返回所有 id 数量
        results = self.collection.get()
        return len(results.get("ids", []))

    def get_all(self) -> List[Dict[str, Any]]:
        results = self.collection.get()
        metas = results.get("metadatas", [])
        return [MetadataUtils.deserialize_metadata(m) for m in metas]

    def query(self, embedding: List[float], topk: int = 5) -> List[str]:
            results = self.collection.query(query_embeddings=[embedding], n_results=topk)
            metadatas = results.get("metadatas", [[]])[0]
            # 元数据中需要确保有 'uid' 字段
            uids = []
            for meta in metadatas:
                meta = MetadataUtils.deserialize_metadata(meta)
                uid = meta.get("uid")
                if uid is not None:
                    uids.append(uid)
            return uids

    def persist(self):
        self.client.persist()
