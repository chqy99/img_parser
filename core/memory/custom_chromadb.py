import chromadb
from typing import List, Optional, Dict, Any


class CustomChromaDB:
    def __init__(self, persist_directory: Optional[str] = None, collection_name: str = "img_parse_units"):
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
        documents = [m.get("text", "") for m in metadatas]
        self.collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas, documents=documents)

    def get(self, id: str) -> Dict[str, Any]:
        results = self.collection.get(ids=[id])
        # chromadb get 返回 dict，包含 ids, embeddings, metadatas, documents
        if results and "metadatas" in results and results["metadatas"]:
            return results["metadatas"][0]
        return {}

    def delete(self, id: str):
        self.collection.delete(ids=[id])

    def count(self) -> int:
        # chromadb 暂无 count，返回所有 id 数量
        results = self.collection.get()
        return len(results.get("ids", []))

    def get_all(self) -> List[Dict[str, Any]]:
        results = self.collection.get()
        return results.get("metadatas", [])

    def query(self, embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        results = self.collection.query(query_embeddings=[embedding], n_results=top_k)
        return results.get("metadatas", [])

    def persist(self):
        self.client.persist()
