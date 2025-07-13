import ast
import chromadb
from typing import List, Optional, Dict, Any


class CustomChromaDB:
    def __init__(
        self,
        persist_directory: Optional[str] = None,
        collection_name: str = "img_parse_units",
    ):
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(collection_name)

    @staticmethod
    def preprocess_metadata(metadata):
        def preprocess(v):
            # 只允许 str, int, float, bool, dict, list, tuple, set
            if v is None:
                return ""
            if isinstance(v, (str, int, float, bool)):
                return v
            if isinstance(v, (dict, list, tuple, set)):
                return str(v)
            # 其它类型直接转字符串
            return str(v)

        return {k: preprocess(v) for k, v in metadata.items()}

    @staticmethod
    def try_eval(val):
        # 只允许反序列化为 dict, list, tuple, set
        if isinstance(val, str):
            try:
                result = ast.literal_eval(val)
                if isinstance(result, (dict, list, tuple, set)):
                    return result
                else:
                    return val
            except Exception:
                return val
        return val

    @staticmethod
    def deserialize_metadata(meta):
        # 支持输入为 dict 或 list，递归反序列化
        def _deserialize(v):
            if isinstance(v, str):
                val = CustomChromaDB.try_eval(v)
                if isinstance(val, dict):
                    return {k: _deserialize(val2) for k, val2 in val.items()}
                elif isinstance(val, (list, tuple, set)):
                    return type(val)(_deserialize(i) for i in val)
                else:
                    return val
            elif isinstance(v, dict):
                return {k: _deserialize(val2) for k, val2 in v.items()}
            elif isinstance(v, (list, tuple, set)):
                return type(v)(_deserialize(i) for i in v)
            else:
                return v

        if isinstance(meta, dict):
            return {k: _deserialize(v) for k, v in meta.items()}
        elif isinstance(meta, list):
            return [_deserialize(v) for v in meta]
        else:
            return meta

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
        metadatas = [self.preprocess_metadata(m) for m in metadatas]
        documents = [m.get("text", "") for m in metadatas]
        self.collection.add(
            ids=ids, embeddings=embeddings, metadatas=metadatas, documents=documents
        )

    def get(self, id: str) -> Dict[str, Any]:
        results = self.collection.get(ids=[id])
        # chromadb get 返回 dict，包含 ids, embeddings, metadatas, documents
        if results and "metadatas" in results and results["metadatas"]:
            return self.deserialize_metadata(results["metadatas"][0])
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
        return [self.deserialize_metadata(m) for m in metas]

    def query(self, embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        results = self.collection.query(query_embeddings=[embedding], n_results=top_k)
        metas = results.get("metadatas", [])[0]
        return [self.deserialize_metadata(m) for m in metas]

    def persist(self):
        self.client.persist()
