# metadata_utils.py

import ast
from typing import Any, Union

class MetadataUtils:
    @staticmethod
    def preprocess_metadata(metadata: dict) -> dict:
        def preprocess(v: Any) -> Union[str, int, float, bool]:
            if v is None:
                return ""
            if isinstance(v, (str, int, float, bool)):
                return v
            if isinstance(v, (dict, list, tuple, set)):
                return str(v)
            return str(v)

        return {k: preprocess(v) for k, v in metadata.items()}

    @staticmethod
    def try_eval(val: Any) -> Any:
        if isinstance(val, str):
            try:
                result = ast.literal_eval(val)
                if isinstance(result, (dict, list, tuple, set)):
                    return result
                return val
            except Exception:
                return val
        return val

    @staticmethod
    def deserialize_metadata(meta: Union[dict, list, str]) -> Any:
        def _deserialize(v: Any) -> Any:
            if isinstance(v, str):
                val = MetadataUtils.try_eval(v)
                if isinstance(val, dict):
                    return {k: _deserialize(val2) for k, val2 in val.items()}
                elif isinstance(val, (list, tuple, set)):
                    return type(val)(_deserialize(i) for i in val)
                return val
            elif isinstance(v, dict):
                return {k: _deserialize(val2) for k, val2 in v.items()}
            elif isinstance(v, (list, tuple, set)):
                return type(v)(_deserialize(i) for i in v)
            return v

        if isinstance(meta, dict):
            return {k: _deserialize(v) for k, v in meta.items()}
        elif isinstance(meta, list):
            return [_deserialize(v) for v in meta]
        return meta
