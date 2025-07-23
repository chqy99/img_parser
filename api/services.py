def parse_image(image_uid: str, settings: dict) -> dict:
    # 这里调用你后端的解析模块
    # 伪返回：
    return {
        "result_uid": f"res_{uuid.uuid4().hex[:8]}",
        "result": {
            "units": [],
            "summary_text": "示范返回",
            "metadata": {},
        }
    }

def sam2_get_mask(image_uid: str, x: int, y: int) -> dict:
    # 调用SAM2模型生成mask，生成unit结构
    # 伪返回：
    return {
        "unit": {
            "bbox": {"x1": 10, "y1": 10, "x2": 100, "y2": 100},
            "source_module": "sam2",
            "mask": None,
            "text": "辅助分割示范",
        }
    }

def save_unit(result_uid: str, unit: dict):
    # 这里保存到数据库或缓存
    pass
