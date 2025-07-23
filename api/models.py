from pydantic import BaseModel
from typing import Optional, Dict, Any

class UploadImageResponse(BaseModel):
    status: str
    image_uid: str
    width: int
    height: int
    preview_url: Optional[str]

class ParseRequest(BaseModel):
    image_uid: str
    settings: Optional[Dict[str, Any]] = None

class ParseResponse(BaseModel):
    status: str
    result_uid: str
    result: Dict[str, Any]  # ImageParseResult dict

class Sam2MaskRequest(BaseModel):
    image_uid: str
    click: Dict[str, int]  # {"x": int, "y": int}

class Sam2MaskResponse(BaseModel):
    status: str
    unit: Dict[str, Any]  # ImageParseUnit dict

class SaveUnitRequest(BaseModel):
    result_uid: str
    unit: Dict[str, Any]

class SaveUnitResponse(BaseModel):
    status: str

# 你可以继续定义 SearchRequest、SearchResponse 等
