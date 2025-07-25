# api/parse_api.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Literal
from core.services.parser_manager import ParserManager

router = APIRouter()

class ParseRequest(BaseModel):
    image_base64: str
    mode: Literal["full", "region", "semantic", "zero_shot", "omni", "yolo", "paddleocr", "clip", "sam2"] = "semantic"
    prompt: Optional[str] = None

@router.post("/parse")
async def parse_image(req: ParseRequest):
    result = ParserManager().parse_image(req.image_base64, mode=req.mode, prompt=req.prompt)
    if result is None:
        raise HTTPException(status_code=404, detail="Image not found or parse failed")
    return result

@router.post("/labeling/assist")
async def assist_label(req: ParseRequest):
    # 可扩展：调用 parse_with_prompts
    return {"msg": "not implemented"}
