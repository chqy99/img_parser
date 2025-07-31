# api/parse_api.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Literal
from core.services.parser_manager import ParserManager

router = APIRouter()

class ParseRequest(BaseModel):
    uid: str
    mode: Literal["full", "region", "semantic", "zero_shot", "omni",
                  "yolo", "paddleocr", "clip", "sam2", "groundingdino"] = "semantic"
    prompt: Optional[str] = None

@router.post("/parse")
async def parse_image(req: ParseRequest):
    result = ParserManager().parse_image(req.uid, mode=req.mode, prompts=req.prompt)
    if result is None:
        raise HTTPException(status_code=404, detail="Image not found or parse failed")
    return result

class AssistRequest(BaseModel):
    uid: str
    mode: Literal["sam2"] = "sam2"
    prompt: str

@router.post("/labeling/assist")
async def assist_label(req: AssistRequest):
    pass
