# api/upload_api.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from core.services.image_cache_service import image_cache_service

router = APIRouter()

class UploadImageRequest(BaseModel):
    image_base64: str

@router.post("/upload_image")
def upload_image(req: UploadImageRequest):
    uid = image_cache_service.upload_image(req.image_base64)
    return {"uid": uid}
