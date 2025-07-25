# api/upload_api.py
from fastapi import APIRouter, UploadFile, HTTPException
from core.services.parser_manager import ParserManager

router = APIRouter()

@router.post("/upload")
async def upload_image(file: UploadFile):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid image file")
    # 读取文件内容并转 base64
    content = await file.read()
    import base64
    image_base64 = base64.b64encode(content).decode()
    image_id = ParserManager().upload_image(image_base64)
    return {"image_id": image_id}
