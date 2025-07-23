from fastapi import APIRouter, UploadFile, File, HTTPException
from models import *
from storage import save_image
from services import parse_image, sam2_get_mask, save_unit

router = APIRouter()

@router.post("/upload_image", response_model=UploadImageResponse)
async def upload_image(file: UploadFile = File(...)):
    try:
        image_uid, path = save_image(file)
        # 这里可以用PIL等获取尺寸，这里示范固定
        return UploadImageResponse(
            status="success",
            image_uid=image_uid,
            width=1024,
            height=768,
            preview_url=f"/static/{image_uid}.png",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/parse", response_model=ParseResponse)
async def parse(req: ParseRequest):
    data = parse_image(req.image_uid, req.settings or {})
    return ParseResponse(status="success", **data)

@router.post("/sam2_mask", response_model=Sam2MaskResponse)
async def sam2_mask(req: Sam2MaskRequest):
    unit_data = sam2_get_mask(req.image_uid, req.click["x"], req.click["y"])
    return Sam2MaskResponse(status="success", **unit_data)

@router.post("/save_unit", response_model=SaveUnitResponse)
async def save_unit_api(req: SaveUnitRequest):
    save_unit(req.result_uid, req.unit)
    return SaveUnitResponse(status="success")
