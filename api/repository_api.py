# api/retrieval_api.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Literal
from core.services.repository_manager import RepositoryManager

router = APIRouter()

repo_mgr = RepositoryManager(collection_name="default")

class RetrievalQuery(BaseModel):
    query_text: Optional[str]
    query_image_base64: Optional[str]
    top_k: int = 10
    mode: Literal["text", "image", "hybrid"] = "text"

@router.post("/search")
async def search(req: RetrievalQuery):
    query_image = None
    if req.query_image_base64:
        import base64
        import numpy as np
        from core.entity.image_parse_data import base64_to_np
        query_image = base64_to_np(req.query_image_base64)
    results = repo_mgr.query_result(query_text=req.query_text, query_image=query_image, topk=req.top_k, as_object=False)
    return results

class UpdateResultRequest(BaseModel):
    result_id: str
    update_fields: dict

@router.post("/result/add")
async def add_result(result: dict):
    from core.entity.image_parse_data import ImageParseResult
    obj = ImageParseResult.from_dict(result)
    repo_mgr.save_result(obj)
    return {"msg": "ok"}

@router.put("/result/update")
async def update_result(req: UpdateResultRequest):
    repo_mgr.update_result(req.result_id, req.update_fields)
    return {"msg": "ok"}

@router.delete("/result/delete")
async def delete_result(result_id: str):
    repo_mgr.delete_result(result_id)
    return {"msg": "ok"}
