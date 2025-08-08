# api/retrieval_api.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Literal, List
from core.services.repository_manager import RepositoryManager
from core.entity.image_parse_data import ImageParseResult, ImageParseUnit
from core.services.image_cache_service import image_cache_service

router = APIRouter()

repo_mgr = RepositoryManager(collection_name="default")

class RetrievalQuery(BaseModel):
    query_text: Optional[str]
    query_image_base64: Optional[str]
    top_k: int = 5

@router.post("/search")
async def search_results(req: RetrievalQuery):
    """
    检索接口，支持文字、图片、混合检索（hybrid 仅为 demo，TODO 后续精细化）。
    """
    query_image = None
    if req.query_image_base64:
        from core.entity.image_parse_data import base64_to_np
        query_image = base64_to_np(req.query_image_base64)
        results = repo_mgr.query_result(query_text=req.query_text, query_image=query_image, topk=req.top_k)
    else:
        raise HTTPException(400, "mode must be one of text/image/hybrid")
    return results

@router.get("/result")
def get_result(result_uid: str):
    """
    获取指定 result 的完整信息（含 units）。
    """
    result = repo_mgr.get_result(result_uid, as_object=True)
    if not result:
        raise HTTPException(404, "result not found")
    return result.to_dict(image_filter=[], unit_image_filter=["mask"])

class ResultRequest(BaseModel):
    result: dict

@router.post("/result/add")
def add_result(req: ResultRequest):
    """
    上传 result 对象，已存在则只增加不重复 unit，重复只警告。
    """
    req.result["image"] = image_cache_service.get_image_by_uid(req.result["uid"])
    result = ImageParseResult.from_dict(req.result)
    # 检查 result 是否已存在
    old = repo_mgr.get_result(result.uid, as_object=True)
    if old:
        # 只追加不重复 unit
        duplicate_uids = repo_mgr.sql_handler.append_units(result.uid, result.units)
        msg = f"result 已存在，已追加 unit，重复 unit: {duplicate_uids}" if duplicate_uids else "result 已存在，已追加 unit，无重复"
        return {"msg": msg, "duplicate_unit_uids": duplicate_uids}
    else:
        repo_mgr.save_result(result)
        return {"msg": "result 新增成功"}

@router.post("/result/update")
def update_result(req: ResultRequest):
    """
    只允许修改 result 的 summary_text/metadata，unit 只允许 label、text、score、metadata 字段更新。
    """
    # 更新 result
    req.result["image"] = image_cache_service.get_image_by_uid(req.result["uid"])
    result = ImageParseResult.from_dict(req.result)
    update_fields = {}
    if result.summary_text is not None:
        update_fields["summary_text"] = result.summary_text
    if result.metadata is not None:
        update_fields["metadata"] = str(result.metadata)
    if update_fields:
        repo_mgr.sql_handler.update_result(result.uid, update_fields)
    # 更新 units
    if result.units:
        units = [ImageParseUnit.from_dict(u) for u in result.units]
        repo_mgr.sql_handler.replace_units(result.uid, units)
    return {"msg": "update ok"}

class DeleteUnitsRequest(BaseModel):
    result_uid: str
    unit_uids: List[str]

@router.post("/units/delete")
def delete_units(req: DeleteUnitsRequest):
    """
    删除指定 result 下的若干 unit。
    """
    repo_mgr.sql_handler.delete_units(req.result_uid, req.unit_uids)
    return {"msg": "delete_units ok"}

class DeleteResultsRequest(BaseModel):
    result_uids: List[str]

@router.post("/results/delete")
def delete_results(req: DeleteResultsRequest):
    """
    删除若干 result 及其所有 units。
    """
    for uid in req.result_uids:
        repo_mgr.delete_result(uid)
    return {"msg": "delete_results ok"}

