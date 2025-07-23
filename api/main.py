from fastapi import FastAPI
from api import router
from fastapi.staticfiles import StaticFiles

app = FastAPI()

app.include_router(router, prefix="/api")

# 方便返回静态图片预览
app.mount("/static", StaticFiles(directory="image_store"), name="static")
