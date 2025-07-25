from fastapi import FastAPI
from api import repository_api, upload_api, parse_api

app = FastAPI(title="img_parser API")

app.include_router(upload_api.router, prefix="/api")
app.include_router(parse_api.router, prefix="/api")
app.include_router(repository_api.router, prefix="/api")
