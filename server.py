from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import uvicorn
from api import repository_api, upload_api, parse_api

app = FastAPI(title="img_parser API")

# ✅ 添加这一段！
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/ping")
def ping():
    return {"time": datetime.now().isoformat(timespec="seconds") + "Z"}

app.include_router(upload_api.router, prefix="/api")
app.include_router(parse_api.router, prefix="/api")
app.include_router(repository_api.router, prefix="/api")

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8200, reload=False)
