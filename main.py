from dotenv import load_dotenv
load_dotenv()  # ← ПЕРВАЯ строка, до любых других импортов

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, WebSocket, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel, field_validator

from src.services.file_service import run_with_temp_file
from src.pipelines.image import run_pipeline_image
from src.core.engine import AIEngine
from src.core.security import rate_limit, check_file, verify_api_key


# ------------------------------------------------------------------
# Lifespan — ОБЯЗАТЕЛЬНО до создания app
# ------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.engine = AIEngine()
    yield
    app.state.engine.stop_rtsp()


# ------------------------------------------------------------------
# App
# ------------------------------------------------------------------

app = FastAPI(
    title="Road AI",
    version="2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    schema = get_openapi(
        title=app.title,
        version=app.version,
        routes=app.routes,
    )
    schema["components"]["securitySchemes"] = {
        "ApiKeyAuth": {
            "type": "apiKey",
            "in": "header",
            "name": "X-API-Key",
        }
    }
    schema["security"] = [{"ApiKeyAuth": []}]
    app.openapi_schema = schema
    return schema


app.openapi = custom_openapi


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def get_engine(request: Request) -> AIEngine:
    return request.app.state.engine


def guard(request: Request) -> None:
    if not verify_api_key(request):
        raise HTTPException(status_code=401, detail="Unauthorized")
    if not rate_limit(request.client.host):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")


# ------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------

@app.get("/", tags=["health"])
def home():
    return {"status": "running", "system": "Road AI", "version": "2.0"}


@app.post("/detect/image", tags=["detection"])
async def detect_image(request: Request, file: UploadFile = File(...)):
    guard(request)
    ok, err = check_file(file)
    if not ok:
        raise HTTPException(status_code=400, detail=err)
    ext = Path(file.filename).suffix.lower()
    file_bytes = await file.read()
    result = await asyncio.to_thread(
        run_with_temp_file,
        file_bytes=file_bytes,
        ext=ext,
        pipeline_func=run_pipeline_image,
    )
    return {"status": "ok", "result": result}


@app.post("/detect/video", tags=["detection"])
async def detect_video(request: Request, file: UploadFile = File(...)):
    guard(request)
    ok, err = check_file(file)
    if not ok:
        raise HTTPException(status_code=400, detail=err)
    ext = Path(file.filename).suffix.lower()
    file_bytes = await file.read()
    engine = get_engine(request)
    result = await asyncio.to_thread(
        run_with_temp_file,
        file_bytes=file_bytes,
        ext=ext,
        pipeline_func=engine.run_video,
    )
    return {"status": "ok", "result": result}


class RTSPRequest(BaseModel):
    url: str

    @field_validator("url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        if not v.startswith(("rtsp://", "rtsps://")):
            raise ValueError("URL должен начинаться с rtsp:// или rtsps://")
        return v


@app.post("/rtsp/start", tags=["rtsp"])
async def rtsp_start(req: RTSPRequest, request: Request):
    guard(request)
    engine = get_engine(request)
    result = await asyncio.to_thread(engine.start_rtsp, req.url)
    return result


@app.post("/rtsp/stop", tags=["rtsp"])
async def rtsp_stop(request: Request):
    engine = get_engine(request)
    result = await asyncio.to_thread(engine.stop_rtsp)
    return result


@app.websocket("/ws/live")
async def ws_live(ws: WebSocket):
    await ws.accept()
    engine = ws.app.state.engine
    try:
        while True:
            data = engine.get_live_frame()
            if not data:
                await asyncio.sleep(0.033)
                continue
            await ws.send_json({"status": "ok", "data": data})
    except Exception as e:
        try:
            await ws.send_json({"error": str(e)})
        except Exception:
            pass
    finally:
        await ws.close()