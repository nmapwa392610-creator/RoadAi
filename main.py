import os
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, UploadFile, File, WebSocket, Request, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator

from src.services.file_service import run_with_temp_file
from src.pipelines.image import run_pipeline_image
from src.core.engine import AIEngine
from src.core.security import rate_limit, check_file, verify_api_key, check_rtsp

from fastapi.security import APIKeyHeader

api_key_scheme = APIKeyHeader(name="X-API-Key", auto_error=False)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.engine = AIEngine()
    yield
    app.state.engine.stop_rtsp()


app = FastAPI(title="Road AI", version="2.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_engine(request: Request) -> AIEngine:
    return request.app.state.engine


async def check_auth(_key: str | None = Depends(api_key_scheme)):
    if not _key:
        raise HTTPException(status_code=401, detail="API key missing")

    if not verify_api_key(_key):
        raise HTTPException(status_code=401, detail="Unauthorized")

    return _key


def check_rate_limit(request: Request, limit: int = 10) -> None:
    client_ip = request.headers.get("X-Forwarded-For")
    if client_ip:
        # ИСПРАВЛЕНО: Сначала берем первый элемент из списка по индексу, затем делаем .strip()
        client_ip = client_ip.split(",")[0].strip()
    else:
        client_ip = request.client.host if request.client else "127.0.0.1"

    if not rate_limit(client_ip, limit=limit, window=60):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")


@app.get("/", tags=["health"])
def home():
    return {"status": "running", "system": "Road AI", "version": "2.0"}


@app.post("/detect/image", tags=["detection"])
async def detect_image(
        request: Request,
        file: UploadFile = File(...),
        _key: str = Depends(check_auth)
):
    check_rate_limit(request)

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
async def detect_video(
        request: Request,
        file: UploadFile = File(...),
        _key: str = Depends(check_auth)
):
    check_rate_limit(request)

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
async def rtsp_start(
        req: RTSPRequest,
        request: Request,
        _key: str = Depends(check_auth)
):
    check_rate_limit(request)

    ok, err = check_rtsp(req.url)
    if not ok:
        raise HTTPException(status_code=400, detail=err)

    engine = get_engine(request)
    result = await asyncio.to_thread(engine.start_rtsp, req.url)
    return result


@app.post("/rtsp/stop", tags=["rtsp"])
async def rtsp_stop(
        request: Request,
        _key: str = Depends(check_auth)
):
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
