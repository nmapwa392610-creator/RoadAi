import asyncio
import traceback

from src.services.file_service import run_with_temp_file
from src.pipelines.image import run_pipeline_image

from fastapi import FastAPI, UploadFile, File, WebSocket, Request
from pydantic import BaseModel
from pathlib import Path

from src.core.engine import AIEngine
from src.core.security import rate_limit, check_file


app = FastAPI(title="Road AI")
engine = AIEngine()


@app.get("/")
def home():
    return {
        "status": "running",
        "system": "Road AI",
        "version": "2.0"
    }


# IMAGE DETECTION
@app.post("/detect/image")
async def detect_image(request: Request, file: UploadFile = File(...)):
    try:
        ip = request.client.host
        if not rate_limit(ip):
            return {"error": "rate limit exceeded"}


        ok, err = check_file(file)
        if not ok:
            return {"error": err}

        ext = Path(file.filename).suffix.lower()
        file_bytes = await file.read()


        result = await asyncio.to_thread(
            run_with_temp_file,
            file_bytes=file_bytes,
            ext=ext,
            pipeline_func=run_pipeline_image
        )

        return {
            "status": "ok",
            "result": result
        }

    except Exception as e:
        return {
            "error": str(e),
            "trace": traceback.format_exc()
        }



@app.post("/detect/video")
async def detect_video(request: Request, file: UploadFile = File(...)):
    try:
        ip = request.client.host

        if not rate_limit(ip):
            return {"error": "rate limit exceeded"}

        ok, err = check_file(file)
        if not ok:
            return {"error": err}

        ext = Path(file.filename).suffix.lower()
        file_bytes = await file.read()

        result = await asyncio.to_thread(
            run_with_temp_file,
            file_bytes=file_bytes,
            ext=ext,
            pipeline_func=engine.run_video
        )

        return {"status": "ok", "result": result}

    except Exception as e:
        return {
            "error": str(e),
            "trace": traceback.format_exc()
        }


# RTSP CONTROL
class RTSPRequest(BaseModel):
    url: str


@app.post("/rtsp/start")
async def rtsp_start(req: RTSPRequest, request: Request):
    try:
        ip = request.client.host

        if not rate_limit(ip):
            return {"error": "rate limit exceeded"}

        if not req.url.startswith("rtsp://"):
            return {"error": "invalid rtsp url"}

        return await asyncio.to_thread(
            engine.start_rtsp,
            req.url
        )

    except Exception as e:
        return {"error": str(e), "trace": traceback.format_exc()}


@app.post("/rtsp/stop")
async def rtsp_stop():
    try:
        return await asyncio.to_thread(
            engine.stop_rtsp
        )

    except Exception as e:
        return {"error": str(e), "trace": traceback.format_exc()}



@app.websocket("/ws/live")
async def ws_live(ws: WebSocket):
    await ws.accept()

    try:
        while True:
            data = engine.get_live_frame()

            if not data:
                await asyncio.sleep(0.05)
                continue

            await ws.send_json({
                "status": "ok",
                "data": data
            })


    except Exception as e:
        try:
            await ws.send_json({"error": str(e)})
        except:
            pass