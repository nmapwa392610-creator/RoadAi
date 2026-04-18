from fastapi import FastAPI, UploadFile, File, WebSocket
from pydantic import BaseModel
from pathlib import Path
import shutil
import traceback
import uuid

from src.engine import AIEngine

app = FastAPI(title="Road AI PRO 🚀")

engine = AIEngine()

UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# -------------------------
# HOME
# -------------------------
@app.get("/")
def home():
    return {"status": "running", "system": "Road AI PRO"}


# -------------------------
# IMAGE
# -------------------------
@app.post("/detect/image")
async def detect_image(file: UploadFile = File(...)):
    try:
        ext = Path(file.filename).suffix
        file_path = UPLOAD_DIR / f"{uuid.uuid4()}{ext}"

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        result = await engine.run("image", str(file_path))

        return {"result": result}

    except Exception as e:
        return {"error": str(e), "trace": traceback.format_exc()}


# -------------------------
# VIDEO
# -------------------------
@app.post("/detect/video")
async def detect_video(file: UploadFile = File(...)):
    try:
        ext = Path(file.filename).suffix
        file_path = UPLOAD_DIR / f"{uuid.uuid4()}{ext}"

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        result = await engine.run("video", str(file_path))

        return {"result": result}

    except Exception as e:
        return {"error": str(e), "trace": traceback.format_exc()}


# -------------------------
# RTSP MODEL
# -------------------------
class RTSPRequest(BaseModel):
    url: str


@app.post("/rtsp/start")
async def rtsp_start(req: RTSPRequest):
    try:
        return await engine.run("rtsp_start", req.url)
    except Exception as e:
        return {"error": str(e), "trace": traceback.format_exc()}


@app.post("/rtsp/stop")
async def rtsp_stop():
    try:
        return await engine.run("rtsp_stop", None)
    except Exception as e:
        return {"error": str(e), "trace": traceback.format_exc()}


# -------------------------
# WEBSOCKET LIVE STREAM
# -------------------------
@app.websocket("/ws/live")
async def websocket_live(ws: WebSocket):
    await ws.accept()

    try:
        while True:
            data = await engine.get_live_frame()
            await ws.send_json(data)

    except Exception as e:
        await ws.send_json({"error": str(e)})