import time
from pathlib import Path

# RATE LIMIT
_requests = {}

def rate_limit(ip: str, limit: int = 10, window: int = 60):
    now = time.time()

    if ip not in _requests:
        _requests[ip] = []

    _requests[ip] = [t for t in _requests[ip] if now - t < window]

    if len(_requests[ip]) >= limit:
        return False

    _requests[ip].append(now)
    return True


# FILE CHECKS
ALLOWED_EXT = [".mp4", ".avi", ".jpg", ".png"]
MAX_SIZE_MB = 100
MAX_FRAMES = 10_000


def check_file(file):
    ext = Path(file.filename).suffix.lower()

    if ext not in ALLOWED_EXT:
        return False, "invalid file type"

    file.file.seek(0, 2)
    size = file.file.tell()
    file.file.seek(0)

    if size > MAX_SIZE_MB * 1024 * 1024:
        return False, "file too large"

    return True, None


# VIDEO LENGTH CHECK
import cv2

def check_video_length(path):
    cap = cv2.VideoCapture(path)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if frames > MAX_FRAMES:
        return False, "video too long"

    return True, None


# RTSP CHECK
def check_rtsp(url: str):
    if not url.startswith("rtsp://"):
        return False
    return True