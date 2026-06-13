import os
import time
import secrets
from pathlib import Path
import cv2
import magic
import logging
from urllib.parse import urlparse


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("auth")

INTERNAL_API_KEY = (os.getenv("INTERNAL_API_KEY") or "").strip()


def verify_api_key(key: str) -> bool:
    key = (key or "").strip()

    if not key:
        return False

    if not INTERNAL_API_KEY:
        logger.error("INTERNAL_API_KEY is not set in environment")
        return False

    # debug only (не включать в prod постоянно)
    logger.debug(f"API KEY received: {repr(key)}")

    return secrets.compare_digest(key, INTERNAL_API_KEY)



# ===== RATE LIMIT =====

_requests = {}

def rate_limit(ip: str, limit: int = 100, window: int = 60):
    if limit <= 0 or window <= 0:
        return False

    now = time.time()

    if ip not in _requests:
        _requests[ip] = []

    _requests[ip] = [t for t in _requests[ip] if now - t < window]

    if len(_requests[ip]) >= limit:
        return False

    _requests[ip].append(now)
    return True


# ===== FILE CHECK =====


ALLOWED_EXT = {".mp4", ".avi", ".jpg", ".jpeg", ".png"}
MAX_SIZE_MB = 100

ALLOWED_MIME = {
    "image/jpeg",
    "image/png",
    "video/mp4",
    "video/x-msvideo"
}


def check_file(file):
    # 1. extension check
    filename = file.filename or ""
    ext = Path(filename).suffix.lower()

    if ext not in ALLOWED_EXT:
        return False, "invalid file type"

    # 2. size check
    file.file.seek(0, 2)
    size = file.file.tell()
    file.file.seek(0)

    if size > MAX_SIZE_MB * 1024 * 1024:
        return False, "file too large"

    # 3. read header
    header = file.file.read(4096)
    file.file.seek(0)

    # 4. MIME check (safe fallback)
    try:
        import magic
        mime = magic.from_buffer(header, mime=True)
    except Exception:
        # fallback если magic сломан в Docker
        mime = None

    # 5. allow fallback if MIME unknown
    if mime and mime not in ALLOWED_MIME:
        return False, f"invalid file content: {mime}"

    return True, None


# ===== RTSP CHECK =====

def check_rtsp(url: str):
    url = (url or "").strip()

    if not url.startswith(("rtsp://", "rtsps://")):
        return False, "invalid protocol"

    parsed = urlparse(url)

    if not parsed.hostname:
        return False, "missing host"

    # защита от localhost / internal networks (очень важно!)
    blocked_hosts = ["localhost", "127.0.0.1", "0.0.0.0"]

    if parsed.hostname in blocked_hosts:
        return False, "blocked host"

    return True, None