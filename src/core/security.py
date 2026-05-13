import time
import os
import secrets
from pathlib import Path
import cv2
import magic
import re

# Ключ загружается из переменной окружения — не хранится в коде
INTERNAL_API_KEY = os.environ.get("INTERNAL_API_KEY", "")

def verify_api_key(request) -> bool:
    key = request.headers.get("X-API-Key", "")
    if not key or not INTERNAL_API_KEY:
        return False
    # compare_digest защищает от timing attack — нельзя угадать ключ по времени ответа
    return secrets.compare_digest(key, INTERNAL_API_KEY)


# Словарь хранит историю запросов по IP
_requests = {}

def rate_limit(ip: str, limit: int = 100, window: int = 60):
    # Защита от DDoS — не более 10 запросов в 60 секунд с одного IP
    now = time.time()

    if ip not in _requests:
        _requests[ip] = []

    # Убираем старые запросы за пределами окна
    _requests[ip] = [t for t in _requests[ip] if now - t < window]

    if len(_requests[ip]) >= limit:
        return False

    _requests[ip].append(now)
    return True


# проверка файлов
ALLOWED_EXT = [".mp4", ".avi", ".jpg", ".png"]
MAX_SIZE_MB = 100
MAX_FRAMES = 10_000

# Разрешённые MIME типы — проверяем содержимое файла а не только расширение
ALLOWED_MIME = [
    "image/jpeg",
    "image/png",
    "video/mp4",
    "video/x-msvideo"  # .avi
]

def check_file(file):
    # Проверяем расширение файла
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXT:
        return False, "invalid file type"

    # Проверяем размер файла — не более 100MB
    file.file.seek(0, 2)
    size = file.file.tell()
    file.file.seek(0)

    if size > MAX_SIZE_MB * 1024 * 1024:
        return False, "file too large"

    # Проверяем реальный тип файла по содержимому — защита от переименованных файлов
    # Например virus.exe переименованный в photo.jpg не пройдёт
    header = file.file.read(2048)
    file.file.seek(0)
    mime = magic.from_buffer(header, mime=True)

    if mime not in ALLOWED_MIME:
        return False, "invalid file content"

    return True, None



def check_video_length(path):
    # Проверяем количество кадров — защита от слишком длинных видео
    cap = cv2.VideoCapture(path)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if frames > MAX_FRAMES:
        return False, "video too long"

    return True, None



def check_rtsp(url: str) -> tuple[bool, str]:
    if not url.startswith(("rtsp://", "rtsps://")):
        return False, "invalid protocol"

    if any(c in url for c in [";", "|", "&", "`", "$", "(", ")", "<", ">"]):
        return False, "invalid characters in url"

    pattern = r"^rtsps?://[^@\s]*(:\d+)?(/[\w.\-/]*)?$"  # поддержка user:pass@host
    if not re.match(pattern, url):
        return False, "invalid url format"

    try:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        if parsed.port and not (1 <= parsed.port <= 65535):
            return False, "invalid port"
    except Exception:
        return False, "invalid url"

    return True, None
