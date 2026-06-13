from src.pipelines.image import run_pipeline_image
from src.pipelines.video import process_video
from src.pipelines.frame import run_pipeline_frame
from src.pipelines.rtsp_usual import start_rtsp_stream,stop_rtsp_stream

class AIEngine:

    def __init__(self):
        self.streams = {}

    def run_image(self, path):
        return run_pipeline_image(path)

    def run_video(self, path):
        return process_video(path)

    def process_frame(self, frame):
        return run_pipeline_frame(frame)

    def start_rtsp(self, url, camera_id="default"):
        # Запускает RTSP поток для указанной камеры
        # callback вызывается на каждом новом кадре и сохраняет последний результат
        def callback(frame):
            result = self.process_frame(frame)
            self.streams[camera_id] = {
                "frame": frame,
                "result": result
            }

        stream = start_rtsp_stream(url, callback)
        self.streams[camera_id] = stream

        return {
            "status": "started",
            "camera_id": camera_id
        }

    def stop_rtsp(self, camera_id="default"):
        # Останавливает поток и удаляет его из памяти
        stream = self.streams.get(camera_id)

        if stream:
            stop_rtsp_stream(stream)
            self.streams.pop(camera_id, None)
        return {"status": "stopped"}

    def get_live_frame(self, camera_id="default"):
        # Возвращает последний результат детекции для WebSocket
        # Если поток не запущен — возвращает None
        data = self.streams.get(camera_id)

        if not data:
            return None
        return data.get("result")