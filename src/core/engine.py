from src.pipelines.image import run_pipeline_image
from src.pipelines.video import process_video
from src.pipelines.frame import run_pipeline_frame
from src.pipelines.rtsp_usual import start_rtsp_stream, stop_rtsp_stream


class AIEngine:

    def __init__(self):
        # Храним отдельно данные кадров и отдельно объекты потоков
        self.streams = {}  # Для хранения объектов потоков (чтобы их останавливать)
        self.live_results = {}  # Для хранения последних результатов кадров (для WebSocket)

    def run_image(self, path):
        return run_pipeline_image(path)

    def run_video(self, path):
        return process_video(path)

    def process_frame(self, frame):
        return run_pipeline_frame(frame)

    def start_rtsp(self, url, camera_id="default"):
        # Callback теперь пишет в изолированный словарь live_results
        def callback(frame):
            result = self.process_frame(frame)
            self.live_results[camera_id] = {
                "frame": frame,
                "result": result
            }

        # Запускаем поток и сохраняем его объект управления
        stream = start_rtsp_stream(url, callback)
        self.streams[camera_id] = stream

        return {
            "status": "started",
            "camera_id": camera_id
        }

    def stop_rtsp(self, camera_id="default"):
        # Извлекаем объект потока
        stream = self.streams.get(camera_id)

        if stream:
            # Передаем сам объект потока во внешнюю функцию остановки
            stop_rtsp_stream(stream)
            # Полностью очищаем память от этого потока
            self.streams.pop(camera_id, None)
            self.live_results.pop(camera_id, None)
            return {"status": "stopped"}

        return {"status": "not_found", "message": f"Stream {camera_id} is not running"}

    def get_live_frame(self, camera_id="default"):
        # Забираем актуальные данные детекции из live_results
        data = self.live_results.get(camera_id)

        if not data:
            return None
        return data.get("result")
