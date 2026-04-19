from src.pipelines.image import run_pipeline_image
from src.pipelines.video import process_video
from src.streams.rtsp_usual import start_rtsp_stream, stop_rtsp_stream, get_live_frame


class AIEngine:
    def run(self, mode: str, data):

        if mode == "image":
            return run_pipeline_image(data)

        if mode == "video":
            return process_video(data)

        if mode == "rtsp_start":
            return start_rtsp_stream(data)

        if mode == "rtsp_stop":
            return stop_rtsp_stream()

        return {"error": "unknown mode"}

    def get_live_frame(self):
        return get_live_frame()