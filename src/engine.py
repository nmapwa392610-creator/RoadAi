from src.pipeline import (
    run_pipeline_image,
    process_video,
    start_rtsp_stream,
    stop_rtsp_stream,
    get_live_frame
)


class AIEngine:
    def __init__(self):
        self.rtsp_running = False

    async def run(self, mode: str, data):
        if mode == "image":
            return run_pipeline_image(data)

        if mode == "video":
            return process_video(data)

        if mode == "rtsp_start":
            self.rtsp_running = True
            return start_rtsp_stream(data)

        if mode == "rtsp_stop":
            self.rtsp_running = False
            return stop_rtsp_stream()

        return {"error": "unknown mode"}

    async def get_live_frame(self):
        return get_live_frame()
