import asyncio
from src.engine import AIEngine

engine = AIEngine()

asyncio.run(engine.run("video", "data/videos/test.mp4"))