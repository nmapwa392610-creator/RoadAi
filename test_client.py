from RoadAi.src.core.detector import Detector

detector = Detector()

results = detector.predict("test.jpg")

print(results)
print(results[0].boxes)