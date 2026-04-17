from src.detector import Detector

detector = Detector()

def run_pipeline(image_path):
    results = detector.predict(image_path)

    output = []

    for r in results:
        for box in r.boxes:
            output.append({
                "bbox": box.xyxy.tolist(),
                "confidence": float(box.conf),
                "class": int(box.cls)
            })

    return output