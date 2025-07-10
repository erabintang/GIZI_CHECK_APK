import replicate
import os

# API Key dari .env atau langsung
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_KEY", "r8_QGcBIKWiwv7ibZ9kscn71wYolAHOr120si1nR")
os.environ["REPLICATE_API_TOKEN"] = REPLICATE_API_TOKEN

def detect_food_yolo(image_path):
    try:
        output = replicate.run(
            "yolov8-food-detection/ultralytics-yolov8",  # Ganti ini dengan model yang sesuai kalau berbeda
            input={"image": open(image_path, "rb")}
        )
        if not output or "predictions" not in output:
            return []
        return [item["class"] for item in output["predictions"]]
    except Exception as e:
        print("Gagal pakai AI:", e)
        return []
