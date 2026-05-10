from ultralytics import YOLO

# 1. Load your model
print("Loading model...")
model = YOLO("best 1.pt")

# 2. Run prediction on the raw image (NO web app, NO FastAPI, NO OpenCV processing)
print("Scanning image...")
# Use the exact double-extension name from your sidebar
results = model.predict(source="test_image.jpg.jpeg", imgsz=1024, conf=0.25, save=True)

print("✅ Scan complete! Open the 'runs/detect/predict' folder to see the result.")