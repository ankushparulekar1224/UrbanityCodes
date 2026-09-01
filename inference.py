from ultralytics import YOLO

# model = YOLO(r"C:/Users/ChinmayParab/Desktop/yolo26/runs/detect/train/weights/best.pt")
model = YOLO(r"C:\weights\best.pt")
# Run on a video, webcam (source=0), or RTSP stream
results = model.predict(source=r"C:/client_data/videos/VID20260215145537~2.mp4", show=True, stream=True,conf=0.5 ,save=True)

for r in results:
    # This loop runs for every frame
    pass
