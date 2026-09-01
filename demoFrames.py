import cv2
import os

VIDEO_PATH = "c:\\Users\\thero\\Downloads\\Realistic_car_dashcam_footage.mp4"
IMAGE_DIR = "c:\\Urbanity-data\\clientDemoFrames\\images"
os.makedirs(IMAGE_DIR, exist_ok=True)

cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
frames_per_half_second = int(fps * 0.5)

frame_idx = 0
saved_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Save a frame exactly every 0.5 seconds
    if frame_idx % frames_per_half_second == 0:
        # Pad numbers with zeros so they sort correctly alphabetically (e.g., frame_0000.jpg)
        filename = os.path.join(IMAGE_DIR, f"frame_{saved_count:04d}.jpg")
        cv2.imwrite(filename, frame)
        saved_count += 1
        
    frame_idx += 1

cap.release()
print(f"Extracted {saved_count} frames to {IMAGE_DIR}!")