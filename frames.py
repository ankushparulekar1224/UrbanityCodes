import os
import cv2
from pathlib import Path

# ===== SETTINGS =====
VIDEOS_FOLDER = "c:/Urbanity-data/VideosLatest"     # folder containing videos
OUTPUT_FOLDER = "c:/Urbanity-data/ankush14/images"     # all extracted frames go here
FRAME_INTERVAL_SEC =  2   # save 1 frame every 1 second

# Supported video formats
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".mpeg")

# Create output folder
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

global_frame_count = 0

# Loop through all videos
for video_file in Path(VIDEOS_FOLDER).iterdir():

    if video_file.suffix.lower() not in VIDEO_EXTENSIONS:
        continue

    print(f"\nProcessing: {video_file.name}")

    cap = cv2.VideoCapture(str(video_file))

    if not cap.isOpened():
        print(f"Could not open {video_file.name}")
        continue

    fps = cap.get(cv2.CAP_PROP_FPS)

    # Frames to skip based on interval
    frame_interval = int(fps * FRAME_INTERVAL_SEC)

    frame_count = 0
    saved_count = 0

    video_name = video_file.stem

    while True:
        success, frame = cap.read()

        if not success:
            break

        # Save frame at interval
        if frame_count % frame_interval == 0:

            # Unique filename
            image_name = (
                f"{video_name}_frame_"
                f"{global_frame_count:06d}.jpg"
            )

            output_path = os.path.join(OUTPUT_FOLDER, image_name)

            cv2.imwrite(output_path, frame)

            saved_count += 1
            global_frame_count += 1

        frame_count += 1

    cap.release()

    print(f"Saved {saved_count} frames from {video_file.name}")

print("\nDone extracting frames.")