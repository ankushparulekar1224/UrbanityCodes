import os
import random


IMAGES_DIR = r"C:/Users/softlabs_group/Downloads/project-18-at-2026-02-23-17-41-9de2dcc9/balanced/images"
LABELS_DIR = r"C:/Users/softlabs_group/Downloads/project-18-at-2026-02-23-17-41-9de2dcc9/balanced/labels"

RIDER_ID = 0
HELMET_ID = 2
NUMBER_PLATE_ID = 3
PILLION_ID = 4

TARGET_RIDER_IMAGES = 700   # desired rider-only images to keep
# ==================

rider_only_files = []

for label_file in os.listdir(LABELS_DIR):
    if not label_file.endswith(".txt"):
        continue

    label_path = os.path.join(LABELS_DIR, label_file)

    with open(label_path, "r") as f:
        lines = f.readlines()

    if not lines:
        continue

    class_ids = set(int(line.split()[0]) for line in lines)

    # rider-dominant image (no useful supervision)
    if (
        RIDER_ID in class_ids
        and HELMET_ID not in class_ids
        and NUMBER_PLATE_ID not in class_ids
        and PILLION_ID not in class_ids
    ):
        rider_only_files.append(label_file)

print(f"Rider-dominant images found: {len(rider_only_files)}")

if len(rider_only_files) <= TARGET_RIDER_IMAGES:
    print("No rider undersampling needed.")
    exit()

remove_count = len(rider_only_files) - TARGET_RIDER_IMAGES
files_to_remove = random.sample(rider_only_files, remove_count)

for label_file in files_to_remove:
    img_file = label_file.replace(".txt", ".jpg")  # change if png

    label_path = os.path.join(LABELS_DIR, label_file)
    img_path = os.path.join(IMAGES_DIR, img_file)

    if os.path.exists(label_path):
        os.remove(label_path)
    if os.path.exists(img_path):
        os.remove(img_path)

print(f"Removed {remove_count} rider-dominant images")
print("Rider undersampling completed")
