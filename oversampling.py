import os
import shutil

IMAGES_DIR = r"C:/Users/softlabs_group/Downloads/project-18-at-2026-02-23-17-41-9de2dcc9/images"
LABELS_DIR = r"C:/Users/softlabs_group/Downloads/project-18-at-2026-02-23-17-41-9de2dcc9/labels"
OUT_IMAGES = r"C:/Users/softlabs_group/Downloads/project-18-at-2026-02-23-17-41-9de2dcc9/balanced/images"
OUT_LABELS = r"C:/Users/softlabs_group/Downloads/project-18-at-2026-02-23-17-41-9de2dcc9/balanced/labels"

os.makedirs(OUT_IMAGES, exist_ok=True)
os.makedirs(OUT_LABELS, exist_ok=True)

PILLION_CLASS = 4
HELMET_CLASS = 2

def get_classes(label_path):
    classes = set()
    with open(label_path, "r") as f:
        for line in f:
            cls = int(line.split()[0])
            classes.add(cls)
    return classes

counter = 0

for label_file in os.listdir(LABELS_DIR):
    if not label_file.endswith(".txt"):
        continue

    label_path = os.path.join(LABELS_DIR, label_file)
    image_name = label_file.replace(".txt", ".jpg")
    image_path = os.path.join(IMAGES_DIR, image_name)

    if not os.path.exists(image_path):
        continue

    classes = get_classes(label_path)

    # default copy (once)
    copies = 1

    if PILLION_CLASS in classes:
        copies = 3
    elif HELMET_CLASS in classes:
        copies = 2

    for i in range(copies):
        new_img = f"{counter}_{image_name}"
        new_lbl = f"{counter}_{label_file}"

        shutil.copy(image_path, os.path.join(OUT_IMAGES, new_img))
        shutil.copy(label_path, os.path.join(OUT_LABELS, new_lbl))

        counter += 1

print("Oversampling completed.")
