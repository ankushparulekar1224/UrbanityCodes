import os
import shutil

# Folder that contains multiple folders of images
source_folder = r"C:\newData\extracted_frames"

# Folder where all images will be collected
destination_folder = r"C:\urbanity\ImageClient"

os.makedirs(destination_folder, exist_ok=True)

counter = 1

for root, dirs, files in os.walk(source_folder):
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            
            source_path = os.path.join(root, file)
            new_name = f"image_{counter:06d}.jpg"
            destination_path = os.path.join(destination_folder, new_name)

            shutil.copy(source_path, destination_path)
            counter += 1

print("All images collected and renamed successfully!")
