import cv2
import matplotlib.pyplot as plt
import os

# --- CONFIGURATION ---
# IMPORTANT: Change these to the exact name of one of your images and its corresponding .txt file
IMAGE_PATH = r"C:\DataLatest\VID1\images\frame_00009.jpg"
LABEL_PATH = r"C:\DataLatest\VID1\labels\frame_00009.txt"

# Your exact class sequence
CLASSES = ['Rider', 'car', 'no_helmet', 'helmet', 'number_plate', 'pillion', 'bike_with_rider']

# Distinct colors for each class (RGB format for Matplotlib)
COLORS = [
    (255, 50, 50),    # 0: Rider (Red)
    (50, 255, 50),    # 1: car (Green)
    (50, 50, 255),    # 2: no_helmet (Blue)
    (255, 255, 0),    # 3: helmet (Yellow)
    (0, 255, 255),    # 4: number_plate (Cyan)
    (255, 0, 255),    # 5: pillion (Magenta)
    (255, 165, 0)     # 6: bike_with_rider (Orange)
]

def visualize_yolo_boxes(image_path, label_path):
    if not os.path.exists(image_path):
        print(f"❌ Image not found at: {image_path}")
        return
        
    # Load image
    img = cv2.imread(image_path)
    img_h, img_w = img.shape[:2]
    
    # Check if label exists
    if not os.path.exists(label_path):
        print(f"⚠️ Label file not found at: {label_path}")
        print("This usually means Florence-2 didn't detect any of your classes in this specific image.")
    else:
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    cx, cy, w, h = map(float, parts[1:5])
                    
                    # Convert YOLO normalized coordinates back to absolute pixel coordinates
                    x1 = int((cx - w/2) * img_w)
                    y1 = int((cy - h/2) * img_h)
                    x2 = int((cx + w/2) * img_w)
                    y2 = int((cy + h/2) * img_h)
                    
                    # Get class name and color
                    class_name = CLASSES[class_id] if class_id < len(CLASSES) else f"Unknown_{class_id}"
                    color = COLORS[class_id % len(COLORS)]
                    
                    # Draw Rectangle bounding box
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
                    
                    # Draw a solid background for text so it's easy to read against messy backgrounds
                    (text_w, text_h), _ = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    cv2.rectangle(img, (x1, y1 - text_h - 10), (x1 + text_w, y1), color, -1)
                    
                    # Put the class name text
                    cv2.putText(img, class_name, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    # Convert BGR (OpenCV format) to RGB (Matplotlib format) so colors look correct
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Display the image beautifully inside Colab
    plt.figure(figsize=(16, 10)) # Adjust these numbers to make it bigger or smaller
    plt.imshow(img_rgb)
    plt.axis('off') # Hide the graph axes
    plt.title(f"Florence-2 Annotations: {os.path.basename(image_path)}", fontsize=16)
    plt.show()

# Run the function
visualize_yolo_boxes(IMAGE_PATH, LABEL_PATH)