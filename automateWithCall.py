import cv2
import os
import math
from ultralytics import YOLO

# --- CONFIGURATION ---
MODEL_PATH ="C:\\Urbanity\\best .pt"  # Path to your YOLO model
IMAGE_DIR = "c:\\Users\\thero\\Downloads\\Mayuri\\Mayuri\\images"  # Path to your images directory
LABEL_DIR = "c:\\Users\\thero\\Downloads\\Mayuri\\Mayuri\\labels" 
PROGRESS_FILE = 'last_index.txt'

os.makedirs(LABEL_DIR, exist_ok=True)

# UPDATED: Added class 7 for manual drawing
CLASSES = {
    0: 'Rider', 
    1: 'car', 
    2: 'no_helmet', 
    3: 'helmet', 
    4: 'number_plate', 
    5: 'pillion', 
    6: 'bike_with_rider',
    7: 'callling_posture' 
}


# Added an extra color for the new class
COLORS = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (0,255,255), (255,0,255), (255,255,255), (0,165,255)]

# --- GLOBAL STATE ---
boxes = []          
current_class = 1   
start_pt = (-1, -1)
current_pt = (-1, -1)

# Drag & Drop State
selected_idx = -1
action = None 
drag_offset_x = 0
drag_offset_y = 0

def get_action(x, y, boxes_list):
    TOLERANCE = 10 
    for i, b in enumerate(boxes_list):
        cid, x1, y1, x2, y2 = b
        if math.hypot(x-x1, y-y1) < TOLERANCE: return i, 'resize_tl'
        if math.hypot(x-x2, y-y2) < TOLERANCE: return i, 'resize_br'
        if math.hypot(x-x2, y-y1) < TOLERANCE: return i, 'resize_tr'
        if math.hypot(x-x1, y-y2) < TOLERANCE: return i, 'resize_bl'
        
    for i, b in enumerate(boxes_list):
        cid, x1, y1, x2, y2 = b
        if min(x1, x2) < x < max(x1, x2) and min(y1, y2) < y < max(y1, y2): 
            return i, 'move'
            
    return -1, 'draw'

def mouse_callback(event, x, y, flags, param):
    global start_pt, current_pt, boxes, current_class
    global selected_idx, action, drag_offset_x, drag_offset_y
    
    if event == cv2.EVENT_LBUTTONDOWN:
        selected_idx, action = get_action(x, y, boxes)
        if action == 'draw':
            start_pt = (x, y)
            current_pt = (x, y)
        elif action == 'move':
            cid, x1, y1, x2, y2 = boxes[selected_idx]
            drag_offset_x = x - min(x1, x2)
            drag_offset_y = y - min(y1, y2)
            
    elif event == cv2.EVENT_MOUSEMOVE:
        if action == 'draw':
            current_pt = (x, y)
        elif action == 'move' and selected_idx != -1:
            cid, x1, y1, x2, y2 = boxes[selected_idx]
            w, h = abs(x2 - x1), abs(y2 - y1)
            nx1 = x - drag_offset_x
            ny1 = y - drag_offset_y
            boxes[selected_idx] = [cid, nx1, ny1, nx1+w, ny1+h]
        elif action and action.startswith('resize') and selected_idx != -1:
            cid, x1, y1, x2, y2 = boxes[selected_idx]
            if action == 'resize_tl': boxes[selected_idx] = [cid, x, y, x2, y2]
            elif action == 'resize_br': boxes[selected_idx] = [cid, x1, y1, x, y]
            elif action == 'resize_tr': boxes[selected_idx] = [cid, x1, y, x, y2]
            elif action == 'resize_bl': boxes[selected_idx] = [cid, x, y1, x2, y]
            
    elif event == cv2.EVENT_LBUTTONUP:
        if action == 'draw':
            x_min, x_max = min(start_pt[0], x), max(start_pt[0], x)
            y_min, y_max = min(start_pt[1], y), max(start_pt[1], y)
            if x_max - x_min > 5 and y_max - y_min > 5:
                boxes.append([current_class, x_min, y_min, x_max, y_max])
        elif action and action.startswith('resize') and selected_idx != -1:
            cid, x1, y1, x2, y2 = boxes[selected_idx]
            boxes[selected_idx] = [cid, min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
        action = None
        
    elif event == cv2.EVENT_RBUTTONDOWN:
        idx, act = get_action(x, y, boxes)
        if act == 'move' or (act and act.startswith('resize')):
            boxes.pop(idx)

def save_yolo_format(img_width, img_height, filename):
    filepath = os.path.join(LABEL_DIR, filename.replace('.jpg', '.txt').replace('.png', '.txt').replace('.jpeg', '.txt'))
    with open(filepath, 'w') as f: 
        for b in boxes:
            cid, x1, y1, x2, y2 = b
            cx = ((x1 + x2) / 2) / img_width
            cy = ((y1 + y2) / 2) / img_height
            w = abs(x2 - x1) / img_width
            h = abs(y2 - y1) / img_height
            f.write(f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

def load_yolo_format(img_width, img_height, filename):
    filepath = os.path.join(LABEL_DIR, filename.replace('.jpg', '.txt').replace('.png', '.txt').replace('.jpeg', '.txt'))
    loaded_boxes = []
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    cid = int(parts[0])
                    cx, cy, w, h = map(float, parts[1:])
                    x1 = int((cx - w/2) * img_width)
                    y1 = int((cy - h/2) * img_height)
                    x2 = int((cx + w/2) * img_width)
                    y2 = int((cy + h/2) * img_height)
                    loaded_boxes.append([cid, x1, y1, x2, y2])
    return loaded_boxes

def main():
    global boxes, current_class
    model = YOLO(MODEL_PATH)
    image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    image_files.sort()
    
    start_idx = 0
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r') as f:
            try: start_idx = int(f.read().strip())
            except ValueError: pass

    cv2.namedWindow('Urbanity Auto-Labeler', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('Urbanity Auto-Labeler', mouse_callback)

    idx = start_idx
    while idx < len(image_files) and idx >= 0:
        img_name = image_files[idx]
        img_path = os.path.join(IMAGE_DIR, img_name)
        frame = cv2.imread(img_path)
        if frame is None:
            idx += 1
            continue
            
        img_h, img_w = frame.shape[:2]
        
        # Load existing labels first
        boxes = load_yolo_format(img_w, img_h, img_name)
        
        # Run AI prediction only if no manual label file exists
        if not boxes:
            results = model.predict(img_path, conf=0.4, verbose=False)
            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    boxes.append([cls_id, x1, y1, x2, y2])
        
        while True:
            display = frame.copy()
            
            for b in boxes:
                cid, x1, y1, x2, y2 = b
                color = COLORS[cid] if cid < len(COLORS) else (255, 255, 255)
                cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
                cv2.putText(display, CLASSES.get(cid, "Unknown"), (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Draw handles for resizing
                cv2.circle(display, (x1, y1), 4, color, -1)
                cv2.circle(display, (x2, y2), 4, color, -1)
                cv2.circle(display, (x2, y1), 4, color, -1)
                cv2.circle(display, (x1, y2), 4, color, -1)
                
            if action == 'draw':
                cv2.rectangle(display, start_pt, current_pt, COLORS[current_class], 2)
                
            # UI Info
            ui_text = f"Img: {idx+1}/{len(image_files)} | Class ({current_class}): {CLASSES.get(current_class, 'Unknown')}"
            cv2.putText(display, ui_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display, "0-7: Select Class | L-Click: Draw/Move | R-Click: Delete", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(display, "N: Next | B: Back | Q: Quit", (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow('Urbanity Auto-Labeler', display)
            key = cv2.waitKey(1) & 0xFF
            
            # UPDATED: Handle 0-7 keys for class selection
            if ord('0') <= key <= ord('7'): 
                current_class = key - ord('0')
            elif key == ord('n') or key == ord('N'):
                save_yolo_format(img_w, img_h, img_name)
                with open(PROGRESS_FILE, 'w') as f: f.write(str(idx + 1))
                idx += 1
                break
            elif key == ord('b') or key == ord('B'):
                save_yolo_format(img_w, img_h, img_name) 
                if idx > 0:
                    idx -= 1
                    with open(PROGRESS_FILE, 'w') as f: f.write(str(idx))
                break
            elif key == ord('q') or key == ord('Q'):
                save_yolo_format(img_w, img_h, img_name) 
                cv2.destroyAllWindows()
                return

if __name__ == '__main__':
    main()
    