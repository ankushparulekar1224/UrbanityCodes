import cv2
import os
import math
from ultralytics import YOLO

# --- CONFIGURATION ---
OLD_MODEL_PATH = "C:/newData/best.pt"  
IMAGE_DIR = "C:/dataNewOne/images"
LABEL_DIR = "C:/dataNewOne/labels"
PROGRESS_FILE = 'last_index.txt'

os.makedirs(LABEL_DIR, exist_ok=True)

OLD_TO_NEW_MAP = {
    0: 2,  
    1: 0,  
    2: 4,  
    3: 6,  
    4: 3   
}

CLASSES = {
    0: 'car', 1: 'motorcycle', 2: 'driver', 
    3: 'pillion', 4: 'helmet', 5: 'no_helmet', 6: 'number_plate'
}

COLORS = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (0,255,255), (255,0,255), (255,255,255)]

# --- GLOBAL STATE ---
boxes = []          
current_class = 1   
start_pt = (-1, -1)
current_pt = (-1, -1)

# Drag & Drop State
selected_idx = -1
action = None # 'draw', 'move', 'resize_tl', 'resize_br', 'resize_tr', 'resize_bl'
drag_offset_x = 0
drag_offset_y = 0

def get_action(x, y, boxes_list):
    """Determines what the user is trying to click on (corner vs inside vs empty space)."""
    TOLERANCE = 10 # Pixels to grab a corner
    
    # Check corners first (highest priority)
    for i, b in enumerate(boxes_list):
        cid, x1, y1, x2, y2 = b
        if math.hypot(x-x1, y-y1) < TOLERANCE: return i, 'resize_tl'
        if math.hypot(x-x2, y-y2) < TOLERANCE: return i, 'resize_br'
        if math.hypot(x-x2, y-y1) < TOLERANCE: return i, 'resize_tr'
        if math.hypot(x-x1, y-y2) < TOLERANCE: return i, 'resize_bl'
        
    # Check inside boxes (moving)
    for i, b in enumerate(boxes_list):
        cid, x1, y1, x2, y2 = b
        if min(x1, x2) < x < max(x1, x2) and min(y1, y2) < y < max(y1, y2): 
            return i, 'move'
            
    return -1, 'draw'

def mouse_callback(event, x, y, flags, param):
    global start_pt, current_pt, boxes, current_class
    global selected_idx, action, drag_offset_x, drag_offset_y
    
    # 1. Mouse Button Down
    if event == cv2.EVENT_LBUTTONDOWN:
        selected_idx, action = get_action(x, y, boxes)
        if action == 'draw':
            start_pt = (x, y)
            current_pt = (x, y)
        elif action == 'move':
            cid, x1, y1, x2, y2 = boxes[selected_idx]
            drag_offset_x = x - min(x1, x2)
            drag_offset_y = y - min(y1, y2)
            
    # 2. Mouse Movement
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
            
    # 3. Mouse Button Release
    elif event == cv2.EVENT_LBUTTONUP:
        if action == 'draw':
            x_min, x_max = min(start_pt[0], x), max(start_pt[0], x)
            y_min, y_max = min(start_pt[1], y), max(start_pt[1], y)
            if x_max - x_min > 5 and y_max - y_min > 5:
                boxes.append([current_class, x_min, y_min, x_max, y_max])
        elif action and action.startswith('resize') and selected_idx != -1:
            # Fix coordinates if user dragged a corner inside-out
            cid, x1, y1, x2, y2 = boxes[selected_idx]
            boxes[selected_idx] = [cid, min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
        action = None
        
    # 4. Right Click Delete
    elif event == cv2.EVENT_RBUTTONDOWN:
        idx, act = get_action(x, y, boxes)
        if act == 'move' or (act and act.startswith('resize')):
            boxes.pop(idx)

def save_yolo_format(img_width, img_height, filename):
    filepath = os.path.join(LABEL_DIR, filename.replace('.jpg', '.txt').replace('.png', '.txt'))
    with open(filepath, 'w') as f: 
        for b in boxes:
            cid, x1, y1, x2, y2 = b
            cx = ((x1 + x2) / 2) / img_width
            cy = ((y1 + y2) / 2) / img_height
            w = abs(x2 - x1) / img_width
            h = abs(y2 - y1) / img_height
            f.write(f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

def main():
    global boxes, current_class
    model = YOLO(OLD_MODEL_PATH)
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
    while idx < len(image_files):
        img_name = image_files[idx]
        img_path = os.path.join(IMAGE_DIR, img_name)
        frame = cv2.imread(img_path)
        img_h, img_w = frame.shape[:2]
        
        boxes = []
        results = model.predict(img_path, conf=0.4, verbose=False)
        for r in results:
            for box in r.boxes:
                old_cls = int(box.cls[0])
                if old_cls in OLD_TO_NEW_MAP:
                    new_cls = OLD_TO_NEW_MAP[old_cls]
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    boxes.append([new_cls, x1, y1, x2, y2])
        
        while True:
            display = frame.copy()
            
            for b in boxes:
                cid, x1, y1, x2, y2 = b
                color = COLORS[cid % len(COLORS)]
                cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
                cv2.putText(display, CLASSES[cid], (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Draw corner handles for easy grabbing
                cv2.circle(display, (x1, y1), 5, color, -1)
                cv2.circle(display, (x2, y2), 5, color, -1)
                cv2.circle(display, (x2, y1), 5, color, -1)
                cv2.circle(display, (x1, y2), 5, color, -1)
                
            if action == 'draw':
                cv2.rectangle(display, start_pt, current_pt, COLORS[current_class % len(COLORS)], 2)
                
            ui_text = f"Img: {idx}/{len(image_files)} | Class ({current_class}): {CLASSES[current_class]}"
            cv2.putText(display, ui_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(display, "0-6: Class | L-Click: Drag/Resize/Draw | R-Click: Delete | N: Save/Next | Q: Quit", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            cv2.imshow('Urbanity Auto-Labeler', display)
            key = cv2.waitKey(1) & 0xFF
            
            if ord('0') <= key <= ord('6'): current_class = key - ord('0')
            elif key == ord('n'): 
                save_yolo_format(img_w, img_h, img_name)
                with open(PROGRESS_FILE, 'w') as f: f.write(str(idx + 1))
                idx += 1
                break
            elif key == ord('q'): 
                cv2.destroyAllWindows()
                return

if __name__ == '__main__':
    main()