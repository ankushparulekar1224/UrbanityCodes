import cv2
import os
import time

# --- CONFIGURATION ---
IMAGE_PATH = "C:/dataNewOne/images/1f8f0a23-8a5e183d-frame_00017.jpg" # Change to your image
LABEL_PATH = "C:/dataNewOne/labels/1f8f0a23-8a5e183d-frame_00017.txt" # Change to your corresponding label

CLASSES = {
    0: 'Rider', 1: 'car', 2: 'no_helmet', 3: 'helmet', 
    4: 'number_plate', 5: 'pillion', 6: 'bike_with_rider'
}

def calculate_ioa(child_box, parent_box):
    """Calculates what percentage of the child box is inside the parent box."""
    cx1, cy1, cx2, cy2 = child_box
    px1, py1, px2, py2 = parent_box

    ix1, iy1 = max(cx1, px1), max(cy1, py1)
    ix2, iy2 = min(cx2, px2), min(cy2, py2)

    inter_area = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    child_area = max(0, cx2 - cx1) * max(0, cy2 - cy1)

    if child_area == 0: return 0
    return inter_area / child_area

def read_yolo_labels(label_path, img_w, img_h):
    """Reads YOLO txt file and converts to [class_id, x1, y1, x2, y2]."""
    boxes = []
    if not os.path.exists(label_path):
        print(f"Label file not found: {label_path}")
        return boxes
        
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                cid = int(parts[0])
                cx, cy, w, h = map(float, parts[1:5])
                x1 = int((cx - w/2) * img_w)
                y1 = int((cy - h/2) * img_h)
                x2 = int((cx + w/2) * img_w)
                y2 = int((cy + h/2) * img_h)
                boxes.append([cid, x1, y1, x2, y2])
    return boxes

def main():
    # 1. Load Image
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        print(f"Could not load image: {IMAGE_PATH}")
        return
    img_h, img_w = img.shape[:2]

    # 2. Load Labels
    all_boxes = read_yolo_labels(LABEL_PATH, img_w, img_h)
    
    # 3. Separate boxes by class
    bikes = [b[1:] for b in all_boxes if b[0] == 6]
    riders = [b[1:] for b in all_boxes if b[0] == 0]
    pillions = [b[1:] for b in all_boxes if b[0] == 5]
    no_helmets = [b[1:] for b in all_boxes if b[0] == 2]

    # 4. Process each bike
    # 4. Process each bike
    for idx, b_box in enumerate(bikes, 1):
        start_time = time.perf_counter() # <-- START TIMER
        
        bx1, by1, bx2, by2 = b_box
        
        bike_riders = []
        people_count = 0
        rider_missing_helmet = False
        
        # Check Riders (>60% IoA)
        for r_box in riders:
            if calculate_ioa(r_box, b_box) > 0.60:
                bike_riders.append(r_box)
                people_count += 1
                cv2.rectangle(img, (r_box[0], r_box[1]), (r_box[2], r_box[3]), (255, 165, 0), 2) # Orange
                
        # Check Pillions (>60% IoA)
        for p_box in pillions:
            if calculate_ioa(p_box, b_box) > 0.60:
                people_count += 1
                cv2.rectangle(img, (p_box[0], p_box[1]), (p_box[2], p_box[3]), (255, 255, 0), 2) # Cyan

        triple_seat_violation = (people_count > 2)

        # Check Helmet Logic for the assigned Rider(s)
        for r_box in bike_riders:
            rx1, ry1, rx2, ry2 = r_box
            rider_height = ry2 - ry1
            rider_top_third_y = ry1 + (rider_height * 0.35)
            
            for h_box in no_helmets:
                hx1, hy1, hx2, hy2 = h_box
                hc_y = (hy1 + hy2) / 2
                
                # Check IoA > 70% and Center Y in top 35% of Rider body
                if calculate_ioa(h_box, r_box) > 0.70 and hc_y < rider_top_third_y:
                    rider_missing_helmet = True
                    cv2.rectangle(img, (hx1, hy1), (hx2, hy2), (0, 0, 255), 2) # Red for bare head
                    break

        
        # 5. Visualization and Penalty Outcome
        bike_color = (0, 255, 0) # Green by default
        violation_text = []
        
        if triple_seat_violation:
            violation_text.append(f"Triple Seat ({people_count})")
            bike_color = (0, 0, 255) # Red
            
        if rider_missing_helmet:
            violation_text.append("No Helmet")
            bike_color = (0, 0, 255) # Red

        # Draw Bike Box
        cv2.rectangle(img, (bx1, by1), (bx2, by2), bike_color, 3)
        
        # Put Status Text above Bike (NOW INCLUDES THE ID)
        if violation_text:
            text = f"ID: {idx} | VIOLATION: " + " | ".join(violation_text)
            cv2.putText(img, text, (bx1, by1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            text = f"ID: {idx} | SAFE"
            cv2.putText(img, text, (bx1, by1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        end_time = time.perf_counter() # <-- END TIMER
        
        # Calculate time in milliseconds and print it
        process_time_ms = (end_time - start_time) * 1000
        print(f"Bike ID: {idx} processed in: {process_time_ms:.4f} ms")

    # Show the result
    cv2.namedWindow("Penalty Logic Visualizer", cv2.WINDOW_NORMAL) # Allows resizing
    cv2.resizeWindow("Penalty Logic Visualizer", 1280, 720)        # Sets a default size (e.g., 720p)
    cv2.imshow("Penalty Logic Visualizer", img)
    
    print("Press 'q' or ESC to close the window.")
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
            
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()