import cv2
import os
from ultralytics import YOLO

# --- CONFIGURATION ---
MODEL_PATH = r"c:\Users\thero\Downloads\ubnitymodel3.pt"  
IMAGE_PATH = r"c:\resultsTracker\NO_HELMET_ID609_20260508_173812_bike.jpg" # Update this to your test image
OUTPUT_IMAGE_PATH = "C:\\resultsTracker3\\inference_output.jpg"
OVERLAP_THRESHOLD = 0.8 

os.makedirs(os.path.dirname(OUTPUT_IMAGE_PATH), exist_ok=True)

def calculate_intersection_area(boxA, boxB):
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    return max(0, xB - xA) * max(0, yB - yA)

def get_ioa(inner_box, outer_box):
    interArea = calculate_intersection_area(inner_box, outer_box)
    innerArea = (inner_box[2] - inner_box[0]) * (inner_box[3] - inner_box[1])
    return 0.0 if innerArea == 0 else interArea / float(innerArea)

def main():
    model = YOLO(MODEL_PATH)
    frame = cv2.imread(IMAGE_PATH)
    
    if frame is None:
        print(f"Error: Could not read image at {IMAGE_PATH}")
        return

    # Run YOLO on the single image (No tracker needed)
    results = model(frame, conf=0.4, verbose=False)
    names = results[0].names

    # Extract boxes and classes
    boxes = results[0].boxes.xyxy.cpu().numpy()
    classes = results[0].boxes.cls.cpu().numpy()

    # --- 1. VISUALIZE ALL RAW DETECTIONS (Optional: helps see what the model sees) ---
    for raw_box, cls in zip(boxes, classes):
        x1, y1, x2, y2 = map(int, raw_box)
        cls_name = names[int(cls)]
        # Draw thin white boxes for base detections
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 1)
        cv2.putText(frame, cls_name, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # Separate detections
    bikes = []
    riders, pillions, no_helmets, helmets, plates = [], [], [], [], []

    for box, cls in zip(boxes, classes):
        cls_name = names[int(cls)]
        if cls_name == 'bike_with_rider': bikes.append(box)
        elif cls_name == 'rider': riders.append(box)
        elif cls_name == 'pillion': pillions.append(box)
        elif cls_name == 'no_helmet': no_helmets.append(box)
        elif cls_name == 'helmet': helmets.append(box)
        elif cls_name == 'number_plate': plates.append(box)

    # --- 2. PROCESS EACH BIKE ---
    bike_counter = 1
    for bike_box in bikes:
        bike_riders = [r for r in riders if get_ioa(r, bike_box) > OVERLAP_THRESHOLD]
        bike_pillions = [p for p in pillions if get_ioa(p, bike_box) > OVERLAP_THRESHOLD]
        
        # Determine Triple Seat
        is_triple_seat = (len(bike_riders) + len(bike_pillions)) >= 3

        # Determine No Helmet (Using the 40% Veto Logic)
        driver_no_helmet_detected = False
        driver_helmet_detected = False

        # STEP A: The Helmet Veto
        for h_box in helmets:
            if get_ioa(h_box, bike_box) > OVERLAP_THRESHOLD:
                for body in bike_riders:
                    if get_ioa(h_box, body) > OVERLAP_THRESHOLD:
                        head_center_y = (h_box[1] + h_box[3]) / 2.0
                        body_top40_y = body[1] + ((body[3] - body[1]) * 0.40)
                        if head_center_y < body_top40_y:
                            driver_helmet_detected = True
                            break
            if driver_helmet_detected: break

        # STEP B: The No-Helmet Check
        if not driver_helmet_detected:
            for nh_box in no_helmets:
                if get_ioa(nh_box, bike_box) > OVERLAP_THRESHOLD:
                    best_match_body = None
                    highest_ioa = 0.0
                    is_driver_head = False

                    for body in (bike_riders + bike_pillions):
                        overlap_ratio = get_ioa(nh_box, body)
                        if overlap_ratio > highest_ioa and overlap_ratio > OVERLAP_THRESHOLD:
                            head_center_y = (nh_box[1] + nh_box[3]) / 2.0
                            body_top40_y = body[1] + ((body[3] - body[1]) * 0.40)
                            
                            if head_center_y < body_top40_y:
                                highest_ioa = overlap_ratio
                                best_match_body = body
                                is_driver_head = any((body == r).all() for r in bike_riders)

                    if best_match_body is not None and is_driver_head:
                        driver_no_helmet_detected = True
                        break

        # --- 3. VISUAL DRAWING ---
        active_penalties = []
        if driver_no_helmet_detected: active_penalties.append("NO HELMET")
        if is_triple_seat: active_penalties.append("TRIPLE")

        bx1, by1, bx2, by2 = map(int, bike_box)

        if active_penalties:
            # Draw Red box for violations
            penalty_text = " + ".join(active_penalties)
            cv2.putText(frame, f"Bike {bike_counter} ({penalty_text})", (bx1, by1 - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 0, 255), 2)
        else:
            # Draw Green box if okay
            cv2.putText(frame, f"Bike {bike_counter} (OK)", (bx1, by1 - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
            
        bike_counter += 1

    # --- SAVE AND DISPLAY ---
    cv2.imwrite(OUTPUT_IMAGE_PATH, frame)
    print(f"Image saved to: {OUTPUT_IMAGE_PATH}")
    
    cv2.imshow("Urbanity Image Test", cv2.resize(frame, (1024, 768)))
    cv2.waitKey(0) # Waits indefinitely until a key is pressed
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()