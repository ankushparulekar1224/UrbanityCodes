import cv2
import csv
import os
import datetime
from ultralytics import YOLO

MODEL_PATH = r"c:\Users\thero\Downloads\ubnitymodel3.pt"  
VIDEO_PATH = r"c:\Users\thero\Downloads\NO20260328-173322-000094F.MP4"
OUTPUT_DIR = r"C:/resultsTracker"
OUTPUT_VIDEO_PATH = r"C:/resultsTracker/inference_output.mp4" 
CSV_LOG = r"C:/resultsTracker/violations_log.csv"
OVERLAP_THRESHOLD = 0.8 
PLATE_OVERLAP_THRESHOLD = 0.4 # Lower threshold specifically for plates
VIOLATION_FRAMES = 15 

# --- FOLDER SETUP ---
BIKE_DIR = os.path.join(OUTPUT_DIR, "bikes")
PLATE_DIR = os.path.join(OUTPUT_DIR, "plates")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(BIKE_DIR, exist_ok=True)   # Creates the bikes folder
os.makedirs(PLATE_DIR, exist_ok=True)  # Creates the plates folder

# Initialize CSV Log
if not os.path.exists(CSV_LOG):
    with open(CSV_LOG, mode='w', newline='') as f:
        writer = csv.writer(f)
        # Added Bike_Image_Path column
        writer.writerow(["Timestamp", "Track_ID", "Violation_Type", "Plate_Image_Path", "Bike_Image_Path"])

# In-Memory State Tracker
# Format: {track_id: {'helmet_frames': 0, 'triple_frames': 0, 'penalized': False, 'best_plate': [x1,y1,x2,y2]}}
track_history = {}

def calculate_intersection_area(boxA, boxB):
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    return max(0, xB - xA) * max(0, yB - yA)

def get_ioa(inner_box, outer_box):
    interArea = calculate_intersection_area(inner_box, outer_box)
    innerArea = (inner_box[2] - inner_box[0]) * (inner_box[3] - inner_box[1])
    return 0.0 if innerArea == 0 else interArea / float(innerArea)

def log_violation(frame, track_id, violation_type, plate_coords, bike_coords):
    """Saves the plate image, the full bike image, and logs to CSV."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    plate_img_path = "NO_PLATE_DETECTED"
    bike_img_path = "NO_BIKE_DETECTED"

    # 1. Crop and save the number plate to the 'plates' folder
    if plate_coords is not None:
        px1, py1, px2, py2 = map(int, plate_coords)
        px1, py1 = max(0, px1), max(0, py1)
        plate_crop = frame[py1:py2, px1:px2]
        
        if plate_crop.size != 0:
            # Save to PLATE_DIR
            plate_img_path = f"{PLATE_DIR}/{violation_type}_ID{track_id}_{timestamp}_plate.jpg"
            cv2.imwrite(plate_img_path, plate_crop)

    # 2. Crop and save the full penalized bike to the 'bikes' folder
    if bike_coords is not None:
        bx1, by1, bx2, by2 = map(int, bike_coords)
        bx1, by1 = max(0, bx1), max(0, by1)
        bike_crop = frame[by1:by2, bx1:bx2]
        
        if bike_crop.size != 0:
            # Save to BIKE_DIR
            bike_img_path = f"{BIKE_DIR}/{violation_type}_ID{track_id}_{timestamp}_bike.jpg"
            cv2.imwrite(bike_img_path, bike_crop)

    # Write to database/CSV
    with open(CSV_LOG, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, track_id, violation_type, plate_img_path, bike_img_path])

    print(f"🚨 TICKET ISSUED: {violation_type} on Bike #{track_id}. Images saved.")

def main():
    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(VIDEO_PATH)

    # --- SETUP VIDEO WRITER ---
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec for .mp4 files
    out_video = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # Run ByteTrack
        results = model.track(frame, persist=True, tracker="bytetrack.yaml", conf=0.4, verbose=False)
        names = results[0].names

        # --- 1. VISUALIZE ALL DETECTED BOXES FIRST ---
        for raw_box in results[0].boxes:
            x1, y1, x2, y2 = map(int, raw_box.xyxy[0].cpu().numpy())
            cls_name = names[int(raw_box.cls[0])]

            # Draw thin white boxes and text for every object detected
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 1)
            cv2.putText(frame, cls_name, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # --- 2. ENSURE TRACKER IS INITIALIZED ---
        if results[0].boxes.id is None:
            # Write the frame with just base boxes to the output video
            out_video.write(frame)
            cv2.imshow("Urbanity Tracker Test", cv2.resize(frame, (1024, 768)))
            if cv2.waitKey(1) & 0xFF == ord('q'): break
            continue

        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids = results[0].boxes.id.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy()

        # Separate detections
        bikes_tracked = []
        riders, pillions, no_helmets, helmets, plates = [], [], [], [], [] # Added helmets

        for box, track_id, cls in zip(boxes, ids, classes):
            cls_name = names[int(cls)]
            if cls_name == 'bike_with_rider': bikes_tracked.append((box, int(track_id)))
            elif cls_name == 'rider': riders.append(box)
            elif cls_name == 'pillion': pillions.append(box)
            elif cls_name == 'no_helmet': no_helmets.append(box)
            elif cls_name == 'helmet': helmets.append(box) # ADD THIS LINE
            elif cls_name == 'number_plate': plates.append(box)

        # --- 3. PROCESS EACH TRACKED BIKE ---
        for bike_box, track_id in bikes_tracked:

            if track_id not in track_history:
                track_history[track_id] = {'helmet_frames': 0, 'triple_frames': 0, 'penalized': False, 'best_plate': None}

            if track_history[track_id]['penalized']:
                cv2.putText(frame, f"ID:{track_id} (PENALIZED)", (int(bike_box[0]), int(bike_box[1])-15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                cv2.rectangle(frame, (int(bike_box[0]), int(bike_box[1])), (int(bike_box[2]), int(bike_box[3])), (0, 0, 255), 2)
                continue

            bike_riders = [r for r in riders if get_ioa(r, bike_box) > OVERLAP_THRESHOLD]
            bike_pillions = [p for p in pillions if get_ioa(p, bike_box) > OVERLAP_THRESHOLD]

            for p in plates:
                if get_ioa(p, bike_box) > PLATE_OVERLAP_THRESHOLD:
                    track_history[track_id]['best_plate'] = p
                    break

            if (len(bike_riders) + len(bike_pillions)) >= 3:
                track_history[track_id]['triple_frames'] += 1
            else:
                track_history[track_id]['triple_frames'] = 0 

            driver_no_helmet_detected = False
            driver_helmet_detected = False

            # --- STEP A: Check if driver HAS a helmet (The Veto) ---
            for h_box in helmets:
                if get_ioa(h_box, bike_box) > OVERLAP_THRESHOLD:
                    for body in bike_riders:
                        if get_ioa(h_box, body) > OVERLAP_THRESHOLD:
                            # Spatial Check: Helmet must be in the upper 40% of the rider's body
                            head_center_y = (h_box[1] + h_box[3]) / 2.0
                            body_y1, body_y2 = body[1], body[3]
                            body_top40_y = body_y1 + ((body_y2 - body_y1) * 0.40) # Calculate the 40% line

                            if head_center_y < body_top40_y:
                                driver_helmet_detected = True
                                break
                if driver_helmet_detected:
                    break

            # --- STEP B: Check for NO helmet (Only if Veto is not active) ---
            if not driver_helmet_detected:
                for nh_box in no_helmets:
                    if get_ioa(nh_box, bike_box) > OVERLAP_THRESHOLD:
                        best_match_body = None
                        highest_ioa = 0.0
                        is_driver_head = False

                        for body in (bike_riders + bike_pillions):
                            overlap_ratio = get_ioa(nh_box, body)
                            if overlap_ratio > highest_ioa and overlap_ratio > OVERLAP_THRESHOLD:

                                # Spatial Check: No-Helmet must be in the upper 40% of the body
                                head_center_y = (nh_box[1] + nh_box[3]) / 2.0
                                body_y1, body_y2 = body[1], body[3]
                                body_top40_y = body_y1 + ((body_y2 - body_y1) * 0.40) # Calculate the 40% line

                                if head_center_y < body_top40_y:
                                    highest_ioa = overlap_ratio
                                    best_match_body = body
                                    is_driver_head = any((body == r).all() for r in bike_riders)

                        if best_match_body is not None and is_driver_head:
                            driver_no_helmet_detected = True
                            break
            if driver_no_helmet_detected:
                track_history[track_id]['helmet_frames'] += 1
            else:
                track_history[track_id]['helmet_frames'] = 0

            # --- THE TRIGGER (15 FRAMES) ---
            if track_history[track_id]['helmet_frames'] >= VIOLATION_FRAMES:
                log_violation(frame, track_id, "NO_HELMET", track_history[track_id]['best_plate'], bike_box)
                track_history[track_id]['penalized'] = True

            elif track_history[track_id]['triple_frames'] >= VIOLATION_FRAMES:
                log_violation(frame, track_id, "TRIPLE_SEAT", track_history[track_id]['best_plate'], bike_box)
                track_history[track_id]['penalized'] = True

            if not track_history[track_id]['penalized']:
                cv2.putText(frame, f"ID:{track_id}", (int(bike_box[0]), int(bike_box[1])-15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # --- WRITE FRAME TO VIDEO ---
        out_video.write(frame)

        cv2.imshow("Urbanity Tracker Test", cv2.resize(frame, (1024, 768)))
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    # Clean up resources
    cap.release()
    out_video.release() # Very important to finalize the video file
    cv2.destroyAllWindows()
    print(f"Inference complete. Video saved to: {OUTPUT_VIDEO_PATH}")

if __name__ == "__main__":                                                                    
    main()