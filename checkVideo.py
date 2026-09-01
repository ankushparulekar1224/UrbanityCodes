import cv2
from ultralytics import YOLO

# --- CONFIGURATION ---
MODEL_PATH = "C:/weights/best.pt"  
VIDEO_PATH = "C:/client_data/videos/VID20260215145537~2.mp4" 
OUTPUT_PATH = "C:/client_data/videos/output_logic_test.mp4"
OVERLAP_THRESHOLD = 0.5  # At least 50% of the inner box must be inside the outer box

def calculate_intersection_area(boxA, boxB):
    """Calculates the area of intersection between two bounding boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Compute the area of intersection. If they don't overlap, area is 0.
    interArea = max(0, xB - xA) * max(0, yB - yA)
    return interArea

def get_ioa(inner_box, outer_box):
    """Calculates Intersection over Area of the inner (smaller) box."""
    interArea = calculate_intersection_area(inner_box, outer_box)
    
    # Calculate the area of the inner box
    innerArea = (inner_box[2] - inner_box[0]) * (inner_box[3] - inner_box[1])
    
    if innerArea == 0:
        return 0.0
    
    # Return the percentage of the inner box that is inside the outer box
    return interArea / float(innerArea)

def main():
    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(VIDEO_PATH)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    out = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        results = model(frame, conf=0.4, verbose=False)
        detections = {name: [] for name in results[0].names.values()}
        
        for box in results[0].boxes:
            cls_name = results[0].names[int(box.cls[0])]
            detections[cls_name].append(box.xyxy[0].cpu().numpy())

        # Draw Base Boxes (Optional)
        for name, boxes in detections.items():
            for b in boxes:
                cv2.rectangle(frame, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (255, 255, 255), 1)

        # --- THE CORE OVERLAP LOGIC ---
        for bike in detections.get('bike_with_rider', []):
            bike_riders = []
            bike_pillions = []
            
            # 1. Assign Riders and Pillions to the Bike using IoA
            for rider in detections.get('Rider', []):
                if get_ioa(rider, bike) > OVERLAP_THRESHOLD:
                    bike_riders.append(rider)
                    
            for pillion in detections.get('pillion', []):
                if get_ioa(pillion, bike) > OVERLAP_THRESHOLD:
                    bike_pillions.append(pillion)

            # --- TRIPLE SEAT CHECK ---
            if (len(bike_riders) + len(bike_pillions)) >= 3:
                cv2.putText(frame, "TRIPLE SEAT!", (int(bike[0]), int(bike[1]) - 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
                cv2.rectangle(frame, (int(bike[0]), int(bike[1])), (int(bike[2]), int(bike[3])), (0, 165, 255), 3)

            # --- HELMET CHECK WITH PILLION OVERLAP FIX ---
            for nh_box in detections.get('no_helmet', []):
                # First, ensure the head actually belongs to this specific bike
                if get_ioa(nh_box, bike) > OVERLAP_THRESHOLD:
                    
                    best_match_body = None
                    highest_ioa = 0.0
                    is_driver_head = False
                    
                    # Test the head against ALL bodies on this bike to see where it fits best
                    all_bodies = bike_riders + bike_pillions
                    
                    for body in all_bodies:
                        overlap_ratio = get_ioa(nh_box, body)
                        
                        # Find the body that contains the LARGEST percentage of this head
                        if overlap_ratio > highest_ioa and overlap_ratio > OVERLAP_THRESHOLD:
                            highest_ioa = overlap_ratio
                            best_match_body = body
                            # Flag true if the winning body is in the Rider array
                            is_driver_head = any((body == r).all() for r in bike_riders)

                    # 3. Penalize ONLY if the head best matched the driver
                    if best_match_body is not None and is_driver_head:
                        cv2.putText(frame, "DRIVER NO HELMET!", (int(bike[0]), int(bike[1]) - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                        cv2.rectangle(frame, (int(bike[0]), int(bike[1])), (int(bike[2]), int(bike[3])), (0, 0, 255), 3)

        out.write(frame)
        cv2.imshow("Urbanity Logic Test", cv2.resize(frame, (1024, 768)))
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()