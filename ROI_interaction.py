import cv2
import csv
import os
import datetime
import numpy as np
from collections import deque
from ultralytics import YOLO

# --- CONFIGURATION ---
MODEL_PATH = "C:/weights/best.pt"
VIDEO_PATH = "C:/Users/softlabs_group/Downloads/14571103_3840_2160_60fps.mp4"
OUTPUT_DIR = "C:/resultsTracker"
OUTPUT_VIDEO_PATH = "C:/resultsTracker/inference_output.mp4"
CSV_LOG = "violations_log.csv"

OVERLAP_THRESHOLD = 0.5
VIOLATION_FRAMES = 15

MIN_BIKE_HEIGHT_PX = 80

# --- ROI INITIAL VALUES ---
HORIZON_Y = 400
LEFT_X = 300
RIGHT_X = 1500

dragging = None

os.makedirs(OUTPUT_DIR, exist_ok=True)

if not os.path.exists(CSV_LOG):
    with open(CSV_LOG, mode='w', newline='') as f:
        csv.writer(f).writerow(
            ["Timestamp", "Track_ID", "Violation_Type", "Plate_Image_Path", "Bike_Image_Path"]
        )

# ---------------- ROI INTERACTION ----------------
def mouse_callback(event, x, y, flags, param):
    global HORIZON_Y, LEFT_X, RIGHT_X, dragging

    if event == cv2.EVENT_LBUTTONDOWN:
        if abs(y - HORIZON_Y) < 10:
            dragging = "horizon"
        elif abs(x - LEFT_X) < 10:
            dragging = "left"
        elif abs(x - RIGHT_X) < 10:
            dragging = "right"

    elif event == cv2.EVENT_MOUSEMOVE:
        if dragging == "horizon":
            HORIZON_Y = y
        elif dragging == "left":
            LEFT_X = x
        elif dragging == "right":
            RIGHT_X = x

    elif event == cv2.EVENT_LBUTTONUP:
        dragging = None


def draw_roi(frame):
    h, w = frame.shape[:2]

    # Horizon line
    cv2.line(frame, (0, HORIZON_Y), (w, HORIZON_Y), (0, 255, 255), 2)

    # Left & Right lines
    cv2.line(frame, (LEFT_X, 0), (LEFT_X, h), (255, 0, 0), 2)
    cv2.line(frame, (RIGHT_X, 0), (RIGHT_X, h), (255, 0, 0), 2)


def is_inside_roi(box):
    x1, y1, x2, y2 = box
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)

    return (cy > HORIZON_Y) and (LEFT_X < cx < RIGHT_X)

# ---------------- HELPERS ----------------
def ioa(inner, outer):
    xA, yA = max(inner[0], outer[0]), max(inner[1], outer[1])
    xB, yB = min(inner[2], outer[2]), min(inner[3], outer[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    area = max(1, (inner[2]-inner[0]) * (inner[3]-inner[1]))
    return inter / area

def box_centre(b):
    return ((b[0]+b[2])/2, (b[1]+b[3])/2)

def box_wh(b):
    return b[2]-b[0], b[3]-b[1]

# ---------------- TRACK STATE ----------------
def new_track_state():
    return {
        'helmet_frames': 0,
        'triple_frames': 0,
        'penalized': False,
        'best_plate': None,
        'centroids': deque(maxlen=30),
    }

# ---------------- LOGGING ----------------
def log_violation(frame, track_id, vtype, plate_coords, bike_coords):
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    plate_path = "NOT_FOUND"
    bike_path = "NOT_FOUND"

    if plate_coords is not None:
        x1,y1,x2,y2 = map(int, plate_coords)
        crop = frame[max(0,y1):y2, max(0,x1):x2]
        if crop.size:
            plate_path = f"{OUTPUT_DIR}/{vtype}_{track_id}_{ts}_plate.jpg"
            cv2.imwrite(plate_path, crop)

    if bike_coords is not None:
        x1,y1,x2,y2 = map(int, bike_coords)
        crop = frame[max(0,y1):y2, max(0,x1):x2]
        if crop.size:
            bike_path = f"{OUTPUT_DIR}/{vtype}_{track_id}_{ts}_bike.jpg"
            cv2.imwrite(bike_path, crop)

    with open(CSV_LOG, 'a', newline='') as f:
        csv.writer(f).writerow([ts, track_id, vtype, plate_path, bike_path])

    print(f"🚨 CHALLAN: {vtype} | Bike #{track_id}")

# ---------------- MAIN ----------------
def main():
    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(VIDEO_PATH)

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    out_video = cv2.VideoWriter(
        OUTPUT_VIDEO_PATH,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (W, H)
    )

    track_history = {}

    cv2.namedWindow("Urbanity Tracker")
    cv2.setMouseCallback("Urbanity Tracker", mouse_callback)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(frame, persist=True, tracker="bytetrack.yaml", conf=0.4, verbose=False)
        names = results[0].names

        draw_roi(frame)

        if results[0].boxes.id is None:
            out_video.write(frame)
            cv2.imshow("Urbanity Tracker", cv2.resize(frame, (1024, 576)))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids = results[0].boxes.id.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy()

        bikes_tracked = []
        riders, pillions, no_helmets, plates = [], [], [], []

        for box, tid, cls in zip(boxes, ids, classes):
            cname = names[int(cls)]
            if cname == 'bike_with_rider': bikes_tracked.append((box, int(tid)))
            elif cname == 'Rider': riders.append(box)
            elif cname == 'pillion': pillions.append(box)
            elif cname == 'no_helmet': no_helmets.append(box)
            elif cname == 'number_plate': plates.append(box)

        for bike_box, track_id in bikes_tracked:

            if track_id not in track_history:
                track_history[track_id] = new_track_state()

            state = track_history[track_id]

            # -------- ROI FILTER --------
            if not is_inside_roi(bike_box):
                cv2.putText(frame, f"ID:{track_id} [OUTSIDE ROI]",
                            (int(bike_box[0]), int(bike_box[1]-15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
                continue

            # -------- SIZE FILTER --------
            _, bh = box_wh(bike_box)
            if bh < MIN_BIKE_HEIGHT_PX:
                continue

            if state['penalized']:
                cv2.putText(frame, f"ID:{track_id} (PENALIZED)",
                            (int(bike_box[0]), int(bike_box[1]-15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
                continue

            bike_riders = [r for r in riders if ioa(r, bike_box) > OVERLAP_THRESHOLD]
            bike_pillions = [p for p in pillions if ioa(p, bike_box) > OVERLAP_THRESHOLD]

            for p in plates:
                if ioa(p, bike_box) > OVERLAP_THRESHOLD:
                    state['best_plate'] = p
                    break

            if (len(bike_riders) + len(bike_pillions)) >= 3:
                state['triple_frames'] += 1
            else:
                state['triple_frames'] = 0

            driver_no_helmet = any(ioa(nh, bike_box) > OVERLAP_THRESHOLD for nh in no_helmets)

            if driver_no_helmet:
                state['helmet_frames'] += 1
            else:
                state['helmet_frames'] = 0

            if state['helmet_frames'] >= VIOLATION_FRAMES:
                log_violation(frame, track_id, "NO_HELMET", state['best_plate'], bike_box)
                state['penalized'] = True

            elif state['triple_frames'] >= VIOLATION_FRAMES:
                log_violation(frame, track_id, "TRIPLE_SEAT", state['best_plate'], bike_box)
                state['penalized'] = True

            if not state['penalized']:
                cv2.putText(frame, f"ID:{track_id}",
                            (int(bike_box[0]), int(bike_box[1]-15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        out_video.write(frame)
        cv2.imshow("Urbanity Tracker", cv2.resize(frame, (1024, 576)))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out_video.release()
    cv2.destroyAllWindows()

    print("✅ Done")

if __name__ == "__main__":
    main()
