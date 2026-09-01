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

# ── Tunable thresholds ─────────────────────────────────────────────────────────
OVERLAP_THRESHOLD   = 0.5
VIOLATION_FRAMES    = 15          # consecutive frames before challan

# Filter 1 – scale gate
MIN_BIKE_HEIGHT_PX  = 80          # ignore bikes shorter than this (far-away / tiny)

# Filter 2 – road zone gate
HORIZON_FRAC        = 0.40        # ignore anything whose centre_y < this fraction of frame height

# Filter 3 – motion gate (ego-motion corrected)
MOTION_FLOW_THRESH  = 1.5         # avg residual optical-flow magnitude to count as "moving"
MOTION_HISTORY      = 8           # frames in the sliding window

# Filter 4 – trajectory gate
TRAJ_WINDOW         = 30          # track centroid over this many frames
TRAJ_DISP_THRESH    = 20          # net pixel displacement required to be "moving"

# Filter 5 – rider association extras
MAX_RIDER_ABOVE_BIKE_FRAC = 0.55  # rider centre_y must be within top 55 % of bike box
MAX_RIDER_WIDTH_FRAC      = 1.2   # rider width must be ≤ 1.2× bike width

os.makedirs(OUTPUT_DIR, exist_ok=True)

if not os.path.exists(CSV_LOG):
    with open(CSV_LOG, mode='w', newline='') as f:
        csv.writer(f).writerow(
            ["Timestamp", "Track_ID", "Violation_Type", "Plate_Image_Path", "Bike_Image_Path"]
        )

# ── Geometry helpers ────────────────────────────────────────────────────────────
def ioa(inner, outer):
    """Intersection-over-area of inner box."""
    xA, yA = max(inner[0], outer[0]), max(inner[1], outer[1])
    xB, yB = min(inner[2], outer[2]), min(inner[3], outer[3])
    inter  = max(0, xB - xA) * max(0, yB - yA)
    area   = max(1, (inner[2]-inner[0]) * (inner[3]-inner[1]))
    return inter / area

def box_centre(b):
    return ((b[0]+b[2])/2, (b[1]+b[3])/2)

def box_wh(b):
    return b[2]-b[0], b[3]-b[1]

# ── Logging helper ──────────────────────────────────────────────────────────────
def log_violation(frame, track_id, vtype, plate_coords, bike_coords):
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    plate_path = bike_path = "NOT_DETECTED"

    if plate_coords is not None:
        x1,y1,x2,y2 = map(int, plate_coords)
        crop = frame[max(0,y1):y2, max(0,x1):x2]
        if crop.size:
            plate_path = f"{OUTPUT_DIR}/{vtype}_ID{track_id}_{ts}_plate.jpg"
            cv2.imwrite(plate_path, crop)

    if bike_coords is not None:
        x1,y1,x2,y2 = map(int, bike_coords)
        crop = frame[max(0,y1):y2, max(0,x1):x2]
        if crop.size:
            bike_path = f"{OUTPUT_DIR}/{vtype}_ID{track_id}_{ts}_bike.jpg"
            cv2.imwrite(bike_path, crop)

    with open(CSV_LOG, 'a', newline='') as f:
        csv.writer(f).writerow([ts, track_id, vtype, plate_path, bike_path])

    print(f"🚨 CHALLAN: {vtype} | Bike #{track_id}")

# ── Ego-motion estimation (sparse optical flow on background points) ───────────
class EgoMotionEstimator:
    """
    Tracks background feature points to estimate how much the camera itself
    moved between frames.  The residual motion of any foreground object
    (after subtracting ego-motion) tells us whether it is truly moving.
    """
    def __init__(self):
        self.prev_gray   = None
        self.prev_pts    = None
        self.lk_params   = dict(winSize=(15,15), maxLevel=2,
                                criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        self.feat_params = dict(maxCorners=200, qualityLevel=0.01,
                                minDistance=20, blockSize=7)

    def update(self, gray):
        if self.prev_gray is None:
            self.prev_gray = gray
            self.prev_pts  = cv2.goodFeaturesToTrack(gray, mask=None, **self.feat_params)
            return np.array([0.0, 0.0])   # zero motion on first frame

        if self.prev_pts is None or len(self.prev_pts) < 10:
            self.prev_pts = cv2.goodFeaturesToTrack(self.prev_gray, mask=None, **self.feat_params)

        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, self.prev_pts, None, **self.lk_params)

        good_new = next_pts[status==1]
        good_old = self.prev_pts[status==1]

        ego = np.array([0.0, 0.0])
        if len(good_new) >= 4:
            diffs = good_new - good_old
            # median is robust against moving foreground objects
            ego = np.median(diffs, axis=0)

        self.prev_gray = gray
        self.prev_pts  = good_new.reshape(-1,1,2) if len(good_new) > 4 else None
        return ego   # (dx, dy) pixels


# ── Per-track state ─────────────────────────────────────────────────────────────
def new_track_state():
    return {
        'helmet_frames':  0,
        'triple_frames':  0,
        'penalized':      False,
        'best_plate':     None,
        # For Filter 4 – trajectory
        'centroids':      deque(maxlen=TRAJ_WINDOW),
        # For Filter 3 – motion (residual flow inside bike box)
        'motion_scores':  deque(maxlen=MOTION_HISTORY),
    }


def compute_residual_flow(flow_map, bike_box, ego):
    """
    Mean magnitude of (optical flow − ego_motion) inside the bike bounding box.
    flow_map: dense flow from cv2.calcOpticalFlowFarneback (H×W×2)
    """
    x1,y1,x2,y2 = map(int, bike_box)
    x1,y1 = max(0,x1), max(0,y1)
    roi = flow_map[y1:y2, x1:x2]
    if roi.size == 0:
        return 0.0
    residual = roi - ego.reshape(1,1,2)
    mag = np.sqrt(residual[...,0]**2 + residual[...,1]**2)
    return float(np.mean(mag))


def is_valid_rider(rider_box, bike_box):
    """
    Filter 5 extras: vertical position and width sanity checks.
    The rider's body should be in the upper-to-mid portion of the bike box
    and should not be wider than the bike.
    """
    bx1,by1,bx2,by2 = bike_box
    rx1,ry1,rx2,ry2 = rider_box
    bike_h = by2 - by1
    bike_w = bx2 - bx1
    rider_w = rx2 - rx1
    rider_cy = (ry1 + ry2) / 2

    # Centre of rider must be within top fraction of bike
    if rider_cy > by1 + bike_h * MAX_RIDER_ABOVE_BIKE_FRAC:
        return False
    # Rider must not be much wider than bike
    if rider_w > bike_w * MAX_RIDER_WIDTH_FRAC:
        return False
    return True


# ── Main ────────────────────────────────────────────────────────────────────────
def main():
    model = YOLO(MODEL_PATH)
    cap   = cv2.VideoCapture(VIDEO_PATH)

    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    HORIZON_Y = int(H * HORIZON_FRAC)   # Filter 2: road zone boundary

    out_video = cv2.VideoWriter(
        OUTPUT_VIDEO_PATH,
        cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H)
    )

    ego_est   = EgoMotionEstimator()
    prev_gray = None                     # for dense flow (Filter 3)
    track_history = {}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ── Ego-motion (sparse flow) ─────────────────────────────────────────
        ego_vec = ego_est.update(gray)

        # ── Dense optical flow for per-box motion score ──────────────────────
        flow_map = None
        if prev_gray is not None:
            flow_map = cv2.calcOpticalFlowFarneback(
                prev_gray, gray, None,
                0.5, 3, 15, 3, 5, 1.2, 0
            )
        prev_gray = gray

        # ── YOLO + BotSort ─────────────────────────────────────────────────
        results = model.track(frame, persist=True, tracker="botsort.yaml",
                              conf=0.4, verbose=False)
        names = results[0].names

        # Draw thin base boxes
        for b in results[0].boxes:
            x1,y1,x2,y2 = map(int, b.xyxy[0].cpu().numpy())
            cv2.rectangle(frame, (x1,y1), (x2,y2), (255,255,255), 1)
            cv2.putText(frame, names[int(b.cls[0])],
                        (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)

        # Draw horizon guide line (visual debug)
        cv2.line(frame, (0, HORIZON_Y), (W, HORIZON_Y), (0,200,255), 1)

        if results[0].boxes.id is None:
            out_video.write(frame)
            cv2.imshow("Urbanity Tracker", cv2.resize(frame, (1024, 576)))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        boxes   = results[0].boxes.xyxy.cpu().numpy()
        ids     = results[0].boxes.id.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy()

        bikes_tracked = []
        riders, pillions, no_helmets, plates = [], [], [], []

        for box, tid, cls in zip(boxes, ids, classes):
            cname = names[int(cls)]
            if   cname == 'bike_with_rider': bikes_tracked.append((box, int(tid)))
            elif cname == 'Rider':           riders.append(box)
            elif cname == 'pillion':         pillions.append(box)
            elif cname == 'no_helmet':       no_helmets.append(box)
            elif cname == 'number_plate':    plates.append(box)

        # ── Per-bike processing ──────────────────────────────────────────────
        for bike_box, track_id in bikes_tracked:

            if track_id not in track_history:
                track_history[track_id] = new_track_state()

            state = track_history[track_id]

            # ── FILTER 1: Scale gate ─────────────────────────────────────────
            _, bh = box_wh(bike_box)
            if bh < MIN_BIKE_HEIGHT_PX:
                cv2.putText(frame, f"ID:{track_id} [far]",
                            (int(bike_box[0]), int(bike_box[1])-15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (128,128,128), 1)
                continue

            # ── FILTER 2: Road zone gate ─────────────────────────────────────
            cx, cy = box_centre(bike_box)
            if cy < HORIZON_Y:
                cv2.putText(frame, f"ID:{track_id} [off-road]",
                            (int(bike_box[0]), int(bike_box[1])-15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (128,128,128), 1)
                continue

            # ── FILTER 3: Motion gate (ego-corrected flow) ───────────────────
            if flow_map is not None:
                residual_mag = compute_residual_flow(flow_map, bike_box, ego_vec)
                state['motion_scores'].append(residual_mag)
            avg_motion = np.mean(state['motion_scores']) if state['motion_scores'] else 0.0

            is_moving_flow = avg_motion >= MOTION_FLOW_THRESH

            # ── FILTER 4: Trajectory gate ────────────────────────────────────
            state['centroids'].append((cx, cy))
            net_disp = 0.0
            if len(state['centroids']) >= 5:
                oldest = state['centroids'][0]
                net_disp = np.hypot(cx - oldest[0], cy - oldest[1])

            is_moving_traj = net_disp >= TRAJ_DISP_THRESH

            # Combined motion decision: OR between flow and trajectory
            # (one sensor failing doesn't block the other)
            is_moving = is_moving_flow or is_moving_traj

            if not is_moving:
                state['helmet_frames'] = 0
                state['triple_frames'] = 0
                cv2.putText(frame, f"ID:{track_id} [parked]",
                            (int(bike_box[0]), int(bike_box[1])-15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180,180,0), 1)
                continue

            if state['penalized']:
                cv2.putText(frame, f"ID:{track_id} (PENALIZED)",
                            (int(bike_box[0]), int(bike_box[1])-15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                cv2.rectangle(frame,
                              (int(bike_box[0]),int(bike_box[1])),
                              (int(bike_box[2]),int(bike_box[3])),
                              (0,0,255), 2)
                continue

            # ── FILTER 5: Improved rider association ─────────────────────────
            bike_riders  = [r for r in riders
                            if ioa(r, bike_box) > OVERLAP_THRESHOLD
                            and is_valid_rider(r, bike_box)]
            bike_pillions= [p for p in pillions
                            if ioa(p, bike_box) > OVERLAP_THRESHOLD
                            and is_valid_rider(p, bike_box)]

            for p in plates:
                if ioa(p, bike_box) > OVERLAP_THRESHOLD:
                    state['best_plate'] = p
                    break

            # Triple-seat counter
            if (len(bike_riders) + len(bike_pillions)) >= 3:
                state['triple_frames'] += 1
            else:
                state['triple_frames'] = 0

            # Helmet-violation counter (driver only)
            driver_no_helmet = False
            for nh in no_helmets:
                if ioa(nh, bike_box) <= OVERLAP_THRESHOLD:
                    continue
                best_body, best_ioa, best_is_rider = None, 0.0, False
                for body in (bike_riders + bike_pillions):
                    ratio = ioa(nh, body)
                    if ratio > best_ioa and ratio > OVERLAP_THRESHOLD:
                        best_ioa    = ratio
                        best_body   = body
                        best_is_rider = any((body == r).all() for r in bike_riders)
                if best_body is not None and best_is_rider:
                    driver_no_helmet = True
                    break

            if driver_no_helmet:
                state['helmet_frames'] += 1
            else:
                state['helmet_frames'] = 0

            # ── Violation trigger ─────────────────────────────────────────────
            if state['helmet_frames'] >= VIOLATION_FRAMES:
                log_violation(frame, track_id, "NO_HELMET",
                              state['best_plate'], bike_box)
                state['penalized'] = True

            elif state['triple_frames'] >= VIOLATION_FRAMES:
                log_violation(frame, track_id, "TRIPLE_SEAT",
                              state['best_plate'], bike_box)
                state['penalized'] = True

            if not state['penalized']:
                cv2.putText(frame, f"ID:{track_id}",
                            (int(bike_box[0]), int(bike_box[1])-15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        out_video.write(frame)
        cv2.imshow("Urbanity Tracker", cv2.resize(frame, (1024, 576)))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out_video.release()
    cv2.destroyAllWindows()
    print(f"Done. Video saved to: {OUTPUT_VIDEO_PATH}")


if __name__ == "__main__":
    main()