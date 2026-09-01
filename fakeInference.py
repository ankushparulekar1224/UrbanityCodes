import cv2
import math
import os
import json
import sys

# --- CONFIGURATION ---
VIDEO_PATH = r"c:\Users\thero\Downloads\NO20260326-185528-000062F_inference.mp4"
OUTPUT_VIDEO = r"C:\Urbanity\blinking_output1.mp4"
ANNOTATIONS_PATH = r"C:\Urbanity\annotations.json"  # autosaved/resumed automatically
SKIP_FRAMES = 5
BLINK_PERIOD_SEC = 0.4  # how long each on/off phase of the blink lasts

CLASSES = {1: "HLR V", 2: "TSR V", 3: "MU V"}
COLORS = {1: (0, 0, 255),    # Red
          2: (0, 0, 255),  # Orange
          3: (0, 0, 255)}  # Magenta

# --- GLOBAL STATE ---
annotations = {}          # { frame_number: [[cid, x1, y1, x2, y2], ...] }
current_boxes = []
active_class = 1
max_render_frame = 0
frame_w = 0
frame_h = 0

start_pt = (-1, -1)
current_pt = (-1, -1)
selected_idx = -1
action = None
drag_offset_x = 0
drag_offset_y = 0


# --- PERSISTENCE ---
def load_annotations():
    """Resume a previous session if a save file exists."""
    if os.path.exists(ANNOTATIONS_PATH):
        try:
            with open(ANNOTATIONS_PATH, "r") as f:
                raw = json.load(f)
            return {int(k): v for k, v in raw.items()}
        except (json.JSONDecodeError, ValueError):
            print(f"Warning: could not parse {ANNOTATIONS_PATH}, starting fresh.")
    return {}


def save_annotations():
    os.makedirs(os.path.dirname(ANNOTATIONS_PATH), exist_ok=True)
    with open(ANNOTATIONS_PATH, "w") as f:
        json.dump(annotations, f)


# --- TEXT RENDERING ---
FONT = cv2.FONT_HERSHEY_DUPLEX  # cleaner, more legible than SIMPLEX at small sizes

def put_label(img, text, org, scale=0.7, fg=(255, 255, 255), bg=None, thickness=2):
    """Draw text that stays readable over any background.
    If bg is given, paints a solid plate behind the text (like the box labels).
    Otherwise falls back to a black outline behind the text (like the HUD)."""
    x, y = org
    (tw, th), baseline = cv2.getTextSize(text, FONT, scale, thickness)

    if bg is not None:
        pad = 6
        cv2.rectangle(img, (x - pad, y - th - pad), (x + tw + pad, y + baseline + pad // 2), bg, -1)
        cv2.putText(img, text, (x, y), FONT, scale, fg, thickness, cv2.LINE_AA)
    else:
        # Outline: draw the text several times in black, offset, then the real text on top
        cv2.putText(img, text, (x, y), FONT, scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
        cv2.putText(img, text, (x, y), FONT, scale, fg, thickness, cv2.LINE_AA)


# --- GEOMETRY HELPERS ---
def clamp_box(box):
    cid, x1, y1, x2, y2 = box
    x1 = max(0, min(x1, frame_w - 1))
    x2 = max(0, min(x2, frame_w - 1))
    y1 = max(0, min(y1, frame_h - 1))
    y2 = max(0, min(y2, frame_h - 1))
    return [cid, x1, y1, x2, y2]


def get_action(x, y, boxes_list):
    TOLERANCE = 35
    for i in range(len(boxes_list) - 1, -1, -1):
        cid, x1, y1, x2, y2 = boxes_list[i]
        if math.hypot(x - x1, y - y1) < TOLERANCE: return i, 'resize_tl'
        if math.hypot(x - x2, y - y2) < TOLERANCE: return i, 'resize_br'
        if math.hypot(x - x2, y - y1) < TOLERANCE: return i, 'resize_tr'
        if math.hypot(x - x1, y - y2) < TOLERANCE: return i, 'resize_bl'

    for i in range(len(boxes_list) - 1, -1, -1):
        cid, x1, y1, x2, y2 = boxes_list[i]
        if min(x1, x2) < x < max(x1, x2) and min(y1, y2) < y < max(y1, y2):
            return i, 'move'

    return -1, 'draw'


def mouse_callback(event, x, y, flags, param):
    global start_pt, current_pt, current_boxes, active_class
    global selected_idx, action, drag_offset_x, drag_offset_y

    x = max(0, min(x, frame_w - 1))
    y = max(0, min(y, frame_h - 1))

    if event == cv2.EVENT_LBUTTONDOWN:
        selected_idx, action = get_action(x, y, current_boxes)
        if action == 'draw':
            start_pt = (x, y)
            current_pt = (x, y)
        elif action == 'move':
            cid, x1, y1, x2, y2 = current_boxes[selected_idx]
            drag_offset_x = x - min(x1, x2)
            drag_offset_y = y - min(y1, y2)

    elif event == cv2.EVENT_MOUSEMOVE:
        if action == 'draw':
            current_pt = (x, y)
        elif action == 'move' and selected_idx != -1:
            cid, x1, y1, x2, y2 = current_boxes[selected_idx]
            w, h = abs(x2 - x1), abs(y2 - y1)
            nx1 = x - drag_offset_x
            ny1 = y - drag_offset_y
            current_boxes[selected_idx] = clamp_box([cid, nx1, ny1, nx1 + w, ny1 + h])
        elif action and action.startswith('resize') and selected_idx != -1:
            cid, x1, y1, x2, y2 = current_boxes[selected_idx]
            if action == 'resize_tl': current_boxes[selected_idx] = [cid, x, y, x2, y2]
            elif action == 'resize_br': current_boxes[selected_idx] = [cid, x1, y1, x, y]
            elif action == 'resize_tr': current_boxes[selected_idx] = [cid, x1, y, x, y2]
            elif action == 'resize_bl': current_boxes[selected_idx] = [cid, x, y1, x2, y]

    elif event == cv2.EVENT_LBUTTONUP:
        if action == 'draw':
            x_min, x_max = min(start_pt[0], x), max(start_pt[0], x)
            y_min, y_max = min(start_pt[1], y), max(start_pt[1], y)
            if x_max - x_min > 5 and y_max - y_min > 5:
                current_boxes.append([active_class, x_min, y_min, x_max, y_max])
        elif action and action.startswith('resize') and selected_idx != -1:
            cid, x1, y1, x2, y2 = current_boxes[selected_idx]
            current_boxes[selected_idx] = clamp_box(
                [cid, min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
            )
        action = None

    elif event == cv2.EVENT_RBUTTONDOWN:
        idx, act = get_action(x, y, current_boxes)
        if act != 'draw':
            current_boxes.pop(idx)


def run_annotation_phase(cap, fps, width, height):
    global current_boxes, active_class, max_render_frame

    cv2.namedWindow("Manual Annotator", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Manual Annotator", mouse_callback)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    current_frame_idx = 0
    key = -1

    while cap.isOpened():
        current_frame_idx = max(0, min(current_frame_idx, max(0, total_frames - 1)))
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_idx)
        ret, frame = cap.read()
        if not ret:
            break

        # Work on a COPY so we never silently mutate saved state before confirming
        current_boxes = [b[:] for b in annotations.get(current_frame_idx, [])]

        while True:
            display = frame.copy()
            cv2.rectangle(display, (0, 0), (width, 110), (0, 0, 0), -1)

            for b in current_boxes:
                cid, x1, y1, x2, y2 = b
                color = COLORS.get(cid, (255, 255, 255))
                cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
                put_label(display, CLASSES[cid], (x1, max(20, y1 - 8)), scale=0.65, fg=(255, 255, 255), bg=color)
                for corner in [(x1, y1), (x2, y2), (x2, y1), (x1, y2)]:
                    cv2.circle(display, corner, 8, color, -1)

            if action == 'draw':
                cv2.rectangle(display, start_pt, current_pt, COLORS[active_class], 2)

            class_hint = " | ".join(f"{k}:{v}" for k, v in CLASSES.items())
            ui1 = f"Frame: {current_frame_idx}/{total_frames} | Class: {CLASSES[active_class]}"
            ui2 = f"{class_hint} | N:Next  B:Back  U:Undo  S:Save  Q:Finish&Render"
            ui3 = "L-Click: Draw/Drag/Resize | R-Click: Delete Box"

            put_label(display, ui1, (20, 35), scale=0.75, fg=(0, 255, 0), thickness=2)
            put_label(display, ui2, (20, 68), scale=0.6, fg=(255, 255, 255), thickness=2)
            put_label(display, ui3, (20, 98), scale=0.6, fg=(0, 255, 255), thickness=2)

            cv2.imshow("Manual Annotator", display)
            key = cv2.waitKey(20) & 0xFF

            if key in (ord(str(k)) for k in CLASSES.keys()):
                active_class = int(chr(key))

            elif key in (ord('u'), ord('U')):
                if current_boxes:
                    current_boxes.pop()

            elif key in (ord('s'), ord('S')):
                annotations[current_frame_idx] = list(current_boxes)
                save_annotations()
                print(f"Saved annotations ({len(annotations)} frames) -> {ANNOTATIONS_PATH}")

            elif key in (ord('n'), ord('N')):
                annotations[current_frame_idx] = list(current_boxes)
                save_annotations()
                current_frame_idx += SKIP_FRAMES
                break

            elif key in (ord('b'), ord('B')):
                annotations[current_frame_idx] = list(current_boxes)
                save_annotations()
                current_frame_idx = max(0, current_frame_idx - SKIP_FRAMES)
                break

            elif key in (ord('q'), ord('Q')):
                annotations[current_frame_idx] = list(current_boxes)
                save_annotations()
                max_render_frame = current_frame_idx
                break

        if key in (ord('q'), ord('Q')):
            break

    cv2.destroyAllWindows()


def render_blinking_video(cap, fps, width, height):
    os.makedirs(os.path.dirname(OUTPUT_VIDEO), exist_ok=True)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    out = cv2.VideoWriter(OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    if not out.isOpened():
        print("Error: could not open VideoWriter. Check OUTPUT_VIDEO path/codec.")
        return

    blink_frame_period = max(1, int(round(fps * BLINK_PERIOD_SEC)))
    render_frame_idx = 0
    total_to_render = max_render_frame + 1

    while cap.isOpened():
        if render_frame_idx > max_render_frame:
            break

        ret, frame = cap.read()
        if not ret:
            break

        base_annotation_frame = (render_frame_idx // SKIP_FRAMES) * SKIP_FRAMES

        if base_annotation_frame in annotations:
            is_blinking_visible = (render_frame_idx // blink_frame_period) % 2 == 0
            if is_blinking_visible:
                for box in annotations[base_annotation_frame]:
                    cid, x1, y1, x2, y2 = box
                    color = COLORS[cid]
                    text = CLASSES[cid]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                    put_label(frame, text, (x1 + 4, max(24, y1 - 8)), scale=0.75, fg=(255, 255, 255), bg=color)

        out.write(frame)
        render_frame_idx += 1

        if render_frame_idx % 50 == 0 or render_frame_idx == total_to_render:
            pct = 100 * render_frame_idx / max(1, total_to_render)
            print(f"\rRendering: {render_frame_idx}/{total_to_render} ({pct:.1f}%)", end="")

    print()
    out.release()


def main():
    global annotations, frame_w, frame_h

    if not os.path.exists(VIDEO_PATH):
        print(f"Error: video not found at {VIDEO_PATH}")
        sys.exit(1)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: could not open video {VIDEO_PATH}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0  # keep float precision; fall back if unreadable
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_w, frame_h = width, height

    annotations = load_annotations()
    if annotations:
        print(f"Resumed {len(annotations)} previously annotated frame(s) from {ANNOTATIONS_PATH}")

    print("--- PHASE 1: MANUAL ANNOTATION ---")
    run_annotation_phase(cap, fps, width, height)

    print("\n--- PHASE 2: RENDERING VIDEO ---")
    print(f"Video will stop automatically at frame {max_render_frame}.")
    render_blinking_video(cap, fps, width, height)

    cap.release()
    print(f"\n✅ Render complete! Your truncated video is saved at: {OUTPUT_VIDEO}")


if __name__ == "__main__":
    main()