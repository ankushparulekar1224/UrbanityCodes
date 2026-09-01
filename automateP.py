import cv2
import os
import math
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass, field
from enum import Enum, auto
from ultralytics import YOLO

# --- LOGGING SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('labeler.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# --- ENUMS AND DATA CLASSES ---
class Action(Enum):
    NONE = auto()
    DRAW = auto()
    MOVE = auto()
    RESIZE_TL = auto()
    RESIZE_TR = auto()
    RESIZE_BL = auto()
    RESIZE_BR = auto()


@dataclass
class BoundingBox:
    class_id: int
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float = 1.0
    
    def normalize(self):
        """Ensure x1 < x2 and y1 < y2."""
        self.x1, self.x2 = min(self.x1, self.x2), max(self.x1, self.x2)
        self.y1, self.y2 = min(self.y1, self.y2), max(self.y1, self.y2)
    
    def contains(self, x: int, y: int) -> bool:
        return self.x1 < x < self.x2 and self.y1 < y < self.y2
    
    def width(self) -> int:
        return abs(self.x2 - self.x1)
    
    def height(self) -> int:
        return abs(self.y2 - self.y1)
    
    def area(self) -> int:
        return self.width() * self.height()
    
    def to_yolo(self, img_w: int, img_h: int) -> Tuple[float, float, float, float]:
        cx = ((self.x1 + self.x2) / 2) / img_w
        cy = ((self.y1 + self.y2) / 2) / img_h
        w = self.width() / img_w
        h = self.height() / img_h
        return cx, cy, w, h


@dataclass
class Config:
    model_path: str
    image_dir: str
    label_dir: str
    progress_file: str = 'session_progress.json'
    confidence_threshold: float = 0.4
    min_box_size: int = 10
    corner_tolerance: int = 12
    auto_save_interval: int = 5
    
    # Visual settings
    box_thickness: int = 2
    font_scale: float = 0.6
    corner_radius: int = 6
    
    classes: Dict[int, str] = field(default_factory=lambda: {
        0: 'Rider',
        1: 'Car',
        2: 'No Helmet',
        3: 'Helmet',
        4: 'Number Plate',
        5: 'Pillion',
        6: 'Bike + Rider',
        7: 'Using Phone',
        8: 'Unknown'
    })
    
    colors: List[Tuple[int, int, int]] = field(default_factory=lambda: [
        (66, 133, 244),   # Blue
        (52, 168, 83),    # Green
        (234, 67, 53),    # Red
        (251, 188, 5),    # Yellow
        (154, 160, 166),  # Gray
        (255, 112, 67),   # Orange
        (171, 71, 188),   # Purple
        (0, 172, 193),    # Cyan
        (128, 128, 128),  # Dark Gray for Unknown
    ])


class SessionState:
    """Manages session progress and statistics."""
    
    def __init__(self, config: Config):
        self.config = config
        self.start_time = datetime.now()
        self.images_labeled = 0
        self.boxes_created = 0
        self.boxes_deleted = 0
        self.boxes_modified = 0
        self.current_index = 0
        
    def save(self):
        data = {
            'current_index': self.current_index,
            'images_labeled': self.images_labeled,
            'boxes_created': self.boxes_created,
            'last_session': datetime.now().isoformat(),
        }
        with open(self.config.progress_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load(self) -> int:
        if os.path.exists(self.config.progress_file):
            try:
                with open(self.config.progress_file, 'r') as f:
                    data = json.load(f)
                    self.current_index = data.get('current_index', 0)
                    self.images_labeled = data.get('images_labeled', 0)
                    return self.current_index
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Could not load session: {e}")
        return 0


class UrbanityLabeler:
    """Production-ready image labeling tool with YOLO integration."""
    
    WINDOW_NAME = 'Urbanity Auto-Labeler'
    
    def __init__(self, config: Config):
        self.config = config
        self.session = SessionState(config)
        self.model: Optional[YOLO] = None
        
        # State
        self.boxes: List[BoundingBox] = []
        self.undo_stack: List[List[BoundingBox]] = []
        self.redo_stack: List[List[BoundingBox]] = []
        self.current_class = 0
        self.selected_idx = -1
        self.hovered_idx = -1
        self.action = Action.NONE
        
        # Mouse tracking
        self.mouse_pos = (0, 0)
        self.drag_start = (0, 0)
        self.drag_offset = (0, 0)
        
        # Display
        self.show_confidence = True
        self.show_help = False
        self.zoom_level = 1.0
        self.pan_offset = (0, 0)
        
        # Image data
        self.current_image: Optional[Any] = None
        self.image_files: List[str] = []
        self.current_idx = 0
        
        self._init_directories()
        
    def _init_directories(self):
        os.makedirs(self.config.label_dir, exist_ok=True)
        logger.info(f"Label directory: {self.config.label_dir}")
        
    def load_model(self):
        logger.info(f"Loading model from {self.config.model_path}")
        try:
            self.model = YOLO(self.config.model_path)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def load_images(self):
        extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
        self.image_files = sorted([
            f for f in os.listdir(self.config.image_dir)
            if f.lower().endswith(extensions)
        ])
        logger.info(f"Found {len(self.image_files)} images")
        
        if not self.image_files:
            raise ValueError(f"No images found in {self.config.image_dir}")
    
    def _save_state(self):
        """Save current state to undo stack."""
        self.undo_stack.append([BoundingBox(**vars(b)) for b in self.boxes])
        self.redo_stack.clear()
        if len(self.undo_stack) > 50:
            self.undo_stack.pop(0)
    
    def undo(self):
        if self.undo_stack:
            self.redo_stack.append([BoundingBox(**vars(b)) for b in self.boxes])
            self.boxes = self.undo_stack.pop()
            
    def redo(self):
        if self.redo_stack:
            self.undo_stack.append([BoundingBox(**vars(b)) for b in self.boxes])
            self.boxes = self.redo_stack.pop()
    
    def _get_action_at(self, x: int, y: int) -> Tuple[int, Action]:
        tol = self.config.corner_tolerance
        
        # Check corners first (resize handles)
        for i, box in enumerate(self.boxes):
            corners = [
                ((box.x1, box.y1), Action.RESIZE_TL),
                ((box.x2, box.y2), Action.RESIZE_BR),
                ((box.x2, box.y1), Action.RESIZE_TR),
                ((box.x1, box.y2), Action.RESIZE_BL),
            ]
            for (cx, cy), action in corners:
                if math.hypot(x - cx, y - cy) < tol:
                    return i, action
        
        # Check for box interior (move)
        for i, box in enumerate(self.boxes):
            if box.contains(x, y):
                return i, Action.MOVE
        
        return -1, Action.DRAW
    
    def _mouse_callback(self, event: int, x: int, y: int, flags: int, param: Any):
        self.mouse_pos = (x, y)
        
        # Update hover state
        self.hovered_idx, _ = self._get_action_at(x, y)
        
        if event == cv2.EVENT_LBUTTONDOWN:
            self.selected_idx, self.action = self._get_action_at(x, y)
            self.drag_start = (x, y)
            
            if self.action == Action.MOVE and self.selected_idx != -1:
                box = self.boxes[self.selected_idx]
                self.drag_offset = (x - box.x1, y - box.y1)
                self._save_state()
            elif self.action == Action.DRAW:
                self._save_state()
            elif self.action in (Action.RESIZE_TL, Action.RESIZE_TR, 
                                  Action.RESIZE_BL, Action.RESIZE_BR):
                self._save_state()
                
        elif event == cv2.EVENT_MOUSEMOVE and self.action != Action.NONE:
            if self.action == Action.MOVE and self.selected_idx != -1:
                box = self.boxes[self.selected_idx]
                w, h = box.width(), box.height()
                box.x1 = x - self.drag_offset[0]
                box.y1 = y - self.drag_offset[1]
                box.x2 = box.x1 + w
                box.y2 = box.y1 + h
                
            elif self.action == Action.RESIZE_TL and self.selected_idx != -1:
                self.boxes[self.selected_idx].x1 = x
                self.boxes[self.selected_idx].y1 = y
            elif self.action == Action.RESIZE_BR and self.selected_idx != -1:
                self.boxes[self.selected_idx].x2 = x
                self.boxes[self.selected_idx].y2 = y
            elif self.action == Action.RESIZE_TR and self.selected_idx != -1:
                self.boxes[self.selected_idx].x2 = x
                self.boxes[self.selected_idx].y1 = y
            elif self.action == Action.RESIZE_BL and self.selected_idx != -1:
                self.boxes[self.selected_idx].x1 = x
                self.boxes[self.selected_idx].y2 = y
                
        elif event == cv2.EVENT_LBUTTONUP:
            if self.action == Action.DRAW:
                x1, x2 = min(self.drag_start[0], x), max(self.drag_start[0], x)
                y1, y2 = min(self.drag_start[1], y), max(self.drag_start[1], y)
                
                if (x2 - x1 > self.config.min_box_size and 
                    y2 - y1 > self.config.min_box_size):
                    self.boxes.append(BoundingBox(
                        class_id=self.current_class,
                        x1=x1, y1=y1, x2=x2, y2=y2
                    ))
                    self.session.boxes_created += 1
                    
            elif self.selected_idx != -1:
                self.boxes[self.selected_idx].normalize()
                self.session.boxes_modified += 1
                
            self.action = Action.NONE
            
        elif event == cv2.EVENT_RBUTTONDOWN:
            idx, act = self._get_action_at(x, y)
            if idx != -1:
                self._save_state()
                self.boxes.pop(idx)
                self.session.boxes_deleted += 1
                self.selected_idx = -1
    
    def _draw_box(self, display: Any, box: BoundingBox, idx: int, 
                  is_selected: bool = False, is_hovered: bool = False):
        color = self.config.colors[box.class_id % len(self.config.colors)]
        thickness = self.config.box_thickness + (1 if is_hovered else 0)
        
        # Draw box with slight transparency for selected
        if is_selected:
            overlay = display.copy()
            cv2.rectangle(overlay, (box.x1, box.y1), (box.x2, box.y2), color, -1)
            cv2.addWeighted(overlay, 0.15, display, 0.85, 0, display)
        
        # Main rectangle
        cv2.rectangle(display, (box.x1, box.y1), (box.x2, box.y2), color, thickness)
        
        # Label background
        label = self.config.classes.get(box.class_id, "Unknown")
        if self.show_confidence and box.confidence < 1.0:
            label += f" {box.confidence:.0%}"
        
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 
                                       self.config.font_scale, 1)
        
        label_y = box.y1 - 8 if box.y1 > 30 else box.y2 + th + 8
        cv2.rectangle(display, (box.x1, label_y - th - 4), 
                      (box.x1 + tw + 8, label_y + 4), color, -1)
        cv2.putText(display, label, (box.x1 + 4, label_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, self.config.font_scale, 
                    (255, 255, 255), 1, cv2.LINE_AA)
        
        # Corner handles
        r = self.config.corner_radius
        for pt in [(box.x1, box.y1), (box.x2, box.y2), 
                   (box.x2, box.y1), (box.x1, box.y2)]:
            cv2.circle(display, pt, r, (255, 255, 255), -1)
            cv2.circle(display, pt, r, color, 2)
    
    def _draw_ui(self, display: Any, img_name: str):
        h, w = display.shape[:2]
        
        # Semi-transparent top bar
        overlay = display.copy()
        cv2.rectangle(overlay, (0, 0), (w, 100), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.85, display, 0.15, 0, display)
        
        # Status info
        progress = f"{self.current_idx + 1}/{len(self.image_files)}"
        class_name = self.config.classes.get(self.current_class, "Unknown")
        class_color = self.config.colors[self.current_class % len(self.config.colors)]
        
        # Progress bar
        bar_w = 200
        bar_h = 8
        bar_x = w - bar_w - 20
        progress_pct = (self.current_idx + 1) / len(self.image_files)
        cv2.rectangle(display, (bar_x, 15), (bar_x + bar_w, 15 + bar_h), 
                      (80, 80, 80), -1)
        cv2.rectangle(display, (bar_x, 15), 
                      (bar_x + int(bar_w * progress_pct), 15 + bar_h), 
                      (66, 133, 244), -1)
        
        cv2.putText(display, progress, (bar_x + bar_w + 10, 22), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        
        # Current file and class
        cv2.putText(display, img_name[:50], (15, 28), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Class indicator
        cv2.circle(display, (20, 55), 10, class_color, -1)
        cv2.putText(display, f"Class: {class_name}", (40, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Box count
        cv2.putText(display, f"Boxes: {len(self.boxes)}", (40, 85), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)
        
        # Help toggle hint
        help_text = "H: Help" if not self.show_help else "H: Hide Help"
        cv2.putText(display, help_text, (w - 100, 85), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1, cv2.LINE_AA)
        
        # Draw help overlay
        if self.show_help:
            self._draw_help_overlay(display)
    
    def _draw_help_overlay(self, display: Any):
        h, w = display.shape[:2]
        
        overlay = display.copy()
        panel_w, panel_h = 320, 340
        panel_x = (w - panel_w) // 2
        panel_y = (h - panel_h) // 2
        
        cv2.rectangle(overlay, (panel_x, panel_y), 
                      (panel_x + panel_w, panel_y + panel_h), (40, 40, 40), -1)
        cv2.addWeighted(overlay, 0.95, display, 0.05, 0, display)
        
        # Border
        cv2.rectangle(display, (panel_x, panel_y), 
                      (panel_x + panel_w, panel_y + panel_h), (100, 100, 100), 1)
        
        # Title
        cv2.putText(display, "Keyboard Shortcuts", (panel_x + 20, panel_y + 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
        
        shortcuts = [
            ("0-8", "Select class"),
            ("T", "Cycle class (hover)"),
            ("N / ->", "Next image"),
            ("B / <-", "Previous image"),
            ("Ctrl+Z", "Undo"),
            ("Ctrl+Y", "Redo"),
            ("C", "Toggle confidence"),
            ("Delete", "Clear all boxes"),
            ("S", "Save current"),
            ("Q / Esc", "Save & quit"),
        ]
        
        y = panel_y + 70
        for key, desc in shortcuts:
            cv2.putText(display, key, (panel_x + 25, y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 180, 255), 1, cv2.LINE_AA)
            cv2.putText(display, desc, (panel_x + 100, y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
            y += 26
        
        # Mouse controls
        y += 10
        cv2.putText(display, "Mouse Controls", (panel_x + 20, y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        y += 25
        
        mouse_controls = [
            ("Left drag", "Draw / Move / Resize"),
            ("Right click", "Delete box"),
        ]
        for key, desc in mouse_controls:
            cv2.putText(display, key, (panel_x + 25, y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 180, 255), 1, cv2.LINE_AA)
            cv2.putText(display, desc, (panel_x + 120, y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
            y += 22
    
    def _draw_drawing_preview(self, display: Any):
        if self.action == Action.DRAW:
            color = self.config.colors[self.current_class % len(self.config.colors)]
            cv2.rectangle(display, self.drag_start, self.mouse_pos, color, 2)
            
            # Dimension preview
            w = abs(self.mouse_pos[0] - self.drag_start[0])
            h = abs(self.mouse_pos[1] - self.drag_start[1])
            dim_text = f"{w}x{h}"
            cv2.putText(display, dim_text, 
                        (self.mouse_pos[0] + 10, self.mouse_pos[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    
    def save_labels(self, img_name: str, img_w: int, img_h: int):
        label_name = Path(img_name).stem + '.txt'
        filepath = os.path.join(self.config.label_dir, label_name)
        
        with open(filepath, 'w') as f:
            for box in self.boxes:
                cx, cy, w, h = box.to_yolo(img_w, img_h)
                f.write(f"{box.class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
        
        logger.debug(f"Saved {len(self.boxes)} boxes to {filepath}")
    
    def load_labels(self, img_name: str, img_w: int, img_h: int) -> List[BoundingBox]:
        label_name = Path(img_name).stem + '.txt'
        filepath = os.path.join(self.config.label_dir, label_name)
        
        boxes = []
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cid = int(parts[0])
                        cx, cy, w, h = map(float, parts[1:5])
                        x1 = int((cx - w/2) * img_w)
                        y1 = int((cy - h/2) * img_h)
                        x2 = int((cx + w/2) * img_w)
                        y2 = int((cy + h/2) * img_h)
                        boxes.append(BoundingBox(cid, x1, y1, x2, y2))
        return boxes
    
    def run_inference(self, img_path: str) -> List[BoundingBox]:
        if not self.model:
            return []
        
        results = self.model.predict(
            img_path, 
            conf=self.config.confidence_threshold, 
            verbose=False
        )
        
        boxes = []
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                boxes.append(BoundingBox(cls_id, x1, y1, x2, y2, conf))
        
        return boxes
    
    def handle_key(self, key: int) -> Optional[str]:
        """Handle keyboard input. Returns 'next', 'prev', 'quit', or None."""
        
        # Class selection 0-8
        if ord('0') <= key <= ord('8'):
            self.current_class = key - ord('0')
            return None
        
        # Cycle class on hovered box
        if key in (ord('t'), ord('T')):
            idx, _ = self._get_action_at(*self.mouse_pos)
            if idx != -1:
                self._save_state()
                self.boxes[idx].class_id = (self.boxes[idx].class_id + 1) % len(self.config.classes)
            return None
        
        # Navigation
        if key in (ord('n'), ord('N'), 83, 3):  # N or Right arrow
            return 'next'
        if key in (ord('b'), ord('B'), 81, 2):  # B or Left arrow
            return 'prev'
        
        # Undo/Redo (Ctrl+Z, Ctrl+Y)
        if key == 26:  # Ctrl+Z
            self.undo()
            return None
        if key == 25:  # Ctrl+Y
            self.redo()
            return None
        
        # Toggle help
        if key in (ord('h'), ord('H')):
            self.show_help = not self.show_help
            return None
        
        # Toggle confidence display
        if key in (ord('c'), ord('C')):
            self.show_confidence = not self.show_confidence
            return None
        
        # Clear all boxes
        if key == 255 or key == 127:  # Delete key
            if self.boxes:
                self._save_state()
                self.boxes.clear()
            return None
        
        # Manual save
        if key in (ord('s'), ord('S')):
            return 'save'
        
        # Quit
        if key in (ord('q'), ord('Q'), 27):  # Q or Esc
            return 'quit'
        
        return None
    
    def run(self):
        logger.info("Starting Urbanity Auto-Labeler")
        
        self.load_model()
        self.load_images()
        self.current_idx = self.session.load()
        
        cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.WINDOW_NAME, self._mouse_callback)
        
        # Set reasonable initial window size
        cv2.resizeWindow(self.WINDOW_NAME, 1280, 800)
        
        try:
            while 0 <= self.current_idx < len(self.image_files):
                img_name = self.image_files[self.current_idx]
                img_path = os.path.join(self.config.image_dir, img_name)
                
                self.current_image = cv2.imread(img_path)
                if self.current_image is None:
                    logger.warning(f"Could not load {img_path}, skipping")
                    self.current_idx += 1
                    continue
                
                img_h, img_w = self.current_image.shape[:2]
                
                # Load existing labels or run inference
                self.boxes = self.load_labels(img_name, img_w, img_h)
                if not self.boxes:
                    self.boxes = self.run_inference(img_path)
                    if self.boxes:
                        logger.info(f"Auto-detected {len(self.boxes)} objects")
                
                self.undo_stack.clear()
                self.redo_stack.clear()
                
                # Image editing loop
                while True:
                    display = self.current_image.copy()
                    
                    # Draw boxes
                    for i, box in enumerate(self.boxes):
                        self._draw_box(
                            display, box, i,
                            is_selected=(i == self.selected_idx),
                            is_hovered=(i == self.hovered_idx)
                        )
                    
                    # Draw preview while drawing
                    self._draw_drawing_preview(display)
                    
                    # Draw UI
                    self._draw_ui(display, img_name)
                    
                    cv2.imshow(self.WINDOW_NAME, display)
                    key = cv2.waitKey(30) & 0xFF
                    
                    if key == 255:  # No key pressed
                        continue
                    
                    action = self.handle_key(key)
                    
                    if action == 'next':
                        self.save_labels(img_name, img_w, img_h)
                        self.session.images_labeled += 1
                        self.current_idx += 1
                        self.session.current_index = self.current_idx
                        self.session.save()
                        break
                        
                    elif action == 'prev':
                        self.save_labels(img_name, img_w, img_h)
                        if self.current_idx > 0:
                            self.current_idx -= 1
                            self.session.current_index = self.current_idx
                            self.session.save()
                        break
                        
                    elif action == 'save':
                        self.save_labels(img_name, img_w, img_h)
                        self.session.save()
                        logger.info(f"Saved {img_name}")
                        
                    elif action == 'quit':
                        self.save_labels(img_name, img_w, img_h)
                        self.session.save()
                        logger.info("Exiting...")
                        return
            
            logger.info("All images processed!")
            
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
            self.session.save()
        finally:
            cv2.destroyAllWindows()


def main():
    config = Config(
        model_path=r"c:\Users\thero\Downloads\best 11.pt",
        image_dir="C:\\Urbanity-data\\ankush7\\images",
        label_dir="c:\\Urbanity-data\\ankush7\\labels",
        confidence_threshold=0.4,
    )
    
    labeler = UrbanityLabeler(config)
    labeler.run()


if __name__ == '__main__':
    main()