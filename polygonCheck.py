import cv2
import numpy as np

# --- CONFIGURATION ---
VIDEO_PATH = "C:/Users/softlabs_group/Downloads/VID20260225142040~2.mp4"

# Global fallback in case lane lines temporarily disappear
last_known_polygon = None 

def get_dynamic_road_polygon(frame):
    global last_known_polygon
    height, width = frame.shape[:2]
    
    # 1. Define the Horizon (Adjust this if your camera points higher or lower)
    # 0.55 means the horizon is 55% of the way down from the top of the screen
    horizon_y = int(height * 0.55) 
    bottom_y = height
    
    # 2. Convert to Grayscale & find edges
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    
    # 3. Mask the edges to only look at the lower half of the screen
    mask = np.zeros_like(edges)
    polygon = np.array([[(0, bottom_y), (width, bottom_y), (width, horizon_y), (0, horizon_y)]])
    cv2.fillPoly(mask, polygon, 255)
    masked_edges = cv2.bitwise_and(edges, mask)
    
    # 4. Find mathematical lines (Hough Transform)
    lines = cv2.HoughLinesP(masked_edges, 1, np.pi/180, 50, minLineLength=100, maxLineGap=50)
    
    if lines is None:
        return last_known_polygon

    left_lines, right_lines = [], []
    
    # 5. Separate Left Lane (negative slope) and Right Lane (positive slope)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 == x1: continue # Prevent divide-by-zero
        slope = (y2 - y1) / (x2 - x1)
        
        # Filter out horizontal lines (slope near 0) and extreme vertical lines
        if slope < -0.3: 
            left_lines.append((slope, y1 - slope * x1))
        elif slope > 0.3: 
            right_lines.append((slope, y1 - slope * x1))
            
    # 6. Average the lines and calculate the 4 corners
    try:
        if left_lines and right_lines:
            left_avg = np.average(left_lines, axis=0)
            right_avg = np.average(right_lines, axis=0)
            
            # y = mx + b  ->  x = (y - b) / m
            left_bottom_x = int((bottom_y - left_avg[1]) / left_avg[0])
            left_top_x = int((horizon_y - left_avg[1]) / left_avg[0])
            
            right_bottom_x = int((bottom_y - right_avg[1]) / right_avg[0])
            right_top_x = int((horizon_y - right_avg[1]) / right_avg[0])
            
            dynamic_polygon = np.array([
                [left_top_x, horizon_y],
                [right_top_x, horizon_y],
                [right_bottom_x, bottom_y],
                [left_bottom_x, bottom_y]
            ], np.int32)
            
            last_known_polygon = dynamic_polygon
            return dynamic_polygon
    except Exception as e:
        pass 

    return last_known_polygon

def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    
    # Allow resizing the window since the source is 4K
    cv2.namedWindow("Dynamic ROI Test", cv2.WINDOW_NORMAL)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Video ended or cannot be read.")
            break
            
        # Get the dynamic polygon for this frame
        dynamic_roi = get_dynamic_road_polygon(frame)
        
        # Draw the semi-transparent highlight
        if dynamic_roi is not None:
            overlay = frame.copy()
            # Fill the polygon with green (BGR format: 0, 255, 0)
            cv2.fillPoly(overlay, [dynamic_roi], (0, 255, 0))
            
            # Blend the overlay with the original frame (0.3 = 30% opacity for the green)
            cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
            
            # Draw a solid red border around the zone for clarity
            cv2.polylines(frame, [dynamic_roi], isClosed=True, color=(0, 0, 255), thickness=3)

        cv2.imshow("Dynamic ROI Test", frame)
        
        # Press 'q' to quit, or Space to pause
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            cv2.waitKey(0) # Pauses the video until you press another key
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()