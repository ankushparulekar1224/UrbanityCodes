import cv2
import time

# Initialize camera
cap = cv2.VideoCapture(cv2.CAP_DSHOW)

# Variables for tracking time
prev_time = 0

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Calculate actual FPS based on clock time
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    # Display the calculated FPS on the window
    cv2.putText(frame, f"Actual FPS: {int(fps)}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Live Stream", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()