import cv2
import time

print("Attempting to open webcam...")
cap = cv2.VideoCapture(0) # Try 0 first, then 1, then 2 if 0 fails

if not cap.isOpened():
    print("ERROR: Cannot open camera at index 0. Trying index 1...")
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("ERROR: Cannot open camera at index 1 either. Trying index 2...")
        cap = cv2.VideoCapture(2)
        if not cap.isOpened():
            print("CRITICAL ERROR: Could not open any camera. Check camera connections and privacy settings.")
            exit()

print("Webcam opened successfully. Press 'q' to quit.")
frame_counter = 0
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print(f"WARNING: Can't receive frame (stream end?). Exiting after {frame_counter} frames.")
        break

    frame_counter += 1
    if frame_counter % 30 == 0: # Print every 30 frames (approx 1 second at 30 FPS)
        print(f"DEBUG: Grabbed frame {frame_counter}. Time elapsed: {time.time() - start_time:.2f}s")

    cv2.imshow('Webcam Test - Press Q to Quit', frame)

    if cv2.waitKey(1) == ord('q'):
        print("Quitting webcam test.")
        break

cap.release()
cv2.destroyAllWindows()
print("Webcam test finished.")