import cv2
from ultralytics import YOLO

# Load your trained model
model = YOLO(r"C:/Users/piyus/Documents/Wildlife_detection/runs/detect/train5/weights/best.pt")   # make sure path is correct

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO prediction
    results = model(frame, conf=0.8, verbose=False)

    # Draw predictions on frame
    annotated_frame = results[0].plot()

    # Show frame
    cv2.imshow("Wildlife Detection - Real Time", annotated_frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# import cv2
# from ultralytics import YOLO

# model = YOLO(r"C:/Users/piyus/OneDrive/Documents/Wildlife_Detection/Dataset/Animal/runs/detect/train4/weights/best.pt")

# cap = cv2.VideoCapture(0)

# frame_count = 0
# detect_every = 5   # Detect once every 5 frames

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     frame_count += 1

#     if frame_count % detect_every == 0:
#         results = model(frame)
#         annotated_frame = results[0].plot()
#     else:
#         annotated_frame = frame

#     cv2.imshow("Wildlife Detection - Real Time", annotated_frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()