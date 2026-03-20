import os
import cv2
from ultralytics import YOLO

# ================== CONFIG ==================
MODEL_PATH = "runs/detect/train4/weights/best.pt"

INPUT_IMAGE_DIR = "input/images"
OUTPUT_IMAGE_DIR = "output/images"

INPUT_VIDEO_DIR = "input/videos"
OUTPUT_VIDEO_DIR = "output/videos"

CONFIDENCE = 0.5
# ============================================

# Create output folders
os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)
os.makedirs(OUTPUT_VIDEO_DIR, exist_ok=True)

# Load model
model = YOLO(MODEL_PATH)

# Get class names
class_names = model.names

# ================== IMAGE INFERENCE ==================
print("\n Processing Images...\n")

for img_name in os.listdir(INPUT_IMAGE_DIR):

    img_path = os.path.join(INPUT_IMAGE_DIR, img_name)

    if not img_name.lower().endswith((".jpg", ".png", ".jpeg")):
        continue

    frame = cv2.imread(img_path)

    results = model(frame, conf=CONFIDENCE)

    annotated = results[0].plot()

    #  PRINT DETECTIONS
    print(f"\nDetections in {img_name}:")
    if len(results[0].boxes) == 0:
        print("No objects detected.")
    else:
        for box in results[0].boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            print(f"{class_names[cls]}: {conf:.2f}")

    # Save output image
    save_path = os.path.join(OUTPUT_IMAGE_DIR, img_name)
    cv2.imwrite(save_path, annotated)

    print(f"Saved: {save_path}")

# ================== VIDEO INFERENCE ==================
print("\n Processing Videos...\n")

for vid_name in os.listdir(INPUT_VIDEO_DIR):

    vid_path = os.path.join(INPUT_VIDEO_DIR, vid_name)

    if not vid_name.lower().endswith((".mp4", ".avi", ".mov")):
        continue

    cap = cv2.VideoCapture(vid_path)

    width = int(cap.get(3))
    height = int(cap.get(4))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    out_path = os.path.join(OUTPUT_VIDEO_DIR, vid_name)

    out = cv2.VideoWriter(
        out_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (width, height)
    )

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=CONFIDENCE)
        annotated = results[0].plot()

        #  PRINT DETECTIONS (every 30 frames to avoid spam)
        if frame_count % 30 == 0:
            print(f"\nDetections in {vid_name} (frame {frame_count}):")

            if len(results[0].boxes) == 0:
                print("No objects detected.")
            else:
                for box in results[0].boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    print(f"{class_names[cls]}: {conf:.2f}")

        out.write(annotated)
        frame_count += 1

    cap.release()
    out.release()

    print(f"Saved: {out_path}")

print("\n✅ Inference Completed!")