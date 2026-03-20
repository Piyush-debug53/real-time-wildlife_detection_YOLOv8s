import os
from ultralytics import YOLO

# ================== CONFIG ==================
MODEL_PATH = r"C:/Users/piyus/Documents/Wildlife_detection/runs/detect/train4/weights/best.pt"
IMAGE_FOLDER = "train/images"
LABEL_FOLDER = "train/labels"
NEW_LABEL_FOLDER = "train/labels_fixed"

WIDTH_THRESHOLD = 0.9
HEIGHT_THRESHOLD = 0.9
CONF_THRESHOLD = 0.25
# ===========================================

os.makedirs(NEW_LABEL_FOLDER, exist_ok=True)

model = YOLO(MODEL_PATH)

for label_file in os.listdir(LABEL_FOLDER):

    if not label_file.endswith(".txt"):
        continue

    label_path = os.path.join(LABEL_FOLDER, label_file)
    image_path = os.path.join(IMAGE_FOLDER, label_file.replace(".txt", ".jpg"))
    new_label_path = os.path.join(NEW_LABEL_FOLDER, label_file)

    if not os.path.exists(image_path):
        continue

    with open(label_path, "r") as f:
        lines = f.readlines()

    suspicious = False
    clean_lines = []

    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue

        cls, xc, yc, w, h = parts

        if float(w) > WIDTH_THRESHOLD or float(h) > HEIGHT_THRESHOLD:
            suspicious = True
        else:
            clean_lines.append(line.strip())

    # If suspicious box found → replace using model prediction
    if suspicious:
        results = model(image_path, conf=CONF_THRESHOLD)

        for r in results:
            boxes = r.boxes.xywhn  # normalized
            classes = r.boxes.cls

            for box, cls_id in zip(boxes, classes):
                x, y, w, h = box.tolist()
                clean_lines.append(f"{int(cls_id)} {x} {y} {w} {h}")

        print(f"✔ Corrected: {label_file}")

    # Save new label file
    with open(new_label_path, "w") as f:
        for line in clean_lines:
            f.write(line + "\n")

print("\n✅ Finished creating corrected labels in labels_fixed folder.")