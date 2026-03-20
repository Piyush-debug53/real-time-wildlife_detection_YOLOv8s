from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolov8s.pt")

    model.train(
        data="C:/Users/piyus/Documents/Wildlife_detection/Dataset/Animal/data.yaml",
        epochs=100,
        imgsz=640,
        batch=16,
        patience=20,
        workers=12,
        device=0

    )



# import os
# import cv2

# folders = [
#     "C:/Users/piyus/Documents/Wildlife_detection/Dataset/Animal/train/images",
#     "C:/Users/piyus/Documents/Wildlife_detection/Dataset/Animal/test/images",
#     "C:/Users/piyus/Documents/Wildlife_detection/Dataset/Animal/val/images"
# ]

# for folder in folders:
#     for file in os.listdir(folder):
#         path = os.path.join(folder, file)
#         img = cv2.imread(path)
#         if img is None:
#             print("Corrupt:", path)
#             os.remove(path)

# print("Scan complete.")
# import os
# import cv2

# folders = [
#     "C:/Users/piyus/Documents/Wildlife_detection/Dataset/Animal/train/images",
#     "C:/Users/piyus/Documents/Wildlife_detection/Dataset/Animal/test/images",
#     "C:/Users/piyus/Documents/Wildlife_detection/Dataset/Animal/val/images"
# ]

# corrupt_images = []

# for folder in folders:
#     for file in os.listdir(folder):
#         path = os.path.join(folder, file)

#         img = cv2.imread(path)

#         if img is None:
#             corrupt_images.append(path)

# print("\nScan complete.")
# print("Total corrupt images found:", len(corrupt_images))

# if corrupt_images:
#     print("\nList of corrupt images:")
#     for img_path in corrupt_images:
#         print(img_path)
# else:
#     print("No completely corrupt images found.")

# from PIL import Image
# import os

# folders = [
#     "C:/Users/piyus/Documents/Wildlife_detection/Dataset/Animal/train/images",
#     "C:/Users/piyus/Documents/Wildlife_detection/Dataset/Animal/test/images",
#     "C:/Users/piyus/Documents/Wildlife_detection/Dataset/Animal/val/images"
# ]

# for folder in folders:
#     for file in os.listdir(folder):
#         path = os.path.join(folder, file)
#         try:
#             with Image.open(path) as img:
#                 img = img.convert("RGB")
#                 img.save(path, "JPEG", quality=95)
#         except Exception:
#             print("Problem fixing:", path)

# print("Repair complete.")