import os
import random
import shutil
from abc import ABC
dataset_path = r"C:/Users/piyus/OneDrive/Documents/Wildlife_Detection/Dataset/New folder"

images_path = os.path.join(dataset_path, "Bear")
labels_path = os.path.join(dataset_path, "labels")

# Get all images
images = [f for f in os.listdir(images_path) if f.endswith((".jpg", ".jpeg", ".png"))]

print(f"Total images found: {len(images)}")

# class Wildlife(ABC):
#     def __init__(self,path):
#         self.path=path
#     def bark():
#         pass
        

# a=Wildlife()
# a.bark()

random.shuffle(images)

train_ratio = 0.7
val_ratio = 0.2

train_count = int(train_ratio * len(images))
val_count = int(val_ratio * len(images))

train_files = images[:train_count]
val_files = images[train_count:train_count + val_count]
test_files = images[train_count + val_count:]

# Create new folder structure
for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(dataset_path, split, "images"), exist_ok=True)
    os.makedirs(os.path.join(dataset_path, split, "labels"), exist_ok=True)

def move_files(files, split):
    for file in files:
        img_src = os.path.join(images_path, file)
        label_file = os.path.splitext(file)[0] + ".txt"
        lbl_src = os.path.join(labels_path, label_file)

        img_dst = os.path.join(dataset_path, split, "images", file)
        lbl_dst = os.path.join(dataset_path, split, "labels", label_file)

        if os.path.exists(lbl_src):
            shutil.move(img_src, img_dst)
            shutil.move(lbl_src, lbl_dst)
        else:
            print(f"Label missing for {file}")

move_files(train_files, "train")
move_files(val_files, "val")
move_files(test_files, "test")

print("✅ Dataset split completed successfully!")
