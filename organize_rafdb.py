"""
ORGANIZE RAF-DB DATA SCRIPT
Author: Yodhitomo Sidhi Pranoto
Dataset: RAF-DB (Real-world Affective Faces Database)

RAF-DB images come with filenames like: train_00001_aligned.jpg
and a label file: list_patition_label.txt

This script reads the label file and sorts images into folders:
  RAF-DB/train/1_surprise/
  RAF-DB/train/2_fear/
  RAF-DB/train/3_disgust/
  RAF-DB/train/4_happiness/
  RAF-DB/train/5_sadness/
  RAF-DB/train/6_anger/
  RAF-DB/train/7_neutral/

Run this ONCE before running train_emotion_model.py
Usage: python3 organize_rafdb.py
"""

import os
import shutil
import glob

# ============================================================
# CONFIGURATION - Update these paths if needed
# ============================================================

# Path to the RAF-DB label file (came with your RAF-DB download)
LABEL_FILE = "../FER_Project/RAFDB/list_patition_label.txt"

# Where to search for your aligned images (_aligned.jpg files)
IMAGE_SEARCH_PATHS = [
    "../FER_Project/basic/Image/aligned",  # Your actual images location
    "../FER_Project/RAFDB/aligned/train",  # Alternative
    "RAF-DB/aligned",
    "RAF-DB",
]

# Output folders for organised images (inside FER_Final_Project)
TRAIN_OUTPUT = "RAF-DB/train"
TEST_OUTPUT  = "RAF-DB/test"

# ============================================================
# RAF-DB EMOTION LABELS
# The label file uses numbers 1-7 mapped to emotions:
# 1=Surprise, 2=Fear, 3=Disgust, 4=Happiness, 5=Sadness, 6=Anger, 7=Neutral
# ============================================================
EMOTION_MAP = {
    1: "1_surprise",
    2: "2_fear",
    3: "3_disgust",
    4: "4_happiness",
    5: "5_sadness",
    6: "6_anger",
    7: "7_neutral"
}

def find_image_directory():
    """Search for the folder that contains the aligned face images."""
    for path in IMAGE_SEARCH_PATHS:
        if os.path.exists(path):
            # Check the folder actually has images in it
            images = glob.glob(os.path.join(path, "*.jpg")) + \
                     glob.glob(os.path.join(path, "*.png"))
            if images:
                print(f"  Found {len(images)} images in: {path}")
                return path
    return None

def organize_images():
    """Read the label file and copy images into emotion-labelled folders."""

    print("=" * 60)
    print("  Organising RAF-DB Images into Emotion Folders")
    print("=" * 60)

    # Check the label file exists
    if not os.path.exists(LABEL_FILE):
        print(f"\n❌ ERROR: Label file not found at: {LABEL_FILE}")
        print("\n  Put 'list_patition_label.txt' inside your RAF-DB folder.")
        print("  It should look like:")
        print("    train_00001   4")
        print("    train_00002   7")
        print("    test_00001    1")
        return

    print(f"\n✅ Label file found: {LABEL_FILE}")

    # Find the folder with images
    image_dir = find_image_directory()
    if image_dir is None:
        print("\n❌ ERROR: Could not find images!")
        print("  Put your _aligned.jpg files in one of:")
        for p in IMAGE_SEARCH_PATHS:
            print(f"    {p}/")
        return

    # Create the output folders (one per emotion, for train and test)
    print("\nCreating output folders...")
    for emotion_folder in EMOTION_MAP.values():
        os.makedirs(os.path.join(TRAIN_OUTPUT, emotion_folder), exist_ok=True)
        os.makedirs(os.path.join(TEST_OUTPUT,  emotion_folder), exist_ok=True)
    print("  ✅ Folders created")

    # Read labels from file
    print("\nReading labels...")
    labels = {}
    with open(LABEL_FILE, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                image_name = parts[0]   # e.g. "train_00001"
                emotion_id = int(parts[1])  # e.g. 4 (happiness)
                labels[image_name] = emotion_id

    print(f"  ✅ Loaded {len(labels)} labels")

    # Copy each image to the correct emotion folder
    print("\nCopying images (this may take a minute)...")
    copied_train = 0
    copied_test  = 0
    not_found    = 0

    for image_name, emotion_id in labels.items():
        is_train = image_name.startswith("train")  # works for both 'train_00001.jpg' and 'train_00001'
        out_dir  = TRAIN_OUTPUT if is_train else TEST_OUTPUT
        emotion_folder = EMOTION_MAP.get(emotion_id, "unknown")

        # The label file has e.g. 'train_00001.jpg' but actual files are 'train_00001_aligned.jpg'
        # So strip the .jpg from image_name and try both patterns
        base_name = image_name.replace('.jpg', '').replace('.png', '')
        for ext in [f"{base_name}_aligned.jpg", f"{base_name}.jpg",
                    f"{base_name}_aligned.png", f"{base_name}.png",
                    image_name]:  # also try exactly as given in label
            src = os.path.join(image_dir, ext)
            if os.path.exists(src):
                dst = os.path.join(out_dir, emotion_folder, ext)
                shutil.copy2(src, dst)
                if is_train:
                    copied_train += 1
                else:
                    copied_test  += 1
                break
        else:
            not_found += 1

    # Print summary
    print(f"\n{'='*60}")
    print(f"  ✅ DONE!")
    print(f"{'='*60}")
    print(f"  Training images: {copied_train}")
    print(f"  Test images:     {copied_test}")
    if not_found:
        print(f"  Not found:       {not_found}")

    print(f"\n  Breakdown per emotion (training):")
    for emotion_folder in EMOTION_MAP.values():
        path  = os.path.join(TRAIN_OUTPUT, emotion_folder)
        count = len(glob.glob(os.path.join(path, "*.jpg")))
        name  = emotion_folder.split("_", 1)[1].capitalize()
        bar   = "█" * (count // 100)
        print(f"    {name:<12} {count:>5}  {bar}")

    print(f"\n  Now run: python3 train_emotion_model.py")

if __name__ == "__main__":
    organize_images()
