"""
Crop labeled regions from YOLO-seg dataset (LabelMe JSON format)
into class-specific folders for SigCLIP2 training.

Output structure:
  cropped_classes/
    1xbet/
    admiralbet/
    eurobet/
    melbet/
    okvip/
"""

import json
import os
import glob
import cv2
import numpy as np
from pathlib import Path


DATASET_DIR = r"d:\SMVL\dataset_train_yoloseg"
OUTPUT_DIR = r"d:\SMVL\cropped_classes"
MIN_CROP_SIZE = 10  # Minimum width/height to keep a crop


def crop_regions():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    json_files = glob.glob(os.path.join(DATASET_DIR, "**", "*.json"), recursive=True)
    print(f"Found {len(json_files)} annotation files")

    class_counts = {}
    skipped = 0

    for jf in json_files:
        with open(jf, encoding="utf-8") as f:
            data = json.load(f)

        img_path = os.path.join(os.path.dirname(jf), data["imagePath"])
        if not os.path.exists(img_path):
            print(f"  [SKIP] Image not found: {img_path}")
            continue

        img = cv2.imread(img_path)
        if img is None:
            print(f"  [SKIP] Cannot read image: {img_path}")
            continue

        h, w = img.shape[:2]

        for i, shape in enumerate(data["shapes"]):
            label = shape["label"]
            points = np.array(shape["points"], dtype=np.float32)

            # Compute bounding box from polygon points
            x_min = max(0, int(np.floor(points[:, 0].min())))
            y_min = max(0, int(np.floor(points[:, 1].min())))
            x_max = min(w, int(np.ceil(points[:, 0].max())))
            y_max = min(h, int(np.ceil(points[:, 1].max())))

            crop_w = x_max - x_min
            crop_h = y_max - y_min

            if crop_w < MIN_CROP_SIZE or crop_h < MIN_CROP_SIZE:
                skipped += 1
                continue

            crop = img[y_min:y_max, x_min:x_max]

            # Create class folder
            class_dir = os.path.join(OUTPUT_DIR, label)
            os.makedirs(class_dir, exist_ok=True)

            # Generate unique filename
            base_name = Path(data["imagePath"]).stem
            crop_name = f"{base_name}_crop{i}.jpg"
            crop_path = os.path.join(class_dir, crop_name)

            cv2.imwrite(crop_path, crop)

            class_counts[label] = class_counts.get(label, 0) + 1

    print(f"\n=== Crop Summary ===")
    print(f"Skipped (too small): {skipped}")
    total = 0
    for cls in sorted(class_counts.keys()):
        count = class_counts[cls]
        total += count
        print(f"  {cls}: {count} crops")
    print(f"  TOTAL: {total} crops")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    crop_regions()
