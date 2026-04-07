# Chạy pipeline SAM từ sam-2.ipynb - đường dẫn local Windows
import os
import cv2
import json
import time
import psutil
import torch
import shutil
import zipfile
import numpy as np
import random
import pandas as pd
import xml.etree.ElementTree as ET
from tqdm import tqdm
from datetime import datetime
from ultralytics import SAM

MODEL_NAME = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sam3_b.pt")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SAM(MODEL_NAME)
print("Model:", MODEL_NAME, "| Device:", device)

def log_resource():
    cpu = psutil.cpu_percent()
    ram = psutil.virtual_memory().percent
    gpu = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
    return cpu, ram, gpu

def read_bbox(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    boxes = []
    for obj in root.findall("object"):
        name = obj.find("name").text
        box = obj.find("bndbox")
        xmin, ymin = int(box.find("xmin").text), int(box.find("ymin").text)
        xmax, ymax = int(box.find("xmax").text), int(box.find("ymax").text)
        boxes.append((name, [xmin, ymin, xmax, ymax]))
    return boxes

def mask_to_polygon(mask):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [c.squeeze().tolist() for c in contours if len(c) >= 3]

def polygon_to_yolo_line(poly, cls_id, img_w, img_h):
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    x_c = ((x_min + x_max) / 2.0) / img_w
    y_c = ((y_min + y_max) / 2.0) / img_h
    bw = (x_max - x_min) / img_w
    bh = (y_max - y_min) / img_h
    norm_poly = []
    for x, y in poly:
        norm_poly.append(x / img_w)
        norm_poly.append(y / img_h)
    parts = [str(cls_id),
             f"{x_c:.6f}", f"{y_c:.6f}", f"{bw:.6f}", f"{bh:.6f}"] + [f"{v:.6f}" for v in norm_poly]
    return " ".join(parts)

def check_mask(mask, bbox):
    x1, y1, x2, y2 = bbox
    bbox_area = (x2 - x1) * (y2 - y1)
    if bbox_area == 0: return False
    return 0.3 < np.sum(mask) / bbox_area < 1.5

def create_json(image_name, image_shape, shapes):
    return {"version":"5.0.1","flags":{},"shapes":shapes,"imagePath":image_name,"imageData":None,"imageHeight":image_shape[0],"imageWidth":image_shape[1]}

def list_images(IMAGE_DIR):
    return [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

def process_dataset(IMAGE_DIR, XML_DIR, OUTPUT_ROOT, pbar=None):
    DATASET_NAME = os.path.basename(os.path.dirname(IMAGE_DIR))
    OUTPUT_DIR = os.path.join(OUTPUT_ROOT, DATASET_NAME)
    GOOD_DIR = os.path.join(OUTPUT_DIR, "good")
    BAD_DIR = os.path.join(OUTPUT_DIR, "bad_case")
    MASK_DIR = os.path.join(OUTPUT_DIR, "masks")
    YOLO_DIR = os.path.join(OUTPUT_DIR, "labels_yolo_seg")
    YOLO_DS_ROOT = os.path.join(OUTPUT_DIR, "yolo_dataset")
    IM_TRAIN = os.path.join(YOLO_DS_ROOT, "images", "train")
    IM_VAL = os.path.join(YOLO_DS_ROOT, "images", "val")
    LB_TRAIN = os.path.join(YOLO_DS_ROOT, "labels", "train")
    LB_VAL = os.path.join(YOLO_DS_ROOT, "labels", "val")
    # Lưu file JSON chung trong thư mục GOOD_DIR
    JSON_DIR = GOOD_DIR
    for d in [GOOD_DIR, BAD_DIR, MASK_DIR, YOLO_DIR, IM_TRAIN, IM_VAL, LB_TRAIN, LB_VAL]:
        os.makedirs(d, exist_ok=True)

    images = list_images(IMAGE_DIR)
    good_count = bad_count = 0
    resource_logs = []
    class_map = {}

    print("\n==================================================")
    print("Processing:", IMAGE_DIR)
    print("Total images:", len(images))

    # Nếu có thanh tiến độ chung (pbar) thì không dùng tqdm riêng
    iterator = images if pbar is not None else tqdm(images, desc=DATASET_NAME)

    for img_name in iterator:
        start = time.time()
        img_path = os.path.join(IMAGE_DIR, img_name)
        xml_path = os.path.join(XML_DIR, img_name.replace(".jpg", ".xml").replace(".jpeg", ".xml").replace(".png", ".xml"))
        if not os.path.exists(xml_path): continue
        image = cv2.imread(img_path)
        if image is None: continue
        h, w = image.shape[:2]
        boxes = read_bbox(xml_path)
        labels, bboxes = [b[0] for b in boxes], [b[1] for b in boxes]
        bad_flag, shapes, yolo_lines = False, [], []

        with torch.no_grad():
            for i, (label, bbox) in enumerate(zip(labels, bboxes)):
                x1, y1, x2, y2 = bbox
                results = model.predict(source=img_path, bboxes=[bbox], verbose=False)
                if len(results) == 0 or results[0].masks is None: bad_flag = True; break
                masks_stack = results[0].masks.data.cpu().numpy()
                best_mask, best_score = None, -1.0
                for m in masks_stack:
                    m_bin = (m > 0.5).astype(np.uint8)
                    bbox_area = max((x2-x1)*(y2-y1), 1)
                    ratio = m_bin.sum() / bbox_area
                    score = -abs(1.0 - ratio) + 0.01 * ratio
                    if score > best_score: best_score, best_mask = score, m_bin
                if best_mask is None: bad_flag = True; break
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
                closed = cv2.morphologyEx((best_mask * 255).astype(np.uint8), cv2.MORPH_CLOSE, kernel)
                contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not contours: bad_flag = True; break
                largest = max(contours, key=cv2.contourArea)
                final_mask = np.zeros_like(best_mask, dtype=np.uint8)
                cv2.fillPoly(final_mask, [largest], 1)
                mask = np.zeros_like(final_mask, dtype=np.uint8)
                mask[y1:y2, x1:x2] = final_mask[y1:y2, x1:x2]
                if not check_mask(mask, bbox): bad_flag = True; break
                for poly in mask_to_polygon(mask):
                    shapes.append({"label":label,"points":poly,"group_id":None,"shape_type":"polygon","flags":{}})
                    if label not in class_map:
                        class_map[label] = len(class_map)
                    cls_id = class_map[label]
                    yolo_lines.append(polygon_to_yolo_line(poly, cls_id, w, h))
                cv2.imwrite(os.path.join(MASK_DIR, f"{os.path.splitext(img_name)[0]}_{i}.png"), (mask * 255).astype(np.uint8))

        cpu, ram, gpu = log_resource()
        resource_logs.append({"image":img_name,"time":time.time()-start,"cpu":cpu,"ram":ram,"gpu":gpu})

        if pbar is not None:
            pbar.update(1)

        if bad_flag or not yolo_lines:
            shutil.copy(img_path, os.path.join(BAD_DIR, img_name))
            bad_count += 1
            continue
        json_data = create_json(img_name, (h, w), shapes)
        with open(os.path.join(JSON_DIR, os.path.splitext(img_name)[0] + ".json"), "w") as f:
            json.dump(json_data, f, indent=2)
        # Ghi nhãn theo định dạng YOLOv11-seg
        base_name = os.path.splitext(img_name)[0]
        yolo_txt_path = os.path.join(YOLO_DIR, base_name + ".txt")
        with open(yolo_txt_path, "w") as f:
            f.write("\n".join(yolo_lines))
        # Chia train/val đơn giản theo tỉ lệ
        is_train = random.random() < 0.8
        if is_train:
            shutil.copy(img_path, os.path.join(IM_TRAIN, img_name))
            shutil.copy(yolo_txt_path, os.path.join(LB_TRAIN, base_name + ".txt"))
        else:
            shutil.copy(img_path, os.path.join(IM_VAL, img_name))
            shutil.copy(yolo_txt_path, os.path.join(LB_VAL, base_name + ".txt"))
        shutil.copy(img_path, os.path.join(GOOD_DIR, img_name))
        good_count += 1

    if resource_logs:
        df = pd.DataFrame(resource_logs)
        csv_path = os.path.join(OUTPUT_DIR, "resource_log.csv")
        df.to_csv(csv_path, index=False)
        print("\n========== SUMMARY ==========")
        print("Total:", len(df), "| Good:", good_count, "| Bad:", bad_count)
        print("Avg time:", round(df["time"].mean(), 3), "s | Avg GPU:", round(df["gpu"].mean(), 3), "GB")
        print("Saved:", csv_path)

    # Lưu file classes.txt cho YOLO (theo thứ tự id)
    if class_map:
        classes_path = os.path.join(OUTPUT_DIR, "classes.txt")
        inv = sorted(class_map.items(), key=lambda x: x[1])
        with open(classes_path, "w", encoding="utf-8") as f:
            for name, _id in inv:
                f.write(f"{name}\n")
        print("Saved classes file:", classes_path)

    # Không zip lại kết quả, chỉ giữ thư mục OUTPUT_DIR (masks, json, yolo, log, good/bad nếu có)
    print("PIPELINE FINISHED:", DATASET_NAME)

if __name__ == "__main__":
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    # Dataset SMVL: ảnh và annotation theo kiểu PASCAL VOC
    BASE = os.path.join(PROJECT_ROOT, "SMVL_dataset")
    # Thư mục đúng theo cấu trúc bạn đang có: SMVL_JPEGImages/JPEGImages và SMVL_Annotations/Annotations
    IMAGE_DIR = os.path.join(BASE, "SMVL_JPEGImages", "JPEGImages")
    XML_DIR = os.path.join(BASE, "SMVL_Annotations", "Annotations")
    OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "SMVL_logo_seg_results_sam3b_yolo")
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    if not (os.path.isdir(IMAGE_DIR) and os.path.isdir(XML_DIR)):
        print("Khong tim thay SMVL_Images/Images hoac SMVL_Annotations/Annotations trong:", BASE)
    else:
        total_images = len(list_images(IMAGE_DIR))
        with tqdm(total=total_images, desc="Tien do SMVL_dataset") as pbar:
            process_dataset(IMAGE_DIR, XML_DIR, OUTPUT_ROOT, pbar=pbar)
