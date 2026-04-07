# -*- coding: utf-8 -*-
"""
Tìm và lưu các crop bị phân loại sai từ eval SigLIP2.
Memory-friendly: encode theo batch từ image, không giữ tất cả crops trong RAM.
"""

import os, random, time, gc
import xml.etree.ElementTree as ET
from collections import defaultdict

import cv2
import numpy as np
import faiss
from tqdm import tqdm

from compare_with_your_model import YourModelEmbedder


def collect_all_crops(ann_dir, img_dir, max_files=None):
    xml_files = sorted([f for f in os.listdir(ann_dir) if f.endswith(".xml")])
    if max_files:
        xml_files = xml_files[:max_files]
    crops_info = []
    for xml_file in tqdm(xml_files, desc="Parsing"):
        xml_path = os.path.join(ann_dir, xml_file)
        tree = ET.parse(xml_path)
        root = tree.getroot()
        filename = root.findtext("filename")
        img_path = os.path.join(img_dir, filename)
        if not os.path.isfile(img_path):
            stem = os.path.splitext(filename)[0]
            for ext in [".jpeg", ".jpg", ".png"]:
                c = os.path.join(img_dir, stem + ext)
                if os.path.isfile(c):
                    img_path = c
                    break
        if not os.path.isfile(img_path):
            continue
        for obj in root.findall("object"):
            name = obj.findtext("name")
            bnd = obj.find("bndbox")
            x1, y1 = int(bnd.findtext("xmin")), int(bnd.findtext("ymin"))
            x2, y2 = int(bnd.findtext("xmax")), int(bnd.findtext("ymax"))
            if (x2 - x1) < 8 or (y2 - y1) < 8:
                continue
            crops_info.append((name, img_path, x1, y1, x2, y2))
    return crops_info


def stratified_split(crops_info, gallery_per_brand=150, query_per_brand=10, seed=42):
    rng = random.Random(seed)
    by_brand = defaultdict(list)
    for item in crops_info:
        by_brand[item[0]].append(item)
    gallery, query = [], []
    for brand, items in sorted(by_brand.items()):
        rng.shuffle(items)
        n_query = min(query_per_brand, max(2, len(items) // 5))
        n_gallery = min(gallery_per_brand, len(items) - n_query)
        if n_gallery < 3 or n_query < 1:
            n_query = min(n_query, len(items) // 2)
            n_gallery = len(items) - n_query
        query.extend(items[:n_query])
        gallery.extend(items[n_query:n_query + n_gallery])
    return gallery, query


def crop_from_info(info):
    """Load and crop a single bbox. Returns BGR numpy or None."""
    name, img_path, x1, y1, x2, y2 = info
    img = cv2.imread(img_path)
    if img is None:
        return None
    h, w = img.shape[:2]
    crop = img[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
    return crop if crop.size > 0 else None


def encode_crops_streaming(crops_info, embedder, batch_size=32):
    """Encode crops without holding all in memory. Returns (N, D) embeddings + labels."""
    all_embs = []
    all_labels = []

    # Group by image for efficiency
    by_image = defaultdict(list)
    for idx, (name, img_path, x1, y1, x2, y2) in enumerate(crops_info):
        by_image[img_path].append((idx, name, x1, y1, x2, y2))

    batch_crops = []
    batch_labels = []

    for img_path, items in tqdm(by_image.items(), desc="Encoding"):
        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]

        for idx, name, x1, y1, x2, y2 in items:
            crop = img[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
            if crop.size == 0:
                continue
            batch_crops.append(crop)
            batch_labels.append(name)

            if len(batch_crops) >= batch_size:
                embs = embedder.get_embeddings_from_bgr_list(batch_crops)
                all_embs.append(embs)
                all_labels.extend(batch_labels)
                batch_crops = []
                batch_labels = []

        del img
        gc.collect()

    if batch_crops:
        embs = embedder.get_embeddings_from_bgr_list(batch_crops)
        all_embs.append(embs)
        all_labels.extend(batch_labels)

    return np.vstack(all_embs).astype(np.float32), all_labels


def main():
    ann_dir = "SMVL_dataset/SMVL_Annotations/Annotations"
    img_dir = "SMVL_dataset/SMVL_JPEGImages/JPEGImages"
    out_dir = "output/misclassified"
    os.makedirs(out_dir, exist_ok=True)

    TOP_K = 5

    # Step 1: Parse + split (same seed as eval)
    print("Step 1: Collecting crops...")
    crops_info = collect_all_crops(ann_dir, img_dir, max_files=2000)
    gallery_info, query_info = stratified_split(crops_info)
    del crops_info
    gc.collect()
    print(f"   Gallery: {len(gallery_info)}, Query: {len(query_info)}")

    # Step 2: Load model
    print("\nStep 2: Loading SigLIP2...")
    embedder = YourModelEmbedder(model_path="model/siglip2_model.pth", use_padding=True)

    # Step 3: Encode gallery (streaming)
    print("\nStep 3: Encoding gallery...")
    gallery_embs, gallery_labels = encode_crops_streaming(gallery_info, embedder)
    print(f"   Gallery embeddings: {gallery_embs.shape}")

    # Step 4: Encode query (streaming)
    print("\nStep 4: Encoding queries...")
    query_embs, query_labels = encode_crops_streaming(query_info, embedder)
    print(f"   Query embeddings: {query_embs.shape}")

    # Step 5: FAISS search
    print("\nStep 5: FAISS search...")
    dim = gallery_embs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(gallery_embs)
    sims, ids = index.search(query_embs, TOP_K)

    # Step 6: Find and visualize misclassified
    print("\nStep 6: Finding misclassified...")
    n_wrong = 0
    for i in range(len(query_labels)):
        true_label = query_labels[i]
        pred_label = gallery_labels[ids[i, 0]]

        if true_label == pred_label:
            continue

        n_wrong += 1
        print(f"\n❌ Case {n_wrong}: TRUE={true_label} → PRED={pred_label} (sim={sims[i,0]:.4f})")

        # Load query crop on the fly
        query_crop = crop_from_info(query_info[i])
        if query_crop is None:
            print("   ⚠️ Cannot load query crop")
            continue

        q_path = os.path.join(out_dir, f"wrong_{n_wrong}_query_{true_label}.jpg")
        cv2.imwrite(q_path, query_crop)

        # Load top-K gallery matches on the fly
        match_crops = []
        match_labels = []
        match_sims = []
        for k in range(TOP_K):
            g_idx = ids[i, k]
            g_label = gallery_labels[g_idx]
            g_sim = sims[i, k]
            g_crop = crop_from_info(gallery_info[g_idx])
            marker = "WRONG" if g_label != true_label else "OK"

            if g_crop is not None:
                g_path = os.path.join(out_dir,
                    f"wrong_{n_wrong}_top{k+1}_{g_label}_sim{g_sim:.3f}_{marker}.jpg")
                cv2.imwrite(g_path, g_crop)
                match_crops.append(g_crop)
                match_labels.append(f"Top-{k+1} {'X' if marker=='WRONG' else 'V'}\n{g_label}\nsim={g_sim:.3f}")
                match_sims.append(g_sim)
                print(f"   Top-{k+1}: {g_label} (sim={g_sim:.4f}) [{marker}]")

        # Build comparison panel
        panel_crops = [query_crop] + match_crops[:3]
        panel_labels = [f"QUERY\n{true_label}"] + match_labels[:3]

        target_h = 128
        resized = []
        for crop in panel_crops:
            h, w = crop.shape[:2]
            scale = target_h / h
            new_w = max(1, int(w * scale))
            resized.append(cv2.resize(crop, (new_w, target_h)))

        labeled = []
        for crop, label in zip(resized, panel_labels):
            h, w = crop.shape[:2]
            label_h = 50
            canvas = np.ones((h + label_h, w, 3), dtype=np.uint8) * 255
            canvas[:h, :w] = crop
            for li, line in enumerate(label.split("\n")):
                cv2.putText(canvas, line, (4, h + 15 + li * 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
            labeled.append(canvas)

        max_h = max(c.shape[0] for c in labeled)
        padded = []
        for c in labeled:
            if c.shape[0] < max_h:
                pad = np.ones((max_h - c.shape[0], c.shape[1], 3), dtype=np.uint8) * 255
                c = np.vstack([c, pad])
            padded.append(c)

        panel = np.hstack(padded)
        panel_path = os.path.join(out_dir, f"wrong_{n_wrong}_panel_{true_label}_vs_{pred_label}.jpg")
        cv2.imwrite(panel_path, panel)
        print(f"   Panel: {panel_path}")

    if n_wrong == 0:
        print("\n✅ No misclassifications found!")
    else:
        print(f"\n📊 Total misclassified: {n_wrong} / {len(query_labels)}")
        print(f"   Saved to: {out_dir}/")


if __name__ == "__main__":
    main()
