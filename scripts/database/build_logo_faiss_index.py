# -*- coding: utf-8 -*-
"""
Build FAISS index từ bounding-box crops trong SMVL_dataset (VOC XML).
Dùng YourModelEmbedder (open_clip SigLIP2-256, 512-d) để encode từng crop,
lưu ra logo_index.faiss + logo_meta.npy.

Usage:
    python build_logo_faiss_index.py
    python build_logo_faiss_index.py --ann-dir SMVL_dataset/SMVL_Annotations/Annotations \
                                     --img-dir SMVL_dataset/SMVL_JPEGImages/JPEGImages \
                                     --out-index logo_index.faiss \
                                     --out-meta logo_meta.npy
"""

import argparse
import os
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import faiss
from tqdm import tqdm

from compare_with_your_model import YourModelEmbedder


def parse_voc_xml(xml_path):
    """Parse VOC XML, trả về list (name, xmin, ymin, xmax, ymax)."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    filename = root.findtext("filename")
    objects = []
    for obj in root.findall("object"):
        name = obj.findtext("name")
        bnd = obj.find("bndbox")
        xmin = int(bnd.findtext("xmin"))
        ymin = int(bnd.findtext("ymin"))
        xmax = int(bnd.findtext("xmax"))
        ymax = int(bnd.findtext("ymax"))
        objects.append((name, xmin, ymin, xmax, ymax))
    return filename, objects


def build_index(ann_dir, img_dir, model_path, out_index, out_meta, blocklist=None):
    """
    Crop tất cả bbox từ annotations → encode SigLIP2 → build FAISS IndexFlatIP.

    Args:
        ann_dir: Thư mục chứa XML annotations
        img_dir: Thư mục chứa JPEG images
        model_path: Path tới SigLIP2 weights
        out_index: Path output FAISS index
        out_meta: Path output metadata (brand names per vector)
        blocklist: Nếu set, chỉ index các brand trong set này. None = index tất cả.
    """
    embedder = YourModelEmbedder(model_path=model_path, use_padding=True)

    xml_files = sorted([f for f in os.listdir(ann_dir) if f.endswith(".xml")])
    print(f"\n📂 Found {len(xml_files)} annotation files")

    all_embeddings = []
    all_names = []
    brand_counts = {}
    skipped = 0

    for xml_file in tqdm(xml_files, desc="Processing annotations"):
        xml_path = os.path.join(ann_dir, xml_file)
        filename, objects = parse_voc_xml(xml_path)

        # Tìm ảnh
        img_path = os.path.join(img_dir, filename)
        if not os.path.isfile(img_path):
            # Thử các extension khác
            stem = os.path.splitext(filename)[0]
            for ext in [".jpeg", ".jpg", ".png"]:
                candidate = os.path.join(img_dir, stem + ext)
                if os.path.isfile(candidate):
                    img_path = candidate
                    break

        if not os.path.isfile(img_path):
            skipped += 1
            continue

        img = cv2.imread(img_path)
        if img is None:
            skipped += 1
            continue

        h, w = img.shape[:2]

        # Collect crops cho batch processing
        batch_crops = []
        batch_names = []

        for name, xmin, ymin, xmax, ymax in objects:
            # Filter blocklist nếu có
            if blocklist is not None and name not in blocklist:
                continue

            # Clip to image bounds
            xmin = max(0, xmin)
            ymin = max(0, ymin)
            xmax = min(w, xmax)
            ymax = min(h, ymax)

            bw, bh = xmax - xmin, ymax - ymin
            if bw < 8 or bh < 8 or bw * bh < 200:
                continue

            crop = img[ymin:ymax, xmin:xmax]
            if crop.size == 0:
                continue

            batch_crops.append(crop)
            batch_names.append(name)

        if not batch_crops:
            continue

        # Batch encode
        embeds = embedder.get_embeddings_from_bgr_list(batch_crops)  # (N, 512)

        all_embeddings.append(embeds)
        all_names.extend(batch_names)

        for name in batch_names:
            brand_counts[name] = brand_counts.get(name, 0) + 1

    if not all_embeddings:
        print("❌ Không tìm thấy crop nào!")
        return

    # Stack all embeddings
    all_embeddings = np.vstack(all_embeddings).astype(np.float32)  # (M, 512)

    # L2 normalize (đã normalize trong embedder nhưng double-check)
    norms = np.linalg.norm(all_embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    all_embeddings = all_embeddings / norms

    print(f"\n📊 Statistics:")
    print(f"   Total crops embedded: {all_embeddings.shape[0]}")
    print(f"   Embedding dim: {all_embeddings.shape[1]}")
    print(f"   Unique brands: {len(brand_counts)}")
    print(f"   Skipped files: {skipped}")
    print(f"\n   Brand distribution:")
    for name, count in sorted(brand_counts.items(), key=lambda x: -x[1]):
        print(f"     {name}: {count}")

    # Build FAISS index (Inner Product = cosine trên L2-normalized vectors)
    dim = all_embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(all_embeddings)

    # Save
    faiss.write_index(index, out_index)
    np.save(out_meta, np.array(all_names, dtype=object))

    print(f"\n✅ Saved:")
    print(f"   Index: {out_index} ({index.ntotal} vectors, dim={dim})")
    print(f"   Meta:  {out_meta}")


def main():
    parser = argparse.ArgumentParser(description="Build FAISS index từ SMVL_dataset bbox crops")
    parser.add_argument("--ann-dir", default="SMVL_dataset/SMVL_Annotations/Annotations",
                        help="Thư mục chứa VOC XML annotations")
    parser.add_argument("--img-dir", default="SMVL_dataset/SMVL_JPEGImages/JPEGImages",
                        help="Thư mục chứa JPEG images")
    parser.add_argument("--model", default="model/siglip2_model.pth",
                        help="Path to SigLIP2 model weights")
    parser.add_argument("--out-index", default="logo_index.faiss",
                        help="Output FAISS index path")
    parser.add_argument("--out-meta", default="logo_meta.npy",
                        help="Output metadata path")
    parser.add_argument("--blocklist-only", action="store_true",
                        help="Chỉ index brands trong blocklist.yaml")
    args = parser.parse_args()

    blocklist = None
    if args.blocklist_only:
        try:
            import yaml
            with open("blocklist.yaml", "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            bl = cfg.get("blocklist", {})
            if bl.get("enabled"):
                blocklist = {c["name"] for c in bl.get("classes", [])}
                print(f"📋 Blocklist filter: {blocklist}")
        except Exception as e:
            print(f"⚠️ Cannot load blocklist.yaml: {e}. Indexing ALL brands.")

    build_index(
        ann_dir=args.ann_dir,
        img_dir=args.img_dir,
        model_path=args.model,
        out_index=args.out_index,
        out_meta=args.out_meta,
        blocklist=blocklist,
    )


if __name__ == "__main__":
    main()
