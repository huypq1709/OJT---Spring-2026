# -*- coding: utf-8 -*-
"""
Append logos from 'failed_crops' directory to the existing FAISS index.
Parses brand names from filenames: frame_{idx}_box_{box_i}_{brand}_{sim}.jpg
"""

import os
import cv2
import numpy as np
import faiss
import shutil
from tqdm import tqdm
from compare_with_your_model import YourModelEmbedder

def append_failed_crops(
    failed_dir="failed_crops",
    model_path="model/siglip2_model.pth",
    index_path="logo_index.faiss",
    meta_path="logo_meta.npy",
    min_sim=0.0,
    exclude_unlogo=True,
    move_to_processed=True
):
    if not os.path.exists(index_path) or not os.path.exists(meta_path):
        print(f"Error: {index_path} or {meta_path} not found!")
        return

    if not os.path.exists(failed_dir):
        print(f"Error: Directory '{failed_dir}' not found!")
        return

    # Backup
    print("Creating backups...")
    shutil.copy(index_path, index_path + ".bak_failed")
    shutil.copy(meta_path, meta_path + ".bak_failed")

    print(f"Loading FAISS index: {index_path}")
    index = faiss.read_index(index_path)
    print(f"Loading Metadata: {meta_path}")
    meta = np.load(meta_path, allow_pickle=True).tolist()
    initial_count = index.ntotal

    print(f"Loading embedder: {model_path}")
    embedder = YourModelEmbedder(model_path=model_path, use_padding=True)

    files = [f for f in os.listdir(failed_dir) if f.lower().endswith((".jpg", ".png"))]
    if not files:
        print("No images found in failed_crops.")
        return

    new_embeddings = []
    new_names = []
    processed_count = 0
    skipped_count = 0
    brand_counts = {}

    processed_dir = os.path.join(failed_dir, "processed")
    if move_to_processed:
        os.makedirs(processed_dir, exist_ok=True)

    for filename in tqdm(files, desc="Processing failed crops"):
        # Pattern: frame_000657_box_0_melbet_0.86.jpg
        parts = filename.rsplit(".", 1)[0].split("_")
        
        if len(parts) < 6:
            print(f"Skipping {filename}: unexpected filename format.")
            skipped_count += 1
            continue
            
        # Parts: frame, idx, box, box_i, brand..., sim
        # Brand can have underscores, sim is the last part
        sim_str = parts[-1]
        try:
            sim = float(sim_str)
        except ValueError:
            print(f"Skipping {filename}: could not parse similarity.")
            skipped_count += 1
            continue

        brand_parts = parts[4:-1]
        brand = "_".join(brand_parts)

        if exclude_unlogo and (brand.lower() == "unlogo" or brand.lower() == "unknown"):
            skipped_count += 1
            continue

        if sim < min_sim:
            skipped_count += 1
            continue

        img_path = os.path.join(failed_dir, filename)
        img = cv2.imread(img_path)
        if img is None:
            skipped_count += 1
            continue

        # Generate embedding
        emb = embedder.get_embeddings_from_bgr_list([img])
        
        # Normalize (FAISS IndexFlatIP expects unit vectors)
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
            
        new_embeddings.append(emb)
        new_names.append(brand)
        brand_counts[brand] = brand_counts.get(brand, 0) + 1
        processed_count += 1

        if move_to_processed:
            shutil.move(img_path, os.path.join(processed_dir, filename))

    if not new_embeddings:
        print("No logos added to the database.")
        return

    # Add to index
    embeddings_stack = np.vstack(new_embeddings).astype(np.float32)
    index.add(embeddings_stack)
    meta.extend(new_names)

    # Save
    faiss.write_index(index, index_path)
    np.save(meta_path, np.array(meta, dtype=object))

    print("\n" + "="*30)
    print("SUCCESS: Logo Database Updated")
    print("="*30)
    print(f"Initial vectors: {initial_count}")
    print(f"Added vectors:   {processed_count}")
    print(f"Final vectors:   {index.ntotal}")
    print(f"Skipped files:   {skipped_count}")
    print("\nBrand Distribution added:")
    for b, c in sorted(brand_counts.items(), key=lambda x: -x[1]):
        print(f"  {b}: {c}")
    print("="*30)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="failed_crops")
    ap.add_argument("--min-sim", type=float, default=0.0)
    ap.add_argument("--keep-unlogo", action="store_true")
    args = ap.parse_args()

    append_failed_crops(
        failed_dir=args.dir,
        min_sim=args.min_sim,
        exclude_unlogo=not args.keep_unlogo
    )
