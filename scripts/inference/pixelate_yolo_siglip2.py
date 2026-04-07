# -*- coding: utf-8 -*-
"""
Pipeline: YOLO-seg detect+mask → SigLIP2 verify (FAISS) → pixelate mask.

Không dùng SAM — mask đến thẳng từ YOLO-seg.
SigLIP2 chỉ làm nhiệm vụ xác nhận lại crop có đúng là logo blocklist không.

Usage:
    python pixelate_yolo_siglip2.py
    python pixelate_yolo_siglip2.py --src public/videos/nha1.mp4 --dst output/result_yolo_siglip.mp4
    python pixelate_yolo_siglip2.py --src video.mp4 --dst out.mp4 --sim-threshold 0.32
"""

import argparse
import os
import time
from dataclasses import dataclass, field

import cv2
import numpy as np
import torch
import faiss
from PIL import Image
from collections import Counter
from ultralytics import YOLO

from compare_with_your_model import YourModelEmbedder


# ── Config defaults ──────────────────────────────────────────────
YOLO_CONF     = 0.25
SIM_THRESHOLD = 0.9
TOP_K         = 3
MIN_AREA      = 600
PIXEL_SIZE    = 14


@dataclass
class TimingStats:
    """Accumulator for per-stage timing."""
    yolo_ms: list = field(default_factory=list)
    crop_ms: list = field(default_factory=list)
    embed_ms: list = field(default_factory=list)
    faiss_ms: list = field(default_factory=list)
    pixelate_ms: list = field(default_factory=list)
    total_ms: list = field(default_factory=list)
    n_crops: list = field(default_factory=list)    # crops per frame
    n_pixelated: list = field(default_factory=list) # pixelated per frame

    def log(self, yolo, crop, embed, faiss_, pix, total, n_crop, n_pix):
        self.yolo_ms.append(yolo)
        self.crop_ms.append(crop)
        self.embed_ms.append(embed)
        self.faiss_ms.append(faiss_)
        self.pixelate_ms.append(pix)
        self.total_ms.append(total)
        self.n_crops.append(n_crop)
        self.n_pixelated.append(n_pix)

    @staticmethod
    def _avg(lst):
        return sum(lst) / len(lst) if lst else 0.0

    def summary(self):
        n = len(self.total_ms)
        return (
            f"  Frames timed   : {n}\n"
            f"  YOLO           : {self._avg(self.yolo_ms):7.2f} ms  (avg)\n"
            f"  Crop extract   : {self._avg(self.crop_ms):7.2f} ms  (avg)\n"
            f"  SigLIP2 embed  : {self._avg(self.embed_ms):7.2f} ms  (avg)\n"
            f"  FAISS search   : {self._avg(self.faiss_ms):7.2f} ms  (avg)\n"
            f"  Pixelate       : {self._avg(self.pixelate_ms):7.2f} ms  (avg)\n"
            f"  ───────────────────────────────────\n"
            f"  Total/frame    : {self._avg(self.total_ms):7.2f} ms  (avg)\n"
            f"  Crops/frame    : {self._avg(self.n_crops):7.1f}       (avg)\n"
            f"  Pixelated/frame: {self._avg(self.n_pixelated):7.1f}       (avg)\n"
            f"  Pipeline FPS   : {1000.0 / self._avg(self.total_ms):7.1f}" if self._avg(self.total_ms) > 0 else ""
        )


# ── Helpers ──────────────────────────────────────────────────────
def pixelate_binary_mask(frame, mask_hw, pixel_size=PIXEL_SIZE):
    """
    Pixelate vùng mask trên frame.
    mask_hw: float32 (H, W) từ YOLO masks.data — giá trị 0-1.
    """
    bin_mask = (mask_hw > 0.5).astype(np.uint8)
    ys, xs = np.where(bin_mask > 0)
    if len(xs) == 0:
        return frame
    x1, x2, y1, y2 = xs.min(), xs.max(), ys.min(), ys.max()
    roi = frame[y1:y2+1, x1:x2+1].copy()
    h, w = roi.shape[:2]
    if h < 2 or w < 2:
        return frame
    small = cv2.resize(roi, (max(1, w // pixel_size), max(1, h // pixel_size)),
                       interpolation=cv2.INTER_LINEAR)
    pix = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    roi_mask = bin_mask[y1:y2+1, x1:x2+1] > 0
    roi[roi_mask] = pix[roi_mask]
    frame[y1:y2+1, x1:x2+1] = roi
    return frame


class YoloSiglipPixelator:
    """YOLO-seg → SigLIP2 FAISS verify → pixelate pipeline."""

    def __init__(
        self,
        yolo_path="smvl_yolo11s_seg_kaggle/weights/best.pt",
        model_path="model/siglip2_model.pth",
        index_path="logo_index.faiss",
        meta_path="logo_meta.npy",
        blocklist=None,
        yolo_conf=YOLO_CONF,
        sim_threshold=SIM_THRESHOLD,
        top_k=TOP_K,
        min_area=MIN_AREA,
        pixel_size=PIXEL_SIZE,
        imgsz=640,
    ):
        self.yolo_conf = yolo_conf
        self.sim_threshold = sim_threshold
        self.top_k = top_k
        self.min_area = min_area
        self.pixel_size = pixel_size
        self.imgsz = imgsz
        self.blocklist = blocklist  # None = match tất cả brands trong index

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Device: {device}")

        # Load YOLO-seg
        print(f"\n📥 [1/3] Loading YOLO-seg: {yolo_path}")
        self.yolo = YOLO(yolo_path)
        print("   ✅ YOLO-seg loaded!")

        # Load SigLIP2 embedder
        print(f"\n📥 [2/3] Loading SigLIP2 embedder: {model_path}")
        self.embedder = YourModelEmbedder(
            model_path=model_path,
            device=device,
            use_padding=True,
        )
        print("   ✅ SigLIP2 loaded!")

        # Load FAISS index
        print(f"\n📥 [3/3] Loading FAISS index: {index_path}")
        self.index = faiss.read_index(index_path)
        self.meta = np.load(meta_path, allow_pickle=True)
        print(f"   ✅ FAISS loaded: {self.index.ntotal} vectors, {len(set(self.meta))} unique brands")

        if self.blocklist:
            print(f"   📋 Blocklist: {self.blocklist}")
        else:
            print(f"   📋 Blocklist: ALL brands in index")

    def process_frame(self, frame, timing: TimingStats = None,
                      crop_dir: str = None, frame_idx: int = 0):
        """YOLO-seg → SigLIP2 filter → pixelate matched masks."""
        t_start = time.perf_counter()

        # ── YOLO detection ──
        t0 = time.perf_counter()
        res = self.yolo.predict(
            frame,
            conf=self.yolo_conf,
            iou=0.45,
            imgsz=self.imgsz,
            verbose=False,
        )[0]
        t_yolo = (time.perf_counter() - t0) * 1000

        if res.masks is None or res.boxes is None:
            if timing:
                timing.log(t_yolo, 0, 0, 0, 0,
                           (time.perf_counter() - t_start) * 1000, 0, 0)
            return frame

        boxes = res.boxes.xyxy.cpu().numpy()   # (N, 4)
        masks = res.masks.data.cpu().numpy()   # (N, mH, mW)

        # Resize masks to frame size if needed
        fh, fw = frame.shape[:2]
        mh, mw = masks.shape[1], masks.shape[2]
        need_resize = (mh != fh or mw != fw)

        # ── Bước 1: crop bbox ──
        t0 = time.perf_counter()
        crops_bgr = []
        valid_idx = []
        crop_boxes = []  # lưu bbox cho debug

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            bw, bh = x2 - x1, y2 - y1

            if bw * bh < self.min_area or min(bw, bh) < 8:
                continue

            crop = frame[max(0, y1):y2, max(0, x1):x2]
            if crop.size == 0:
                continue

            crops_bgr.append(crop)
            valid_idx.append(i)
            crop_boxes.append((x1, y1, x2, y2))
        t_crop = (time.perf_counter() - t0) * 1000

        if not crops_bgr:
            if timing:
                timing.log(t_yolo, t_crop, 0, 0, 0,
                           (time.perf_counter() - t_start) * 1000, 0, 0)
            return frame

        # ── Bước 2: SigLIP2 embedding ──
        t0 = time.perf_counter()
        embeds = self.embedder.get_embeddings_from_bgr_list(crops_bgr)  # (M, D)
        embeds = embeds.astype(np.float32)
        t_embed = (time.perf_counter() - t0) * 1000

        # ── Bước 3: FAISS search ──
        t0 = time.perf_counter()
        sims, ids = self.index.search(embeds, self.top_k)  # (M, K)
        t_faiss = (time.perf_counter() - t0) * 1000

        # ── Bước 4: Filter + pixelate mask ──
        t0 = time.perf_counter()
        n_pix = 0
        for j, i in enumerate(valid_idx):
            top_sims = sims[j]
            top_names = [str(self.meta[k]) for k in ids[j] if k >= 0]

            # Cosine threshold
            if top_sims[0] < self.sim_threshold:
                continue

            # Vote top-K
            vote = Counter(
                name for name, s in zip(top_names, top_sims)
                if s >= self.sim_threshold
            )
            if not vote:
                continue
            best_name = vote.most_common(1)[0][0]

            # Blocklist check
            if self.blocklist is not None and best_name not in self.blocklist:
                continue

            # ── Save debug crop BEFORE pixelate ──
            if crop_dir is not None:
                before_crop = crops_bgr[j].copy()

            # Pixelate đúng mask của detection i
            mask_i = masks[i]
            if need_resize:
                mask_i = cv2.resize(
                    mask_i, (fw, fh), interpolation=cv2.INTER_LINEAR
                )
            frame = pixelate_binary_mask(frame, mask_i, self.pixel_size)
            n_pix += 1

            # ── Save debug crop AFTER pixelate ──
            if crop_dir is not None:
                x1, y1, x2, y2 = crop_boxes[j]
                after_crop = frame[max(0, y1):y2, max(0, x1):x2].copy()
                self._save_debug_crop(
                    crop_dir, frame_idx, n_pix,
                    before_crop, after_crop,
                    best_name, top_sims[0],
                    top_names, top_sims,
                )

        t_pix = (time.perf_counter() - t0) * 1000

        t_total = (time.perf_counter() - t_start) * 1000

        if timing:
            timing.log(t_yolo, t_crop, t_embed, t_faiss, t_pix,
                       t_total, len(crops_bgr), n_pix)

        return frame

    @staticmethod
    def _save_debug_crop(crop_dir, frame_idx, det_idx,
                         before, after, brand, sim,
                         top_names, top_sims):
        """Save side-by-side before/after crop with annotation."""
        h1, w1 = before.shape[:2]
        h2, w2 = after.shape[:2]
        # Resize after to match before height
        if h2 != h1:
            after = cv2.resize(after, (w1, h1))
            w2 = w1
        # Build side-by-side canvas
        gap = 4
        canvas_w = w1 + gap + w2
        label_h = 50
        canvas_h = h1 + label_h
        canvas = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)
        # Place before (left)
        canvas[0:h1, 0:w1] = before
        # Place after (right)
        canvas[0:h1, w1 + gap:w1 + gap + w2] = after
        # Separator line
        canvas[0:h1, w1:w1 + gap] = (0, 200, 0)
        # Labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(canvas, "BEFORE", (4, h1 + 18),
                    font, 0.45, (0, 0, 0), 1)
        cv2.putText(canvas, "AFTER", (w1 + gap + 4, h1 + 18),
                    font, 0.45, (0, 0, 200), 1)
        # Brand + sim info
        top3_str = " | ".join(
            f"{n}:{s:.3f}" for n, s in zip(top_names[:3], top_sims[:3])
        )
        cv2.putText(canvas, f"{brand} sim={sim:.3f}  [{top3_str}]",
                    (4, h1 + 40), font, 0.35, (80, 80, 80), 1)
        # Save
        fname = f"f{frame_idx:06d}_d{det_idx}_{brand}_{sim:.3f}.jpg"
        cv2.imwrite(os.path.join(crop_dir, fname), canvas,
                    [cv2.IMWRITE_JPEG_QUALITY, 90])

    def run_video(self, src, dst, max_frames=None, save_crops=False):
        """Process video: src → dst."""
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            print(f"❌ Cannot open video: {src}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if max_frames:
            total = min(total, max_frames)

        os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
        out = cv2.VideoWriter(dst, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        # Setup crop debug directory
        crop_dir = None
        if save_crops:
            dst_base = os.path.splitext(os.path.basename(dst))[0]
            crop_dir = os.path.join(os.path.dirname(dst) or ".",
                                    f"{dst_base}_crops")
            os.makedirs(crop_dir, exist_ok=True)
            print(f"   📂 Debug crops → {crop_dir}")

        print(f"\n🎬 Processing: {src}")
        print(f"   {w}x{h} @ {fps:.1f} fps, ~{total} frames")
        print(f"   Output: {dst}")
        print(f"   Config: conf={self.yolo_conf} sim={self.sim_threshold} topK={self.top_k} "
              f"minArea={self.min_area} pixel={self.pixel_size} imgsz={self.imgsz}")

        timing = TimingStats()
        fi = 0
        t0 = time.perf_counter()

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if max_frames and fi >= max_frames:
                break

            frame_out = self.process_frame(
                frame, timing=timing,
                crop_dir=crop_dir, frame_idx=fi,
            )
            out.write(frame_out)

            fi += 1
            if fi % 100 == 0:
                elapsed = time.perf_counter() - t0
                fps_actual = fi / elapsed if elapsed > 0 else 0
                # Show live timing breakdown
                avg_yolo  = timing._avg(timing.yolo_ms[-100:])
                avg_embed = timing._avg(timing.embed_ms[-100:])
                avg_faiss = timing._avg(timing.faiss_ms[-100:])
                avg_total = timing._avg(timing.total_ms[-100:])
                print(f"   {fi}/{total} frames ({fps_actual:.1f} fps) "
                      f"│ YOLO {avg_yolo:.1f}ms │ Embed {avg_embed:.1f}ms "
                      f"│ FAISS {avg_faiss:.1f}ms │ Total {avg_total:.1f}ms")

        cap.release()
        out.release()

        elapsed = time.perf_counter() - t0
        fps_actual = fi / elapsed if elapsed > 0 else 0

        print(f"\n{'='*60}")
        print(f"✅ Done: {fi} frames in {elapsed:.1f}s ({fps_actual:.1f} fps)")
        print(f"   → {dst}")
        if crop_dir:
            n_crops = len([f for f in os.listdir(crop_dir) if f.endswith(".jpg")])
            print(f"   📂 Debug crops: {n_crops} files → {crop_dir}")
        print(f"\n⏱  TIMING BREAKDOWN (per frame):")
        print(f"{'='*60}")
        print(timing.summary())
        print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="YOLO-seg + SigLIP2 FAISS → Pixelate blocklist logos"
    )
    parser.add_argument("--src", default="public/videos/nha1.mp4", help="Input video")
    parser.add_argument("--dst", default="output/result_yolo_siglip.mp4", help="Output video")
    parser.add_argument("--yolo", default="smvl_yolo11s_seg_kaggle/weights/best.pt",
                        help="YOLO-seg model path")
    parser.add_argument("--model", default="model/siglip2_model.pth",
                        help="SigLIP2 model path")
    parser.add_argument("--index", default="logo_index.faiss", help="FAISS index")
    parser.add_argument("--meta", default="logo_meta.npy", help="FAISS metadata")
    parser.add_argument("--yolo-conf", type=float, default=YOLO_CONF)
    parser.add_argument("--sim-threshold", type=float, default=SIM_THRESHOLD)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--min-area", type=int, default=MIN_AREA)
    parser.add_argument("--pixel-size", type=int, default=PIXEL_SIZE)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Max frames to process (None = all)")
    parser.add_argument("--blocklist", nargs="*", default=None,
                        help="Brand names to pixelate. None = all brands in index")
    parser.add_argument("--save-crops", action="store_true",
                        help="Save before/after debug crops for QA")
    args = parser.parse_args()

    blocklist = set(args.blocklist) if args.blocklist else None

    pixelator = YoloSiglipPixelator(
        yolo_path=args.yolo,
        model_path=args.model,
        index_path=args.index,
        meta_path=args.meta,
        blocklist=blocklist,
        yolo_conf=args.yolo_conf,
        sim_threshold=args.sim_threshold,
        top_k=args.top_k,
        min_area=args.min_area,
        pixel_size=args.pixel_size,
        imgsz=args.imgsz,
    )

    pixelator.run_video(args.src, args.dst,
                        max_frames=args.max_frames,
                        save_crops=args.save_crops)


if __name__ == "__main__":
    main()
