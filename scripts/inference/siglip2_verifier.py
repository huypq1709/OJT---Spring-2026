# -*- coding: utf-8 -*-
"""
SigLIP2 embedding + FAISS vector search để lọc false-positive sau YOLO-seg.
Hỗ trợ load weights từ siglip2_model.pt (state_dict hoặc checkpoint).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

try:
    import faiss
except ImportError:
    faiss = None


def _load_siglip2_model(
    model_name_or_path: str,
    weights_pt: Optional[str],
    device: str,
):
    """Load SigLIP2 từ HF hoặc thư mục local; merge state_dict từ .pt nếu có."""
    try:
        from transformers import Siglip2Model, Siglip2Processor
    except ImportError:
        from transformers import AutoModel, AutoProcessor as Siglip2Processor  # type: ignore
        Siglip2Model = None  # type: ignore

    processor = None
    model = None

    if Siglip2Model is not None:
        try:
            processor = Siglip2Processor.from_pretrained(model_name_or_path)
            model = Siglip2Model.from_pretrained(model_name_or_path)
        except Exception:
            processor = None
            model = None

    if model is None:
        from transformers import AutoModel, AutoProcessor

        processor = AutoProcessor.from_pretrained(model_name_or_path)
        model = AutoModel.from_pretrained(model_name_or_path)

    model = model.to(device)
    model.eval()

    if weights_pt and os.path.isfile(weights_pt):
        ckpt = torch.load(weights_pt, map_location=device, weights_only=False)
        if isinstance(ckpt, dict):
            if "state_dict" in ckpt:
                sd = ckpt["state_dict"]
            elif "model" in ckpt:
                sd = ckpt["model"]
            else:
                sd = ckpt
            missing, unexpected = model.load_state_dict(sd, strict=False)
            print(f"[SigLIP2] Loaded {weights_pt} | missing={len(missing)} unexpected={len(unexpected)}")
        else:
            print("[SigLIP2] Warning: checkpoint không phải dict, bỏ qua load_state_dict")

    return model, processor


@torch.no_grad()
def image_embed(
    model,
    processor,
    images_rgb: List[np.ndarray],
    device: str,
) -> np.ndarray:
    """
    images_rgb: list ảnh HWC RGB uint8 (crop từ OpenCV sau cv2.cvtColor BGR->RGB)
    Trả về (N, D) L2-normalized numpy float32
    """
    from PIL import Image

    pil_list = [Image.fromarray(im) for im in images_rgb]
    inputs = processor(images=pil_list, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    if hasattr(model, "get_image_features"):
        try:
            feats = model.get_image_features(**inputs)
        except TypeError:
            pv = inputs.get("pixel_values")
            feats = model.get_image_features(pixel_values=pv) if pv is not None else model.get_image_features(**inputs)
    else:
        out = model(**inputs)
        if hasattr(out, "image_embeds"):
            feats = out.image_embeds
        elif hasattr(out, "pooler_output"):
            feats = out.pooler_output
        else:
            feats = out.last_hidden_state.mean(dim=1)

    feats = F.normalize(feats.float(), dim=-1)
    return feats.cpu().numpy().astype(np.float32)


@dataclass
class FilterLayerConfig:
    """6+ tầng lọc (geometry + YOLO conf + SigLIP + FAISS + blur + edge)."""

    enabled: bool = True
    model_name: str = "google/siglip2-base-patch16-384"
    weights_pt: str = "siglip2_model.pt"
    index_dir: str = "siglip_index"
    # L1: diện tích bbox / diện tích frame
    min_area_ratio: float = 1e-5
    max_area_ratio: float = 0.45
    # L2: tỉ lệ khung (w/h)
    min_aspect: float = 0.05
    max_aspect: float = 25.0
    # L3: độ "solid" của mask (diện tích mask / diện tích bbox)
    min_mask_fill_ratio: float = 0.08
    # L4: YOLO confidence tối thiểu (áp thêm sau conf inference)
    min_yolo_conf: float = 0.2
    # L5: cosine similarity với prototype class (nếu có vector trong index)
    min_cosine_class: float = 0.18
    # L6: FAISS: khoảng cách L2 trên vector đã chuẩn hóa -> dùng inner product (cosine) qua ngưỡng
    min_faiss_score: float = 0.22
    faiss_topk: int = 3
    # L7: Laplacian variance (blur) — quá mờ có thể bỏ qua
    min_laplacian_var: float = 5.0
    # L8: mật độ cạnh trong bbox (tránh khối đồng nhất)
    min_edge_density: float = 0.01


class SiglipFaissVerifier:
    """
    Mỗi class_id có một FAISS index (inner product = cosine trên vector đã L2 norm).
    """

    def __init__(
        self,
        model_name_or_path: str = "google/siglip2-base-patch16-384",
        weights_pt: Optional[str] = None,
        device: Optional[str] = None,
        index_dir: Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.processor = _load_siglip2_model(
            model_name_or_path, weights_pt, self.device
        )
        self.dim: Optional[int] = None
        self._indices: Dict[int, "faiss.Index"] = {}
        self._class_ids_in_index: Dict[int, np.ndarray] = {}

        if index_dir and os.path.isdir(index_dir):
            self.load_indices(index_dir)

    def reload_indices(self, index_dir: str):
        self._indices.clear()
        self._class_ids_in_index.clear()
        if index_dir and os.path.isdir(index_dir):
            self.load_indices(index_dir)

    def _ensure_dim(self, x: np.ndarray):
        d = x.shape[-1]
        if self.dim is None:
            self.dim = d
        elif self.dim != d:
            raise ValueError(f"Embedding dim mismatch: {self.dim} vs {d}")

    def build_index_for_class(
        self,
        class_id: int,
        embeddings: np.ndarray,
    ):
        """embeddings: (N, D) đã L2 normalize."""
        if faiss is None:
            raise RuntimeError("Cần cài faiss-cpu: pip install faiss-cpu")
        self._ensure_dim(embeddings)
        embeddings = embeddings.astype(np.float32)
        n, d = embeddings.shape
        index = faiss.IndexFlatIP(d)
        index.add(embeddings)
        self._indices[class_id] = index

    def load_indices(self, index_dir: str):
        """Load class_<id>.index + class_<id>.meta.npy (optional)."""
        if faiss is None:
            return
        for fn in os.listdir(index_dir):
            if not fn.endswith(".index"):
                continue
            if fn.startswith("class_"):
                cid = int(fn.replace("class_", "").replace(".index", ""))
            else:
                try:
                    cid = int(fn.split(".")[0])
                except ValueError:
                    continue
            path = os.path.join(index_dir, fn)
            self._indices[cid] = faiss.read_index(path)
            meta_path = path.replace(".index", ".meta.npy")
            if os.path.isfile(meta_path):
                self._class_ids_in_index[cid] = np.load(meta_path)

    def query_class(
        self,
        class_id: int,
        emb: np.ndarray,
        topk: int = 3,
    ) -> Tuple[float, np.ndarray]:
        """Trả về (best_inner_product, distances hoặc indices)."""
        if class_id not in self._indices:
            return 0.0, np.array([])
        emb = emb.astype(np.float32).reshape(1, -1)
        faiss.normalize_L2(emb)
        index = self._indices[class_id]
        k = min(topk, index.ntotal)
        if k == 0:
            return 0.0, np.array([])
        sims, idxs = index.search(emb, k)
        return float(sims[0, 0]), idxs[0]

    def verify_crop(
        self,
        frame_bgr: np.ndarray,
        bbox_xyxy: List[int],
        mask: Optional[np.ndarray],
        cls_id: int,
        yolo_conf: float,
        cfg: FilterLayerConfig,
    ) -> Tuple[bool, str]:
        """
        Chạy 6+ lớp lọc. Trả về (pass, reason).
        """
        if not cfg.enabled:
            return True, "disabled"

        h, w = frame_bgr.shape[:2]
        x1, y1, x2, y2 = [int(x) for x in bbox_xyxy]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        bw, bh = x2 - x1, y2 - y1
        if bw < 2 or bh < 2:
            return False, "L1: bbox_too_small"

        area_bbox = bw * bh
        area_frame = h * w
        r = area_bbox / max(area_frame, 1)
        if r < cfg.min_area_ratio:
            return False, f"L1: area_ratio_low({r:.6f})"
        if r > cfg.max_area_ratio:
            return False, f"L1: area_ratio_high({r:.6f})"

        ar = bw / max(bh, 1)
        if ar < cfg.min_aspect or ar > cfg.max_aspect:
            return False, f"L2: aspect({ar:.3f})"

        if mask is not None:
            mh, mw = mask.shape[:2]
            m_use = mask
            if (mh, mw) != (h, w):
                import cv2 as _cv2

                m_use = _cv2.resize(
                    (mask > 0).astype(np.uint8), (w, h), interpolation=_cv2.INTER_NEAREST
                )
            sub = m_use[y1:y2, x1:x2]
            fill = float(sub.sum()) / max(area_bbox, 1)
            if fill < cfg.min_mask_fill_ratio:
                return False, f"L3: mask_fill({fill:.3f})"

        if yolo_conf < cfg.min_yolo_conf:
            return False, f"L4: yolo_conf({yolo_conf:.3f})"

        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            return False, "L1: empty_crop"

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        lap = cv2.Laplacian(gray, cv2.CV_64F).var()
        if lap < cfg.min_laplacian_var:
            return False, f"L7: blur(lap={lap:.1f})"

        edges = cv2.Canny(gray, 50, 150)
        edge_density = float(edges > 0).mean()
        if edge_density < cfg.min_edge_density:
            return False, f"L8: edge({edge_density:.4f})"

        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        emb = image_embed(self.model, self.processor, [crop_rgb], self.device)[0]

        if cls_id in self._indices and self._indices[cls_id].ntotal > 0:
            best_sim, _ = self.query_class(cls_id, emb, topk=cfg.faiss_topk)
            if best_sim < cfg.min_faiss_score:
                return False, f"L6: faiss_sim({best_sim:.3f})"
        else:
            # Không có index cho class -> chỉ geometry + blur (bỏ L5/L6 chặt)
            pass

        return True, "ok"


def default_filter_config_from_yaml(path: Optional[str]) -> FilterLayerConfig:
    """Đọc blocklist.yaml nếu có section siglip_filter."""
    cfg = FilterLayerConfig()
    if not path or not os.path.isfile(path):
        return cfg
    try:
        import yaml

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        s = data.get("siglip_filter") or {}
        if not s.get("enabled", False):
            cfg.enabled = False
            return cfg
        cfg.enabled = True
        key_map = {
            "model": "model_name",
            "weights": "weights_pt",
            "index_dir": "index_dir",
        }
        for k, v in s.items():
            if k == "enabled":
                continue
            attr = key_map.get(k, k)
            if hasattr(cfg, attr):
                setattr(cfg, attr, v)
    except Exception as e:
        print(f"[siglip2_verifier] yaml warn: {e}")
    return cfg
