"""
Script detect và match logo trong video với ảnh mẫu.
Sử dụng YOLOv11 để detect và SigLIP2 để embedding.

Usage:
    python match_logo_in_video.py --video test.mp4 --sample test_build_mu.jpg --threshold 0.9
"""

import argparse
import os
import tempfile
import cv2
import torch
import numpy as np
import time
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm
from compare_with_your_model import YourModelEmbedder

# Optional imports for ensemble methods
try:
    from paddleocr import PaddleOCR
    HAS_PADDLEOCR = True
except ImportError:
    HAS_PADDLEOCR = False
    # print("⚠️ PaddleOCR not available. Install with: pip install paddleocr")

try:
    import easyocr
    HAS_EASYOCR = True
except ImportError:
    HAS_EASYOCR = False
    print("⚠️ EasyOCR not available. Install with: pip install easyocr")

try:
    import imagehash
    HAS_IMAGEHASH = True
except ImportError:
    HAS_IMAGEHASH = False
    print("⚠️ imagehash not available. Install with: pip install imagehash")

try:
    # MobileSAM (segment anything, lightweight)
    from mobile_sam import sam_model_registry, SamPredictor
    HAS_MOBILE_SAM = True
except Exception:
    HAS_MOBILE_SAM = False
    print("⚠️ MobileSAM not available. Install it or disable segmentation.")

try:
    # SAM 2 (Segment Anything Model 2) image predictor
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    HAS_SAM2 = True
except Exception:
    HAS_SAM2 = False
    print("⚠️ SAM2 not available. Install it or disable SAM2 backend.")


class LogoMatcher:
    """Detect và match logo trong video với ảnh mẫu."""
    
    def __init__(
        self,
        yolo_model_path="model/yolov11s_model.pt",
        embedding_model_path="model/siglip2_model.pth",
        device="cuda" if torch.cuda.is_available() else "cpu",
        sam_checkpoint=None,
        sam_model_type="vit_t",
        enable_segmentation=False,
        sam_device=None,
        sam_every_n_frames: int = 3,
        sam_alpha: float = 0.6,
        sam_backend: str = "sam2",
        sam_config: str | None = None,
    ):
        """
        Initialize matcher.
        
        Args:
            yolo_model_path: Path to YOLO model
            embedding_model_path: Path to embedding model
            device: Device to use
            sam_checkpoint: Path to SAM checkpoint (.pt)
            sam_model_type: Model type (for MobileSAM) (e.g. 'vit_t')
            enable_segmentation: Enable segmentation in video
            sam_device: Device for segmentation model (default = device)
            sam_every_n_frames: Segment every N frames (>=1)
            sam_alpha: Alpha blending for overlay masks
            sam_backend: 'sam2' (default) or 'mobilesam'
            sam_config: Config .yaml for SAM2 (required when using SAM2 backend)
        """
        self.device = device
        print(f"Using device: {self.device}")
        
        # Initialize optional ensemble tools to None
        self.ocr = None
        self.sift = None
        self.ocr_type = None
        self.bf_matcher = None
        
        # Segmentation config (MobileSAM or SAM2)
        self.sam_predictor = None
        self.sam_enabled = False
        self.sam_alpha = sam_alpha
        self.sam_every_n_frames = max(1, int(sam_every_n_frames))
        self.sam_backend = None

        # Load YOLO
        print(f"\n📥 [1/2] Loading YOLO model: {yolo_model_path}")
        self.yolo = YOLO(yolo_model_path)
        print("   ✅ YOLO loaded!")
        
        # Load embedding model (SigLIP2 via YourModelEmbedder - 168/168 keys)
        print(f"\n📥 [2/2] Loading embedding model (SigLIP2): {embedding_model_path}")
        # Dùng padding/letterbox để giữ toàn bộ logo, giống test_logo_yolo_dino.py
        self.embedder = YourModelEmbedder(
            model_path=embedding_model_path,
            device=self.device,
            use_padding=True,
        )
        print("   ✅ SigLIP2 embedding model loaded!")

        # Optional: load segmentation backend (SAM2 or MobileSAM)
        if enable_segmentation and sam_checkpoint is not None:
            sam_device = sam_device or self.device
            sam_checkpoint_path = Path(sam_checkpoint)
            if not sam_checkpoint_path.is_file():
                print(f"⚠️ SAM checkpoint not found: {sam_checkpoint_path}. Segmentation disabled.")
            else:
                # Prefer SAM2 backend if requested and available
                if sam_backend == "sam2":
                    if not HAS_SAM2:
                        print("⚠️ SAM2 backend selected but SAM2 is not installed. Segmentation disabled.")
                    elif sam_config is None:
                        print("⚠️ SAM2 backend selected but no config .yaml provided (sam_config). Segmentation disabled.")
                    else:
                        try:
                            print(f"\n📥 [3/3] Loading SAM2: {sam_checkpoint_path}")
                            print(f"   Config: {sam_config}")
                            sam2_model = build_sam2(sam_config, str(sam_checkpoint_path))
                            sam2_model.to(device=sam_device)
                            sam2_model.eval()
                            self.sam_predictor = SAM2ImagePredictor(sam2_model)
                            self.sam_enabled = True
                            self.sam_device = sam_device
                            self.sam_backend = "sam2"
                            print(f"   ✅ SAM2 loaded on {sam_device}!")
                            print(f"   ➡️ Segmentation every {self.sam_every_n_frames} frame(s), alpha={self.sam_alpha:.2f}")
                        except Exception as e:
                            print(f"⚠️ Failed to load SAM2: {e}. Segmentation disabled.")
                # Fallback: MobileSAM backend (legacy)
                elif sam_backend == "mobilesam":
                    if not HAS_MOBILE_SAM:
                        print("⚠️ MobileSAM backend selected but mobile_sam is not installed. Segmentation disabled.")
                    else:
                        try:
                            print(f"\n📥 [3/3] Loading MobileSAM: {sam_checkpoint_path} (type={sam_model_type})")
                            mobile_sam = sam_model_registry.get(
                                sam_model_type, sam_model_registry["vit_t"]
                            )(checkpoint=str(sam_checkpoint_path))
                            mobile_sam.to(device=sam_device)
                            mobile_sam.eval()
                            self.sam_predictor = SamPredictor(mobile_sam)
                            self.sam_enabled = True
                            self.sam_device = sam_device
                            self.sam_backend = "mobilesam"
                            print(f"   ✅ MobileSAM loaded on {sam_device}!")
                            print(f"   ➡️ Segmentation every {self.sam_every_n_frames} frame(s), alpha={self.sam_alpha:.2f}")
                        except Exception as e:
                            print(f"⚠️ Failed to load MobileSAM: {e}. Segmentation disabled.")
    
    def _letterbox_resize_bgr(self, bgr_image, target_size=224):
        """
        Resize BGR image với letterbox padding (INTER_LANCZOS4 cho text clarity).
        
        Best practices từ ArcFace preprocessing:
        - INTER_LANCZOS4: Tốt nhất cho text (giảm 2-5% loss)
        - White padding: Phù hợp logo/text
        - Center alignment
        
        Returns: BGR numpy array (target_size x target_size)
        """
        old_h, old_w = bgr_image.shape[:2]
        
        # Scale ratio (fit vào target_size)
        ratio = float(target_size) / max(old_h, old_w)
        
        # New size (giữ tỷ lệ)
        new_h, new_w = int(old_h * ratio), int(old_w * ratio)
        
        # Chọn interpolation: LANCZOS4 (downscale)  / CUBIC (upscale)
        interp = cv2.INTER_LANCZOS4 if ratio < 1.0 else cv2.INTER_CUBIC
        
        # Resize
        resized = cv2.resize(bgr_image, (new_w, new_h), interpolation=interp)
        
        # Padding (center alignment)
        delta_h, delta_w = target_size - new_h, target_size - new_w
        top, left = delta_h // 2, delta_w // 2
        bottom, right = delta_h - top, delta_w - left
        
        # Add white padding
        padded = cv2.copyMakeBorder(
            resized, top, bottom, left, right,
            cv2.BORDER_CONSTANT, value=(255, 255, 255)
        )
        
        return padded
    
    def _initialize_ensemble_tools(self, use_ocr=False, use_sift=True):
        """
        Initialize ensemble verification tools (OCR, SIFT).
        Call this only if you want to use ensemble matching.
        """
        # Initialize OCR
        self.ocr_type = None
        if use_ocr:
            # Try PaddleOCR first
            if HAS_PADDLEOCR:
                print("   📝 Initializing OCR (PaddleOCR)...")
                try:
                    # Generic type error workaround for PaddleOCR
                    import logging
                    logging.getLogger("ppocr").setLevel(logging.ERROR)
                    self.ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
                    self.ocr_type = 'paddle'
                    print("   ✅ PaddleOCR ready!")
                except Exception as e:
                    print(f"   ⚠️ PaddleOCR Init failed: {e}")
                    self.ocr = None
            
            # Fallback to EasyOCR
            if self.ocr is None and HAS_EASYOCR:
                print("   📝 Initializing OCR (EasyOCR)...")
                try:
                    self.ocr = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
                    self.ocr_type = 'easy'
                    print("   ✅ EasyOCR ready!")
                except Exception as e:
                    print(f"   ⚠️ EasyOCR Init failed: {e}")
                    self.ocr = None
        else:
            self.ocr = None
        
        # Initialize SIFT
        if use_sift:
            print("   🔍 Initializing SIFT...")
            self.sift = cv2.SIFT_create()
            self.bf_matcher = cv2.BFMatcher()
            print("   ✅ SIFT ready!")
        else:
            self.sift = None
            self.bf_matcher = None
    
    @torch.no_grad()
    def get_embedding(self, bgr_image):
        """
        Get embedding từ ảnh BGR bằng SigLIP2 (YourModelEmbedder).
        Dùng file tạm để tái sử dụng pipeline của embedder (letterbox + normalize).
        """
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name
            cv2.imwrite(tmp_path, bgr_image)

        try:
            emb = self.embedder.get_embedding(tmp_path)
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

        return emb
    
    def get_batch_embeddings(self, bgr_images):
        """
        Get embeddings cho batch ảnh (dùng SigLIP2 embedder).
        Đơn giản là gọi get_embedding từng ảnh rồi stack lại.
        """
        if not bgr_images:
            return np.zeros((0, 0), dtype=np.float32)
        
        embs = [self.get_embedding(img) for img in bgr_images]
        return np.stack(embs, axis=0)

    def segment_with_mobilesam(self, frame_bgr, boxes):
        """
        Run MobileSAM trên 1 frame với danh sách bbox đã match.

        Args:
            frame_bgr: Full frame (BGR)
            boxes: List of (x1, y1, x2, y2) for matched logos

        Returns:
            overlay_bgr: Frame với mask màu overlay
            union_mask: Mask nhị phân gộp (H, W) uint8
            timing: dict với encode_ms, total_decode_ms, decode_ms_per_box
        """
        if not self.sam_enabled or self.sam_predictor is None or not boxes:
            h, w = frame_bgr.shape[:2]
            return frame_bgr, np.zeros((h, w), dtype=np.uint8), {
                "encode_ms": 0.0,
                "total_decode_ms": 0.0,
                "decode_ms_per_box": 0.0,
            }

        image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w = frame_bgr.shape[:2]
        overlay = frame_bgr.copy()
        final_mask = np.zeros((h, w), dtype=np.uint8)

        # Encode image một lần
        t_enc0 = time.perf_counter()
        self.sam_predictor.set_image(image_rgb)
        t_enc1 = time.perf_counter()
        encode_ms = (t_enc1 - t_enc0) * 1000.0

        total_decode = 0.0
        num_boxes = max(len(boxes), 1)

        for i, (x1, y1, x2, y2) in enumerate(boxes):
            # Clip bbox to frame size to tránh lỗi indexing
            x1_i = max(0, min(w - 1, int(x1)))
            y1_i = max(0, min(h - 1, int(y1)))
            x2_i = max(0, min(w, int(x2)))
            y2_i = max(0, min(h, int(y2)))
            if x2_i <= x1_i or y2_i <= y1_i:
                continue

            box = np.array([x1_i, y1_i, x2_i, y2_i], dtype=np.float32)

            t_dec0 = time.perf_counter()
            masks, scores, logits = self.sam_predictor.predict(
                box=box,
                multimask_output=False,
            )
            t_dec1 = time.perf_counter()
            total_decode += (t_dec1 - t_dec0) * 1000.0

            mask = masks[0].astype(np.uint8)  # (H, W) 0/1
            final_mask = np.maximum(final_mask, mask * 255)

            # ===== Blur ONLY the segmented area INSIDE bbox =====
            # ROI trên frame gốc và mask tương ứng
            roi = overlay[y1_i:y2_i, x1_i:x2_i]
            roi_mask = mask[y1_i:y2_i, x1_i:x2_i]

            if roi.size == 0 or roi_mask.size == 0:
                continue

            # Gaussian blur ROI, rồi chỉ apply ở vùng mask==1
            # Kernel 21x21 cho blur rõ, vẫn mượt
            blurred_roi = cv2.GaussianBlur(roi, (21, 21), 0)
            roi[roi_mask == 1] = blurred_roi[roi_mask == 1]

            color = (
                int(37 * (i + 1) % 255),
                int(17 * (i + 3) % 255),
                int(29 * (i + 5) % 255),
            )
            # Optional: overlay nhẹ màu lên vùng mask đã blur để dễ nhìn
            color_mask = np.zeros_like(frame_bgr, dtype=np.uint8)
            color_mask[mask == 1] = color
            overlay = cv2.addWeighted(overlay, 1.0, color_mask, self.sam_alpha, 0)

        decode_ms_per_box = total_decode / num_boxes

        timing = {
            "encode_ms": encode_ms,
            "total_decode_ms": total_decode,
            "decode_ms_per_box": decode_ms_per_box,
        }

        return overlay, final_mask, timing
    
    def compute_color_histogram(self, img, bins=32):
        """
        Compute normalized color histogram in HSV space.
        Fast pre-filter to reject logos with completely different colors.
        
        Args:
            img: BGR image
            bins: Number of bins per channel
        
        Returns:
            Flattened normalized histogram
        """
        # Convert to HSV (better than BGR for color comparison)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Compute 2D histogram (H and S channels, ignore V for lighting invariance)
        hist = cv2.calcHist([hsv], [0, 1], None, [bins, bins], [0, 180, 0, 256])
        
        # Normalize
        hist = cv2.normalize(hist, hist).flatten()
        
        return hist
    
    def compare_histograms(self, hist1, hist2):
        """
        Compare two histograms using correlation.
        
        Returns:
            Similarity score [0, 1], higher = more similar
        """
        return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    
    def extract_text_from_crop(self, crop):
        """
        Extract text from crop using OCR.
        
        Args:
            crop: BGR image
        
        Returns:
            Extracted text (lowercase, joined)
        """
        if self.ocr is None:
            return ""
        
        try:
            # PaddleOCR Logic
            if self.ocr_type == 'paddle':
                results = self.ocr.ocr(crop, cls=True)
                
                if not results or not results[0]:
                    return ""
                
                # Concatenate all detected text
                texts = []
                for line in results[0]:
                    text = line[1][0]
                    conf = line[1][1]
                    if conf > 0.5:
                        texts.append(text.lower())
                return " ".join(texts)
            
            # EasyOCR Logic
            elif self.ocr_type == 'easy':
                results = self.ocr.readtext(crop)
                # result format: ([[x,y]...], text, conf)
                texts = [text.lower() for (bbox, text, conf) in results if conf > 0.4]
                return " ".join(texts)
                
            return ""
        except Exception as e:
            # OCR can fail on some images
            return ""
    
    def extract_sift_features(self, img):
        """
        Extract SIFT keypoints and descriptors.
        
        Args:
            img: BGR image
        
        Returns:
            (keypoints, descriptors)
        """
        if self.sift is None:
            return None, None
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, des = self.sift.detectAndCompute(gray, None)
        return kp, des
    
    def match_sift_features(self, des1, des2):
        """
        Match two SIFT descriptor sets using Lowe's ratio test.
        
        Args:
            des1, des2: SIFT descriptors
        
        Returns:
            Number of good matches
        """
        if self.bf_matcher is None or des1 is None or des2 is None:
            return 0
        
        if len(des1) < 2 or len(des2) < 2:
            return 0
        
        try:
            matches = self.bf_matcher.knnMatch(des1, des2, k=2)
            
            # Lowe's ratio test
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.75 * n.distance:
                        good_matches.append(m)
            
            return len(good_matches)
        except Exception as e:
            return 0
    
    def get_effective_size(self, shape):
        """
        Calculate 'effective square area' based on longest dimension.
        This provides fair comparison between square and elongated logos.
        
        Args:
            shape: (height, width)
        
        Returns:
            effective_area: max_dim²
            max_dim: longest dimension
            min_dim: shortest dimension
        
        Examples:
            280×20  → max=280 → area=280²=78,400
            280×280 → max=280 → area=280²=78,400 (same!)
            100×50  → max=100 → area=100²=10,000
        """
        h, w = shape
        max_dim = max(h, w)
        min_dim = min(h, w)
        effective_area = max_dim * max_dim
        return effective_area, max_dim, min_dim
    
    def calculate_coverage(self, bbox, crop_w, crop_h):
        """
        Calculate coverage of bbox relative to crop area.
        Result is typically ~1.0 if crop is tight around bbox.
        Useful if crop size diverges from bbox or to enable thresholding based on fill.
        """
        x1, y1, x2, y2 = bbox
        w_box = max(0, x2 - x1)
        h_box = max(0, y2 - y1)
        bbox_area = w_box * h_box
        crop_area = crop_w * crop_h
        
        if crop_area == 0:
            return 0.0
            
        return bbox_area / crop_area
    
    def load_templates_with_shapes(self, template_paths, extract_features=True):
        """
        Load template images and compute embeddings with size info and optional features.
        
        Args:
            template_paths: Dict of {logo_id: image_path} or single path string
            extract_features: If True, extract color histogram, OCR text, and SIFT features
        
        Returns:
            Dict of {logo_id: template_data}
        """
        templates = {}
        
        # Handle string input
        if isinstance(template_paths, str):
            if os.path.isdir(template_paths):
                # Directory - load all images
                import glob
                files = []
                for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.webp']:
                    files.extend(glob.glob(os.path.join(template_paths, ext)))
                    files.extend(glob.glob(os.path.join(template_paths, ext.upper())))
                
                template_paths = {Path(f).stem: f for f in files}
                if not template_paths:
                    print(f"   ⚠️ No images found in directory: {template_paths}")
            else:
                # Single file
                # Use filename stem as ID for better logging
                p = Path(template_paths)
                template_paths = {p.stem: str(p)}
        
        print(f"\n📸 Loading {len(template_paths)} template(s)...")
        
        for logo_id, img_path in template_paths.items():
            img = cv2.imread(img_path)
            if img is None:
                print(f"   ⚠️ Cannot load: {img_path}")
                continue
            
            h, w = img.shape[:2]
            effective_area, max_dim, min_dim = self.get_effective_size((h, w))
            
            # Get embedding
            emb = self.get_embedding(img)
            
            template_data = {
                'shape': (h, w),
                'area': h * w,
                'effective_area': effective_area,
                'max_dim': max_dim,
                'min_dim': min_dim,
                'embedding': emb,
                'path': img_path
            }
            
            # Extract additional features if enabled
            if extract_features:
                # Color histogram
                template_data['histogram'] = self.compute_color_histogram(img)
                
                # OCR text
                if self.ocr is not None:
                    text = self.extract_text_from_crop(img)
                    template_data['text'] = text
                    if text:
                        print(f"   ✅ {logo_id}: {w}×{h} (effective: {max_dim}×{max_dim}) - text: '{text}'")
                    else:
                        print(f"   ✅ {logo_id}: {w}×{h} (effective: {max_dim}×{max_dim}) - no text")
                else:
                    template_data['text'] = ""
                    print(f"   ✅ {logo_id}: {w}×{h} (effective: {max_dim}×{max_dim})")
                
                # SIFT features
                if self.sift is not None:
                    kp, des = self.extract_sift_features(img)
                    template_data['sift_descriptors'] = des
                    template_data['sift_keypoints_count'] = len(kp) if kp else 0
                else:
                    template_data['sift_descriptors'] = None
                    template_data['sift_keypoints_count'] = 0
            else:
                print(f"   ✅ {logo_id}: {w}×{h} (effective: {max_dim}×{max_dim})")
            
            templates[logo_id] = template_data
        
        return templates
    
    def match_logo_with_gating(self, bbox, crop, templates, config):
        """
        Full ensemble multi-gate filtering pipeline for logo matching.
        
        Pipeline:
        1. Size + Coverage filtering (fast)
        2. Color Histogram check (1ms) - reject different colors
        3. Embedding + Cosine (50ms)  
        4. Verification (if not confident):
           - OCR text check (50ms)
           - SIFT features (20ms)
        
        Args:
            bbox: (x1, y1, x2, y2)
            crop: Cropped image (BGR)
            templates: Dict of template data
            config: Filtering config
        
        Returns:
            (logo_id, confidence, status, debug_info)
        """
        debug_info = {}
        
        # ===== GATE 1: EFFECTIVE SIZE FILTERING =====
        crop_h, crop_w = crop.shape[:2]
        crop_effective_area, crop_max_dim, crop_min_dim = self.get_effective_size((crop_h, crop_w))
        
        debug_info['crop_size'] = (crop_h, crop_w)
        debug_info['crop_effective_area'] = crop_effective_area
        debug_info['crop_min_dim'] = crop_min_dim
        
        # Filter by effective area
        if crop_effective_area < config['min_effective_area']:
            return None, 0.0, "crop_too_small", debug_info
        
        # Filter by minimum dimension (reject very thin logos)
        if crop_min_dim < config['min_dimension']:
            return None, 0.0, "dimension_too_small", debug_info
        
        # ===== GATE 1.5: COVERAGE GATING =====
        x1, y1, x2, y2 = bbox
        bbox_w = max(0, x2 - x1)
        bbox_h = max(0, y2 - y1)
        bbox_area = bbox_w * bbox_h
        crop_area = crop_h * crop_w
        
        coverage = bbox_area / crop_area if crop_area > 0 else 0
        debug_info['coverage'] = coverage
        
        if coverage < config['min_coverage']:
            return None, 0.0, "coverage_too_low", debug_info
        
        # ===== GATE 2: COLOR HISTOGRAM CHECK =====
        crop_hist = self.compute_color_histogram(crop)
        debug_info['color_similarities'] = {}
        
        color_candidates = {}
        for logo_id, template_data in templates.items():
            if 'histogram' in template_data:
                hist_sim = self.compare_histograms(crop_hist, template_data['histogram'])
                debug_info['color_similarities'][logo_id] = float(hist_sim)
                
                # Pass if color similarity > threshold
                if hist_sim > config.get('min_color_similarity', 0.3):
                    color_candidates[logo_id] = template_data
            else:
                # No histogram - accept (backward compatibility)
                color_candidates[logo_id] = template_data
        
        if not color_candidates:
            return None, 0.0, "failed_color_check", debug_info
        
        # ===== GATE 3: SIZE-BASED FILTERING PER TEMPLATE =====
        size_candidates = {}
        
        for logo_id, template_data in color_candidates.items():
            template_effective_area = template_data['effective_area']
            
            # Size ratio based on effective area
            size_ratio = crop_effective_area / template_effective_area if template_effective_area > 0 else 0
            
            if config['min_size_ratio'] <= size_ratio <= config['max_size_ratio']:
                size_candidates[logo_id] = template_data
                if 'size_ratios' not in debug_info:
                    debug_info['size_ratios'] = {}
                debug_info['size_ratios'][logo_id] = size_ratio
        
        if not size_candidates:
            return None, 0.0, "no_size_match", debug_info
        
        # ===== GATE 4: EMBEDDING + COSINE =====
        crop_emb = self.get_embedding(crop)
        
        cosine_scores = {}
        for logo_id, template_data in size_candidates.items():
            template_emb = template_data['embedding']
            cos_sim = float(np.dot(crop_emb, template_emb))
            cosine_scores[logo_id] = cos_sim
        
        # Sort by cosine
        sorted_logos = sorted(cosine_scores.items(), key=lambda x: x[1], reverse=True)
        
        if not sorted_logos:
            return None, 0.0, "no_candidates", debug_info
        
        best_logo, best_cos = sorted_logos[0]
        second_cos = sorted_logos[1][1] if len(sorted_logos) > 1 else 0.0
        
        debug_info['best_cosine'] = best_cos
        debug_info['second_cosine'] = second_cos
        debug_info['all_scores'] = cosine_scores
        
        # ===== DETERMINE ADAPTIVE THRESHOLD =====
        if coverage >= 0.15:  # Large, clear logo
            threshold = config['threshold_high_coverage']
        elif coverage >= 0.08:  # Medium logo
            threshold = config['threshold_medium_coverage']
        else:  # Small logo
            threshold = config['threshold_low_coverage']
        
        debug_info['threshold_used'] = threshold
        
        # ===== GATE 5: CONFIDENCE CHECK → DECIDE IF VERIFICATION NEEDED =====
        margin = best_cos - second_cos
        debug_info['margin'] = margin
        
        # High confidence thresholds (from config)
        confident_threshold = config.get('confident_cosine', 0.8)
        confident_margin = config.get('confident_margin', 0.1)
        
        is_confident = (best_cos >= confident_threshold and margin >= confident_margin)
        
        if is_confident:
            # Confident match
            # SAFETY CHECK: If strict text check is enabled, verify text even for confident matches
            # SAFETY CHECK: If strict text check is enabled, verify text even for confident matches
            if config.get('strict_text_check', False) and self.ocr is not None:
                 # Determine target text to check
                 target_text = config.get('expected_text')
                 if not target_text and 'text' in templates[best_logo]:
                     target_text = templates[best_logo]['text']
                 
                 if target_text: 
                     crop_text = self.extract_text_from_crop(crop)
                     if crop_text:
                         # Advanced checking logic: Word-based set intersection
                         t_words = set(target_text.lower().split())
                         c_words = set(crop_text.lower().split())
                         
                         # Count matching words
                         common_words = t_words.intersection(c_words)
                         match_ratio = len(common_words) / len(t_words) if len(t_words) > 0 else 0
                         
                         found = False
                         
                         # Condition 1: Word overlap
                         if len(t_words) == 1:
                             found = (match_ratio >= 1.0) # Single word must match
                             # Also allow substring match for single word if it's long enough
                             if not found and len(target_text) > 3 and target_text.lower() in crop_text.lower():
                                 found = True
                         else:
                             found = (match_ratio >= 0.5) # At least 50% match for multi-word
                             
                         # Condition 2: Fallback to exact substring (handling spacing issues)
                         if not found:
                             clean_target = target_text.lower().replace(" ", "")
                             clean_crop = crop_text.lower().replace(" ", "")
                             if clean_target in clean_crop:
                                 found = True
                             
                         # If target text not found, reject
                         if not found:
                             # Mismatch detected on confident match!
                             debug_info['text_mismatch_on_confident'] = True
                             debug_info['crop_text'] = crop_text
                             debug_info['expected_text'] = target_text
                             debug_info['match_ratio'] = match_ratio
                             
                             # Penalize score significantly
                             best_cos -= 0.3 # Stronger penalty
                             # Proceed to normal verification flow
                             pass
                         else:
                             # Text matches, confirm confident
                             return best_logo, best_cos, "matched", debug_info
                     else:
                        # Crop has no text found by OCR
                        # If strict mode is VERY strict, we might reject this too?
                        # For now, let it pass if image is confident
                        return best_logo, best_cos, "matched", debug_info
                 else:
                    return best_logo, best_cos, "matched", debug_info
            else:
                # Accept immediately
                return best_logo, best_cos, "matched", debug_info
        
        # ===== NOT CONFIDENT → VERIFICATION STAGE =====
        
        # Check if it's worth verifying (must be at least close to threshold)
        # Allow potential candidates: score >= threshold - 0.1
        potential_threshold = threshold - config.get('verification_leeway', 0.1)
        
        if best_cos < potential_threshold:
            # Too low even for verification
            return None, best_cos, "below_threshold", debug_info
            
        debug_info['verification_triggered'] = True
        
        # If score is between potential_threshold and threshold, we NEED verification to pass
        needs_verification_boost = (best_cos < threshold)
        
        # Check margin (if multiple candidates)
        if margin < config['min_margin'] and len(sorted_logos) > 1:
            # Low margin also needs verification
            needs_verification_boost = True
        
        # ===== VERIFICATION 1: OCR TEXT CHECK =====
        # ===== VERIFICATION 1: OCR TEXT CHECK =====
        # Determine target text again (to ensure we use expected_text if set)
        target_text = config.get('expected_text')
        if not target_text and 'text' in templates[best_logo]:
             target_text = templates[best_logo]['text']
             
        if self.ocr is not None and target_text:
            if target_text:  # Template has expected text
                crop_text = self.extract_text_from_crop(crop)
                debug_info['crop_text'] = crop_text
                debug_info['template_text'] = target_text
                
                if crop_text:
                    # Advanced checking logic: Word-based set intersection
                    t_words = set(target_text.lower().split())
                    c_words = set(crop_text.lower().split())
                    
                    common_words = t_words.intersection(c_words)
                    match_ratio = len(common_words) / len(t_words) if len(t_words) > 0 else 0
                    
                    found = False
                    if len(t_words) == 1:
                        found = (match_ratio >= 1.0)
                        if not found and len(target_text) > 3 and target_text.lower() in crop_text.lower():
                            found = True
                    else:
                        found = (match_ratio >= 0.5)
                        
                    if not found:
                         clean_target = target_text.lower().replace(" ", "")
                         clean_crop = crop_text.lower().replace(" ", "")
                         if clean_target in clean_crop:
                             found = True

                    if found:
                        # Text matches - boost confidence
                        debug_info['text_match'] = True
                        boost = config.get('text_match_boost', 0.1)  # Boost 0.1 for exact text match
                        best_cos += boost
                        debug_info['score_boost_ocr'] = boost
                    else:
                        # Text mismatch - likely wrong logo (even if not confident to start with)
                        debug_info['text_match'] = False
                        debug_info['match_ratio'] = match_ratio
                        # Text mismatch - likely wrong logo
                        debug_info['text_match'] = False
                        
                        # Only reject if we were relying on verification or strict check triggered
                        # If score was already high but text is DIFFERENT, that's a strong negative signal
                        return None, best_cos, "text_mismatch", debug_info
        
        # ===== VERIFICATION 2: SIFT FEATURE CHECK =====
        if self.sift is not None and 'sift_descriptors' in templates[best_logo]:
            template_des = templates[best_logo]['sift_descriptors']
            
            if template_des is not None:
                crop_kp, crop_des = self.extract_sift_features(crop)
                
                if crop_des is not None:
                    n_matches = self.match_sift_features(crop_des, template_des)
                    debug_info['sift_matches'] = n_matches
                    
                    min_sift_matches = config.get('min_sift_matches', 10)
                    
                    if n_matches >= min_sift_matches:
                        # Good feature match - boost confidence
                        boost = config.get('sift_match_boost', 0.05)
                        best_cos += boost
                        debug_info['score_boost_sift'] = boost
                    elif needs_verification_boost:
                         # If we needed verification but failed SIFT
                         # We don't reject immediately (might have passed OCR), but we don't boost
                         pass
        
        # ===== FINAL DECISION =====
        # Check against threshold again (with boosted score)
        if best_cos >= threshold:
            return best_logo, best_cos, "matched_after_verification", debug_info
        else:
            return None, best_cos, "below_threshold_after_verification", debug_info
    
    def match_logo_simple(self, bbox, crop, templates, config):
        """
        Phiên bản đơn giản: chỉ dùng embedding + cosine, KHÔNG qua multi-gate.
        
        Args:
            bbox: (x1, y1, x2, y2) - không dùng trong logic đơn giản, chỉ để log nếu cần
            crop: Cropped image (BGR)
            templates: Dict of template data (phải có 'embedding')
            config: Dict có key 'simple_threshold'
        
        Returns:
            (logo_id, confidence, status, debug_info)
        """
        debug_info = {}
        
        # Embedding cho crop
        crop_emb = self.get_embedding(crop)
        
        # Cosine với từng template
        cosine_scores = {}
        for logo_id, template_data in templates.items():
            template_emb = template_data["embedding"]
            cos_sim = float(np.dot(crop_emb, template_emb))
            cosine_scores[logo_id] = cos_sim
        
        if not cosine_scores:
            return None, 0.0, "no_candidates", debug_info
        
        # Lấy best
        best_logo, best_cos = max(cosine_scores.items(), key=lambda x: x[1])
        debug_info["best_cosine"] = best_cos
        debug_info["all_scores"] = cosine_scores
        
        threshold = config.get("simple_threshold", 0.7)
        debug_info["threshold_used"] = threshold
        
        if best_cos >= threshold:
            return best_logo, best_cos, "matched", debug_info
        else:
            return None, best_cos, "below_threshold", debug_info

    def process_video(
        self,
        video_path,
        sample_image_path,
        output_path,
        conf_threshold=0.5,
        config=None  # Multi-gate config
    ):
        """
        Process video với multi-gate filtering pipeline.
        
        Args:
            video_path: Path to input video
            sample_image_path: Path to sample image or dict of templates
            output_path: Path to output video
            conf_threshold: YOLO confidence threshold
            config: Dict of filtering thresholds
        """
        # Default config
        if config is None:
            config = {
                # Gate 1: Effective size
                'min_effective_area': 2500,    # 50×50 effective square
                'min_dimension': 15,           # Min shortest side
                
                # Gate 1.5: Coverage
                'min_coverage': 0.1,           # 10% of crop
                
                # Gate 2: Color histogram
                'min_color_similarity': 0.3,   # Min color similarity (0-1)
                
                # Gate 3: Size filtering
                'min_size_ratio': 0.25,        # Crop ≥25% template size
                'max_size_ratio': 4.0,         # Crop ≤400% template size
                
                # Gate 4: Adaptive thresholds
                'threshold_high_coverage': 0.88,   # ≥15% coverage
                'threshold_medium_coverage': 0.92,  # 8-15% coverage
                'threshold_low_coverage': 0.95,    # <8% coverage
                
                # Gate 5: Confidence check
                'min_margin': 0.05,            # Best - second ≥ 0.05
                'confident_cosine': 0.8,      # Cosine ≥ 0.95 = confident
                'confident_margin': 0.1,      # Margin ≥ 0.08 = confident
                
                # Verification stage
                'min_sift_matches': 10,        # Min SIFT feature matches
                'verification_leeway': 0.1,    # Allow candidates within 0.1 of threshold
                'text_match_boost': 0.1,       # Boost score by 0.1 if text matches
                'sift_match_boost': 0.05,      # Boost score by 0.05 if SIFT matches
                'strict_text_check': False     # Check text even for confident matches
            }
        
        # Load templates
        templates = self.load_templates_with_shapes(sample_image_path)
        
        if not templates:
            raise ValueError("No templates loaded!")
        
        # Open video
        print(f"\n🎬 Processing video: {video_path}")
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"   FPS: {fps:.2f}")
        print(f"   Resolution: {width}x{height}")
        print(f"   Total frames: {total_frames}")
        
        # Create output directory
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Stats - track filtering at each gate
        stats = {
            'total_frames': 0,
            'total_detections': 0,
            'filtered_crop_too_small': 0,
            'filtered_dimension_too_small': 0,
            'filtered_coverage_too_low': 0,
            'filtered_failed_color_check': 0,  # NEW
            'filtered_no_size_match': 0,
            'filtered_below_threshold': 0,
            'filtered_margin_too_small': 0,
            'filtered_text_mismatch': 0,  # NEW
            'filtered_text_mismatch_on_confident': 0,  # NEW - caught by strict check
            'filtered_insufficient_sift_matches': 0,  # NEW
            'filtered_below_threshold_after_verification': 0,  # NEW
            'matched': 0,
            'matched_after_verification': 0,  # NEW - matched via verification
            'matched_by_logo': {},  # Count per logo_id
            'verification_count': 0,  # NEW - how many went to verification
            # MobileSAM stats
            'sam_calls': 0,
            'sam_total_encode_ms': 0.0,
            'sam_total_decode_ms': 0.0,
        }
        
        # Initialize per-logo counts
        for logo_id in templates.keys():
            stats['matched_by_logo'][logo_id] = 0
        
        print(f"\n🔍 Detecting with multi-gate filtering...")
        print(f"   Config: min_effective_area={config['min_effective_area']}, "
              f"min_dim={config['min_dimension']}, min_coverage={config['min_coverage']}")
        
        total_det_time_ms = 0
        total_match_time_ms = 0
        total_proc_time_s = 0
        
        frame_idx = 0

        with tqdm(total=total_frames, desc="Processing") as pbar:
            while cap.isOpened():
                t_frame_start = time.time()
                
                ret, frame = cap.read()
                if not ret:
                    break
                
                stats['total_frames'] += 1
                frame_idx += 1

                # Detect với YOLO
                t_det_start = time.time()
                results = self.yolo(frame, conf=conf_threshold, verbose=False)
                t_det = (time.time() - t_det_start) * 1000 # ms
                total_det_time_ms += t_det
                
                t_match_total = 0
                matched_boxes_for_sam = []
                
                # Process detections
                for result in results:
                    boxes = result.boxes
                    
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        
                        # Get crop
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(width, x2), min(height, y2)
                        
                        if x2 > x1 and y2 > y1:
                            crop = frame[y1:y2, x1:x2]
                            stats['total_detections'] += 1
                            
                            # Chọn chế độ matching: multi-gate hoặc simple
                            t_gate_start = time.time()
                            if config.get("use_gates", True):
                                logo_id, confidence, status, debug_info = self.match_logo_with_gating(
                                    (x1, y1, x2, y2), crop, templates, config
                                )
                            else:
                                logo_id, confidence, status, debug_info = self.match_logo_simple(
                                    (x1, y1, x2, y2), crop, templates, config
                                )
                            t_match = (time.time() - t_gate_start) * 1000
                            t_match_total += t_match
                            
                            # Update stats
                            if status == "matched" or status == "matched_after_verification":
                                stats['matched'] += 1
                                stats['matched_by_logo'][logo_id] += 1
                                matched_boxes_for_sam.append((x1, y1, x2, y2))
                                
                                # Track verification separately
                                if status == "matched_after_verification":
                                    stats['matched_after_verification'] += 1
                                
                                # Track if verification was triggered
                                if debug_info.get('verification_triggered', False):
                                    stats['verification_count'] += 1
                                
                                # Draw green box
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                
                                # Label: logo_id + confidence
                                label = f"{logo_id} {confidence:.2f}"
                                color = (0, 255, 0)
                            else:
                                # Update filter stats
                                stats[f'filtered_{status}'] += 1
                                
                                if debug_info.get('text_mismatch_on_confident', False):
                                     stats['filtered_text_mismatch_on_confident'] += 1
                                
                                # Track verification even if rejected
                                if debug_info.get('verification_triggered', False):
                                    stats['verification_count'] += 1
                                
                                # Draw red/orange box for filtered
                                color = (0, 0, 255) # Red
                                if "text" in status: color = (0, 165, 255) # Orange for text mismatch
                                if "below_threshold" in status: color = (0, 100, 255) # Light Orange
                                
                                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
                                
                                # Short label
                                reason_map = {
                                    "crop_too_small": "Small",
                                    "dimension_too_small": "Small",
                                    "coverage_too_low": "Cov",
                                    "failed_color_check": "Color",
                                    "no_size_match": "Size",
                                    "below_threshold": f"Low {confidence:.2f}",
                                    "text_mismatch": "Text",
                                    "below_threshold_after_verification": f"LowV {confidence:.2f}",
                                    "insufficient_sift_matches": "SIFT"
                                }
                                label = reason_map.get(status, status)

                            # Draw label background and text
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            font_scale = 0.5
                            thickness = 1
                            (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
                            cv2.rectangle(frame, (x1, y1 - th - 4), (x1 + tw + 4, y1), color, -1)
                            cv2.putText(frame, label, (x1 + 2, y1 - 2), font, font_scale, (255, 255, 255), thickness)
                                
                total_match_time_ms += t_match_total

                # Optional: MobileSAM segmentation on matched boxes
                if (
                    self.sam_enabled
                    and self.sam_predictor is not None
                    and matched_boxes_for_sam
                    and (frame_idx % self.sam_every_n_frames == 0)
                ):
                    overlay_frame, union_mask, sam_timing = self.segment_with_mobilesam(
                        frame, matched_boxes_for_sam
                    )
                    frame = overlay_frame
                    stats['sam_calls'] += 1
                    stats['sam_total_encode_ms'] += sam_timing['encode_ms']
                    stats['sam_total_decode_ms'] += sam_timing['total_decode_ms']

                # Write frame (with or without segmentation overlay)
                out.write(frame)
                
                # Calculate FPS & Update Pbar
                t_frame_end = time.time()
                dt = t_frame_end - t_frame_start
                total_proc_time_s += dt
                fps_curr = 1.0 / dt if dt > 0 else 0
                
                pbar.set_description(f"Proc (FPS: {fps_curr:.1f} | Det: {t_det:.0f}ms | Match: {t_match_total:.0f}ms)")
                pbar.update(1)
        
        # Calculate Averages
        avg_det_time = total_det_time_ms / stats['total_frames'] if stats['total_frames'] > 0 else 0
        avg_match_time = total_match_time_ms / stats['total_frames'] if stats['total_frames'] > 0 else 0
        avg_fps = stats['total_frames'] / total_proc_time_s if total_proc_time_s > 0 else 0
        
        stats['avg_det_time_ms'] = avg_det_time
        stats['avg_match_time_ms'] = avg_match_time
        stats['avg_fps'] = avg_fps
        
        # Release resources
        cap.release()
        out.release()
        
        # Print comprehensive stats
        print(f"\n{'='*70}")
        print("📊 ENSEMBLE MULTI-GATE FILTERING RESULTS")
        print('='*70)
        print(f"Total frames: {stats['total_frames']}")
        print(f"Total detections: {stats['total_detections']}")
        
        if stats['total_detections'] > 0:
            match_pct = stats['matched'] / stats['total_detections'] * 100
            print(f"\n✅ Matched: {stats['matched']} ({match_pct:.1f}%)")
            
            # Verification stats
            if stats['verification_count'] > 0:
                ver_pct = stats['verification_count'] / stats['total_detections'] * 100
                print(f"   → Verification triggered: {stats['verification_count']} ({ver_pct:.1f}%)")
                
                if stats['matched_after_verification'] > 0:
                    ver_match_pct = stats['matched_after_verification'] / stats['verification_count'] * 100
                    print(f"   → Matched after verification: {stats['matched_after_verification']} ({ver_match_pct:.1f}% of verified)")
            
            # Per-logo breakdown
            if len(templates) > 1:
                print(f"\n   Breakdown by logo:")
                for logo_id, count in stats['matched_by_logo'].items():
                    if count > 0:
                        print(f"   - {logo_id}: {count}")
            
            print(f"\n❌ Filtered: {stats['total_detections'] - stats['matched']} "
                  f"({100 - match_pct:.1f}%)")
            
            # Breakdown by filter reason (in pipeline order)
            filter_keys = [
                ('filtered_crop_too_small', 'Gate 1: Crop too small'),
                ('filtered_dimension_too_small', 'Gate 1: Dimension too small'),
                ('filtered_coverage_too_low', 'Gate 1.5: Coverage too low'),
                ('filtered_failed_color_check', 'Gate 2: Color histogram failed'),
                ('filtered_no_size_match', 'Gate 3: No size match'),
                ('filtered_below_threshold', 'Gate 4: Below embedding threshold'),
                ('filtered_text_mismatch', 'Verification: Text mismatch'),
                ('filtered_text_mismatch_on_confident', 'Verification: Strict text mismatch found'),
                ('filtered_insufficient_sift_matches', 'Verification: Insufficient SIFT matches'),
                ('filtered_below_threshold_after_verification', 'Verification: Still below threshold'),
            ]
            
            for key, label in filter_keys:
                val = stats.get(key, 0)
                if val > 0:
                    pct = val / stats['total_detections'] * 100
                    print(f"   - {label}: {val} ({pct:.1f}%)")
        
        print(f"\n⏱️ Performance Metrics (Average):")
        print(f"   - Processing FPS: {stats.get('avg_fps', 0):.1f}")
        print(f"   - Detection Time: {stats.get('avg_det_time_ms', 0):.2f} ms")
        print(f"   - Matching Time : {stats.get('avg_match_time_ms', 0):.2f} ms")
        if stats.get('sam_calls', 0) > 0:
            avg_sam_enc = stats['sam_total_encode_ms'] / stats['sam_calls']
            avg_sam_dec = stats['sam_total_decode_ms'] / stats['sam_calls']
            print(f"   - MobileSAM Encode: {avg_sam_enc:.2f} ms (per call)")
            print(f"   - MobileSAM Decode: {avg_sam_dec:.2f} ms (per call, all boxes)")
        
        print(f"\n💾 Output saved to: {output_path}")
        print('='*70)
        
        return stats


def main():
    parser = argparse.ArgumentParser(
        description="Detect và match logo trong video với multi-gate filtering"
    )
    
    parser.add_argument("--video", type=str, required=True,
                        help="Path to input video")
    parser.add_argument("--sample", type=str, required=True,
                        help="Path to sample image or directory of templates")
    parser.add_argument("--output", type=str, default="output/matched_video.mp4",
                        help="Path to output video (default: output/matched_video.mp4)")
    parser.add_argument("--yolo-model", type=str, default="model/yolov11s_model.pt",
                        help="Path to YOLO model (default: model/yolov11s_model.pt)")
    parser.add_argument("--embedding-model", type=str, default="model/siglip2_model.pth",
                        help="Path to embedding model (default: model/siglip2_model.pth)")
    parser.add_argument("--conf", type=float, default=0.5,
                        help="YOLO confidence threshold (default: 0.5)")
    parser.add_argument("--no-gates", action="store_true",
                        help="Tắt toàn bộ multi-gate filters, chỉ dùng embedding + cosine.")
    parser.add_argument("--simple-threshold", type=float, default=0.7,
                        help="Ngưỡng cosine cho chế độ đơn giản (no-gates).")

    # Segmentation options (SAM2 or MobileSAM)
    parser.add_argument("--enable-sam", action="store_true",
                        help="Bật segmentation overlay cho logo đã match (SAM2 hoặc MobileSAM).")
    parser.add_argument("--sam-checkpoint", type=str, default="MobileSAM/weights/mobile_sam.pt",
                        help="Đường dẫn checkpoint (.pt) cho SAM backend.")
    parser.add_argument("--sam-backend", type=str, default="sam2",
                        choices=["sam2", "mobilesam"],
                        help="Backend segmentation: 'sam2' (mặc định) hoặc 'mobilesam'.")
    parser.add_argument("--sam-config", type=str, default=None,
                        help="Đường dẫn file config .yaml cho SAM2 (bắt buộc nếu dùng backend sam2).")
    parser.add_argument("--sam-model-type", type=str, default="vit_t",
                        help="Kiểu MobileSAM model (default: vit_t, chỉ dùng khi backend=mobilesam).")
    parser.add_argument("--sam-device", type=str, default=None,
                        help="Device riêng cho SAM backend (mặc định = cùng device).")
    parser.add_argument("--sam-every-n-frames", type=int, default=3,
                        help="Chạy SAM segmentation mỗi N frame (default: 3).")
    parser.add_argument("--sam-alpha", type=float, default=0.6,
                        help="Alpha overlay mask segmentation (default: 0.6).")
    
    # Multi-gate filtering config
    parser.add_argument("--min-effective-area", type=float, default=2500,
                        help="Min effective area (max_dim²) (default: 2500 = 50×50)")
    parser.add_argument("--min-dimension", type=float, default=15,
                        help="Min shortest dimension in pixels (default: 15)")
    parser.add_argument("--min-coverage", type=float, default=0.1,
                        help="Min coverage (bbox/crop) (default: 0.1 = 10%%)")
    parser.add_argument("--min-size-ratio", type=float, default=0.5,
                        help="Min size ratio (crop/template) (default: 0.25)")
    parser.add_argument("--max-size-ratio", type=float, default=2.5,
                        help="Max size ratio (crop/template) (default: 3.0)")
    parser.add_argument("--threshold-high", type=float, default=0.88,
                        help="Similarity threshold for high coverage (≥15%%) (default: 0.88)")
    parser.add_argument("--threshold-medium", type=float, default=0.92,
                        help="Similarity threshold for medium coverage (8-15%%) (default: 0.92)")
    parser.add_argument("--threshold-low", type=float, default=0.8 ,
                        help="Similarity threshold for low coverage (<8%%) (default: 0.8)")
    parser.add_argument("--min-margin", type=float, default=0.05,
                        help="Min margin between best and second best (default: 0.05)")
    
    # Ensemble verification config
    parser.add_argument("--use-ocr", action="store_true",
                        help="Enable OCR text verification (requires paddleocr)")
    parser.add_argument("--use-sift", action="store_true",
                        help="Enable SIFT feature verification")
    parser.add_argument("--min-color-similarity", type=float, default=0.1,
                        help="Min color histogram similarity (default: 0.3)")
    parser.add_argument("--confident-cosine", type=float, default=0.8,
                        help="Cosine threshold for confident match (default: 0.95)")
    parser.add_argument("--confident-margin", type=float, default=0.08,
                        help="Margin threshold for confident match (default: 0.08)")
    parser.add_argument("--min-sift-matches", type=int, default=10,
                        help="Min SIFT feature matches for verification (default: 10)")
    parser.add_argument("--verification-leeway", type=float, default=0.1,
                        help="Allow candidates within this range of threshold to verify (default: 0.1)")
    parser.add_argument("--text-match-boost", type=float, default=0.1,
                        help="Boost score by this amount if text matches (default: 0.1)")
    parser.add_argument("--sift-match-boost", type=float, default=0.05,
                        help="Boost score by this amount if SIFT matches (default: 0.05)")
    
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use: 'cuda' or 'cpu' (default: auto-detect)")
    parser.add_argument("--strict-ocr", action="store_true",
                        help="Force OCR check even for confident matches (rejects if text mismatch)")
    parser.add_argument("--expected-text", type=str, default=None,
                        help="Manually specify expected text (overrides OCR from template). E.g: 'Manchester United'")
    
    args = parser.parse_args()
    
    # Check files exist
    if not os.path.exists(args.video):
        print(f"❌ Video not found: {args.video}")
        return
    
    if not os.path.exists(args.sample):
        print(f"❌ Sample image not found: {args.sample}")
        return
    
    if not os.path.exists(args.yolo_model):
        print(f"❌ YOLO model not found: {args.yolo_model}")
        return
    
    if not os.path.exists(args.embedding_model):
        print(f"❌ Embedding model not found: {args.embedding_model}")
        return
    
    # Initialize matcher
    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    
    matcher = LogoMatcher(
        yolo_model_path=args.yolo_model,
        embedding_model_path=args.embedding_model,
        device=device,
        sam_checkpoint=args.sam_checkpoint if args.enable_sam else None,
        sam_model_type=args.sam_model_type,
        enable_segmentation=args.enable_sam,
        sam_device=args.sam_device,
        sam_every_n_frames=args.sam_every_n_frames,
        sam_alpha=args.sam_alpha,
        sam_backend=args.sam_backend,
        sam_config=args.sam_config,
    )
    
    # Initialize ensemble tools if requested
    if args.use_ocr or args.use_sift:
        print(f"\n🔧 Initializing ensemble verification tools...")
        matcher._initialize_ensemble_tools(use_ocr=args.use_ocr, use_sift=args.use_sift)
    
    # Build config from arguments
    config = {
        'min_effective_area': args.min_effective_area,
        'min_dimension': args.min_dimension,
        'min_coverage': args.min_coverage,
        'min_color_similarity': args.min_color_similarity,
        'min_size_ratio': args.min_size_ratio,
        'max_size_ratio': args.max_size_ratio,
        'threshold_high_coverage': args.threshold_high,
        'threshold_medium_coverage': args.threshold_medium,
        'threshold_low_coverage': args.threshold_low,
        'min_margin': args.min_margin,
        'confident_cosine': args.confident_cosine,
        'confident_margin': args.confident_margin,
        'min_sift_matches': args.min_sift_matches,
        'verification_leeway': args.verification_leeway,
        'text_match_boost': args.text_match_boost,
        'sift_match_boost': args.sift_match_boost,
        'strict_text_check': args.strict_ocr,
        'expected_text': args.expected_text,
        'use_gates': not args.no_gates,
        'simple_threshold': args.simple_threshold,
    }
    
    # Process video
    matcher.process_video(
        video_path=args.video,
        sample_image_path=args.sample,
        output_path=args.output,
        conf_threshold=args.conf,
        config=config
    )
    
    print("\n✅ Done!")


if __name__ == "__main__":
    main()

