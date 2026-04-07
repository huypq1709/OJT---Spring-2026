# 🎯 SMVL — Hệ thống Nhận diện và Che mờ Logo Bản quyền trên Video Truyền hình

> **Giải pháp AI tự động phát hiện, phân vùng và che mờ (pixelate) logo vi phạm bản quyền trong video phát sóng trực tiếp, sử dụng pipeline kết hợp YOLOv11 + SigLIP2 FAISS + SAM và YOLOv11-seg + SigLIP2.**

---

## 📋 Mục lục

- [Tổng quan](#-tổng-quan)
- [Kiến trúc Pipeline](#-kiến-trúc-pipeline)
- [Cấu trúc Dự án](#-cấu-trúc-dự-án)
- [Cài đặt](#-cài-đặt)
- [Quy trình chung (Cả 2 Phương pháp)](#-quy-trình-chung-cả-2-phương-pháp)
- [Phương pháp 1: YOLOv11 + SigLIP2 + SAM](#-phương-pháp-1-yolov11--siglip2--sam)
- [Phương pháp 2: YOLOv11-seg + SigLIP2 ⭐](#-phương-pháp-2-yolov11-seg--siglip2-)
- [Kết quả Đánh giá](#-kết-quả-đánh-giá)
- [Công cụ Hỗ trợ](#-công-cụ-hỗ-trợ)
- [Tối ưu Hiệu năng](#-tối-ưu-hiệu-năng)
- [Cấu hình Hệ thống](#-cấu-hình-hệ-thống)

---

## 🔍 Tổng quan

Trong bài toán xử lý video phát sóng (truyền hình thể thao, sự kiện trực tiếp), việc phát hiện và làm mờ chính xác logo các nhãn hàng vi phạm bản quyền là yêu cầu giám sát trọng yếu. Hệ thống SMVL giải quyết bài toán này bằng kiến trúc multi-stage:

1. **YOLOv11** — Phát hiện vị trí logo (Bounding Box / Segmentation Mask)
2. **SigLIP2 + FAISS** — Xác minh danh tính logo bằng vector embedding, loại bỏ false positive
3. **SAM (Segment Anything)** — Tạo polygon mask pixel-perfect (phương pháp thay thế)

### So sánh 2 Phương pháp

| | **PP1:** YOLO + SigLIP2 + SAM | **PP2:** YOLOv11-seg + SigLIP2 ⭐ |
|---|---|---|
| **Pipeline** | Detect Box → Lọc → SAM Segment | Detect+Segment → Lọc |
| **Models cần load** | 3 (YOLO + SigLIP2 + SAM) | 2 (YOLO-seg + SigLIP2) |
| **Chất lượng Mask** | Pixel-perfect | Phụ thuộc dữ liệu polygon |
| **Tốc độ** | Chậm (~3 FPS) | Nhanh (~6.6 FPS) |
| **VRAM** | Cao (≥8GB) | Thấp (≥4GB) |
| **Phù hợp** | Offline, chất lượng cao | **Realtime, phát sóng trực tiếp** |

---

## 🏗 Kiến trúc Pipeline

### Sơ đồ tổng thể — PP2 (Khuyến nghị)
```
Video Input
    │
    ▼
┌──────────────┐     ┌───────────────┐     ┌──────────────┐
│  YOLOv11-seg │────▶│  Crop Bbox    │────▶│  SigLIP2     │
│  Detect +    │     │  từ Frame     │     │  Embedding   │
│  Mask        │     └───────────────┘     │  (512-d)     │
└──────────────┘                           └──────┬───────┘
                                                  │
                                                  ▼
                                          ┌───────────────┐
                                          │  FAISS Index  │
                                          │  Cosine ≥0.91 │
                                          └──────┬────────┘
                                                 │
                                    ┌────────────┴────────────┐
                                    ▼                         ▼
                            ┌──────────────┐        ┌──────────────┐
                            │  ✅ Match    │        │  ❌ Reject   │
                            │  Pixelate    │        │  (False +)   │
                            │  Mask Area   │        │  Loại bỏ     │
                            └──────────────┘        └──────────────┘
                                    │
                                    ▼
                              Video Output
```

### Sơ đồ tổng thể — PP1 (Chất lượng cao)
```
Video Input
    │
    ▼
┌──────────────┐     ┌───────────────┐     ┌──────────────┐
│  YOLOv11     │────▶│  Crop Bbox    │────▶│  SigLIP2     │
│  Detect Box  │     │  từ Frame     │     │  Embedding   │
└──────────────┘     └───────────────┘     │  (512-d)     │
                                           └──────┬───────┘
                                                  │
                                          ┌───────────────┐
                                          │  FAISS Index  │
                                          │  Cosine ≥0.91 │
                                          └──────┬────────┘
                                                 │
                                    ┌────────────┴────────────┐
                                    ▼                         ▼
                         ┌─────────────────┐        ┌──────────────┐
                         │  ✅ Match       │        │  ❌ Reject   │
                         │  → SAM Segment  │        │  Loại bỏ     │
                         │  → Blur Mask    │        └──────────────┘
                         └─────────────────┘
                                    │
                                    ▼
                              Video Output
```

---

## 📂 Cấu trúc Dự án

```
SMVL/
│
├── README.md                              # 📖 Tài liệu hướng dẫn
├── requirements.txt                       # 📦 Thư viện Python
├── .gitignore                             # Git ignore rules
├── blocklist.yaml                         # ⚙️ Cấu hình brand cần che mờ
├── brands.txt                             # 📋 Danh sách tất cả nhãn hiệu
│
├── 📁 weights/                            # 🧠 Model weights
│   ├── siglip2_model.pth                 #    SigLIP2 fine-tuned (~357MB)
│   └── yolov11s_model.pt                 #    YOLOv11s detection (~39MB)
│
├── 📁 scripts/                            # 📂 Tất cả source code
│   │
│   ├── 📁 training/                       # 🏋️ Huấn luyện
│   │   ├── train_yolo_detect.py          #    Bước 2: Train YOLOv11 Detection
│   │   ├── crop_labeled_regions.py       #    Bước 3a: Crop ảnh theo class
│   │   ├── train_siglip2.py              #    Bước 3b: Train SigLIP2 Classifier
│   │   ├── run_sam.py                    #    PP2-A: Tạo polygon bằng SAM
│   │   └── train_yolo_seg.py             #    PP2-B: Train YOLOv11-seg
│   │
│   ├── 📁 database/                       # 🗄️ FAISS Vector Index
│   │   ├── build_logo_faiss_index.py     #    Bước 4: Build FAISS index
│   │   └── append_failed_crops_to_index.py  # Cập nhật index
│   │
│   ├── 📁 inference/                      # 🎬 Chạy Pipeline
│   │   ├── pixelate_yolo_siglip2.py      #    ⭐ PP2: YOLOv11-seg + SigLIP2
│   │   ├── match_logo_in_video.py        #    PP1: YOLO + SigLIP2 + SAM
│   │   ├── detect_video.py               #    Detect-only (debug)
│   │   └── siglip2_verifier.py           #    Module xác minh SigLIP2
│   │
│   ├── 📁 evaluation/                     # 📊 Đánh giá & Báo cáo
│   │   ├── generate_report.py            #    Sinh báo cáo training
│   │   └── inspect_misclassified.py      #    Phân tích lỗi phân loại
│   │
│   └── 📁 utils/                          # 🔧 Tiện ích
│       └── export_yolo_tensorrt.py       #    Export YOLO → TensorRT
│
└── (Các thư mục dữ liệu — không có trong repo, tải riêng)
    ├── SMVL_dataset/                      # Dataset YOLOv11 (từ Roboflow)
    └── cropped_classes/                   # Dataset SigLIP2 (từ Kaggle)
```

---

## ⚙ Cài đặt

### Yêu cầu hệ thống
- Python ≥ 3.9
- NVIDIA GPU có CUDA (khuyến nghị ≥ 4GB VRAM)
- CUDA Toolkit ≥ 11.7

### Cài đặt Dependencies

```bash
# Clone repository
git clone https://github.com/<your-username>/SMVL.git
cd SMVL

# Tạo virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Cài đặt thư viện
pip install -r requirements.txt

# Cài thêm các thư viện cần thiết
pip install open_clip_torch faiss-cpu scikit-learn matplotlib pyyaml tqdm
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### `requirements.txt`

```
opencv-python>=4.8.0
torch>=2.0.0
numpy>=1.24.0
ultralytics>=8.0.0
albumentations>=1.3.0
faiss-cpu>=1.7.4
Pillow>=10.0.0
tqdm>=4.65.0
transformers>=4.30.0
sentencepiece>=0.1.99
```

---

## 📦 Dữ liệu (Datasets)

Dự án sử dụng 2 nguồn dữ liệu chính:

| Dataset | Mục đích | Link |
|---|---|---|
| **Logo Betting (Roboflow)** | Train YOLOv11 Detection — 12.926 ảnh, 223 classes, PASCAL VOC XML | [🔗 Roboflow](https://app.roboflow.com/dpl-8udgz/logo_betting/browse?queryText=&pageSize=50&startingIndex=0&browseQuery=true) |
| **SigLIP2 Crops (Kaggle)** | Train SigLIP2 Classifier — Ảnh crop theo class, 227 thư mục | [🔗 Kaggle](https://www.kaggle.com/datasets/se184775/dataset) |

### Cách tải dữ liệu

```bash
# 1. Dataset YOLOv11 Detection — tải từ Roboflow
#    Truy cập link Roboflow ở trên → Export → chọn format PASCAL VOC XML
#    Giải nén vào thư mục SMVL_dataset/

# 2. Dataset SigLIP2 Training — tải từ Kaggle
pip install kaggle
kaggle datasets download -d se184775/dataset
#    Giải nén vào thư mục cropped_classes/
```

---

## 🔄 Quy trình Chung (Cả 2 Phương pháp)

Cả PP1 và PP2 đều chia sẻ **4 bước chuẩn bị** sau đây. Sau khi hoàn thành, bạn chọn theo phương pháp tương ứng.

```
┌─────────────────────────────────────────────────────────────────┐
│                    QUY TRÌNH CHUNG                              │
│                                                                 │
│  Bước 1: Chuẩn bị Dataset (tải từ Roboflow + Kaggle)          │
│      │                                                          │
│      ▼                                                          │
│  Bước 2: Train YOLOv11 Detection (train_yolo_detect.py)        │
│      │                                                          │
│      ▼                                                          │
│  Bước 3: Crop + Train SigLIP2 Embedding                        │
│      │   crop_labeled_regions.py → train_siglip2.py             │
│      │                                                          │
│      ▼                                                          │
│  Bước 4: Build FAISS Vector Index (build_logo_faiss_index.py)  │
│      │                                                          │
│      ├──────────────────┬───────────────────────┐               │
│      ▼                  ▼                       │               │
│  ┌────────┐      ┌─────────────┐                │               │
│  │  PP1   │      │  PP2 ⭐     │                │               │
│  │ +SAM   │      │ +YOLO-seg   │                │               │
│  └────────┘      └─────────────┘                │               │
└─────────────────────────────────────────────────────────────────┘
```

---

### Bước 1: Chuẩn bị Dữ liệu

> 📥 **Tải dataset từ Roboflow:** [Logo Betting Dataset](https://app.roboflow.com/dpl-8udgz/logo_betting/browse?queryText=&pageSize=50&startingIndex=0&browseQuery=true)

Dataset gồm **12.926 hình ảnh** thu thập từ dữ liệu sân cỏ, phân bố trên **223 lớp nhãn** (classes), annotate bounding box theo chuẩn PASCAL VOC (XML).

```
SMVL_dataset/
├── SMVL_Annotations/Annotations/   # XML annotation files
└── SMVL_JPEGImages/JPEGImages/     # Original images
```

---

### Bước 2: Huấn luyện YOLOv11 Detection

> **Script:** [`train_yolo_detect.py`](scripts/training/train_yolo_detect.py)

```bash
python scripts/training/train_yolo_detect.py \
    --data SMVL_dataset/data.yaml \
    --model yolo11s.pt \
    --epochs 100 --imgsz 1280 --batch 8 --device 0
```

| Cấu hình | Giá trị |
|---|---|
| Optimizer | AdamW (lr=0.001) |
| Augmentation | mosaic=1.0, mixup=0.1, fliplr=0.5 |
| Early stopping | patience=20 |

**Kết quả (100 Epochs):** mAP50 = **96.33%**, Precision = **93.53%**, Recall = **92.83%**

---

### Bước 3: Crop Dataset + Huấn luyện SigLIP2

#### 3a. Crop ảnh theo class

> **Script:** [`crop_labeled_regions.py`](scripts/training/crop_labeled_regions.py)

```bash
python scripts/training/crop_labeled_regions.py
# Output: cropped_classes/  (227 thư mục, mỗi thư mục 1 nhãn hiệu)
```

> 💡 **Hoặc tải trực tiếp dataset đã crop từ Kaggle:** [SigLIP2 Training Dataset](https://www.kaggle.com/datasets/se184775/dataset) — bỏ qua bước 3a nếu sử dụng dataset này.

#### 3b. Huấn luyện SigLIP2 Classifier

> **Script:** [`train_siglip2.py`](scripts/training/train_siglip2.py) | **Báo cáo:** [`generate_report.py`](scripts/evaluation/generate_report.py)

```bash
python scripts/training/train_siglip2.py \
    --data-dir cropped_classes \
    --output-dir train_output \
    --epochs 25 --batch-size 192 --unfreeze-epoch 8

# Sinh báo cáo
python scripts/evaluation/generate_report.py --output-dir train_output --data-dir cropped_classes --full
```

**Kiến trúc:**
```
Backbone: open_clip ViT-B-16-SigLIP2-256 (frozen → unfreeze epoch 8)
Head:     LN(768) → Linear(768→512) → ReLU → Dropout(0.3) → Linear(512→num_classes)
```

**Kết quả (25 Epochs):** Val Accuracy = **99.95%** | 222/227 classes ≥ 95% F1

---

### Bước 4: Xây dựng FAISS Vector Index

> **Script:** [`build_logo_faiss_index.py`](scripts/database/build_logo_faiss_index.py)

```bash
python scripts/database/build_logo_faiss_index.py \
    --ann-dir SMVL_dataset/SMVL_Annotations/Annotations \
    --img-dir SMVL_dataset/SMVL_JPEGImages/JPEGImages \
    --model weights/siglip2_model.pth
```

**Output:** `logo_index.faiss` (~750MB) + `logo_meta.npy`

#### Cập nhật Index (tùy chọn)

> **Script:** [`append_failed_crops_to_index.py`](scripts/database/append_failed_crops_to_index.py)

```bash
python scripts/database/append_failed_crops_to_index.py --dir failed_crops
```

---

> ✅ **Hoàn thành Quy trình Chung.** Tiếp tục chọn phương pháp bên dưới.

---

## 🟦 Phương pháp 1: YOLOv11 + SigLIP2 + SAM

> **Ưu tiên chất lượng mask — Phù hợp xử lý offline, yêu cầu thẩm mỹ cao.**

### Quy trình riêng PP1

```
Sau Bước 4 (Quy trình Chung)
    │
    ▼
┌──────────────────────────────────────────────────────────┐
│  PP1 — Không cần thêm bước training                     │
│                                                          │
│  Chỉ cần tải thêm SAM checkpoint:                       │
│    • sam2_b.pt (SAM 2 Base — 161MB)                     │
│    • hoặc sam2_s.pt / sam2_t.pt (nhỏ hơn)              │
│                                                          │
│  YOLO detect → SigLIP2 lọc → SAM segment → blur mask   │
└──────────────────────────────────────────────────────────┘
```

### Luồng xử lý PP1 (per frame)

```
Frame ──▶ YOLOv11 Detect ──▶ Danh sách Bounding Box
                                      │
                                      ▼
                              Crop mỗi Box ──▶ SigLIP2 Embedding (512-d)
                                                        │
                                                        ▼
                                                FAISS Cosine Search
                                                        │
                                          ┌─────────────┴──────────────┐
                                          ▼                            ▼
                                  Cosine ≥ 0.91                Cosine < 0.91
                                  (Logo hợp lệ)               (Loại bỏ)
                                          │
                                          ▼
                                  SAM Segment (Box Prompt)
                                  → Pixel-perfect Mask
                                          │
                                          ▼
                                  Gaussian Blur vùng Mask
                                          │
                                          ▼
                                     Frame Output
```

### Chạy PP1

> **Script:** [`match_logo_in_video.py`](scripts/inference/match_logo_in_video.py)

```bash
# Chạy với SAM2 backend (khuyến nghị)
python scripts/inference/match_logo_in_video.py \
    --video input_video.mp4 \
    --sample logo_template.jpg \
    --threshold 0.9 \
    --enable-segmentation \
    --sam-checkpoint sam2_b.pt \
    --sam-backend sam2 \
    --sam-config sam2_hiera_b+.yaml

# Chạy với MobileSAM (nhẹ hơn)
python scripts/inference/match_logo_in_video.py \
    --video video.mp4 \
    --sample logo.jpg \
    --enable-segmentation \
    --sam-checkpoint mobile_sam.pt \
    --sam-backend mobilesam
```

### Models cần cho PP1

| Model | File | Kích thước | Vai trò |
|---|---|---|---|
| YOLOv11 Detect | `weights/yolov11s_model.pt` | ~39MB | Phát hiện bbox |
| SigLIP2 | `weights/siglip2_model.pth` | ~357MB | Embedding xác minh |
| SAM 2 Base | `sam2_b.pt` (tải riêng) | ~162MB | Segment mask |
| FAISS Index | `logo_index.faiss` (sinh từ Bước 4) | ~750MB | Vector database |

### Đặc điểm PP1

| Chỉ số | Giá trị |
|---|---|
| FPS (RTX 3050) | ~3-4 FPS |
| VRAM tối thiểu | ≥ 8GB |
| Chất lượng Mask | ⭐⭐⭐⭐⭐ Pixel-perfect |
| Cần polygon data để train? | ❌ Không |
| Phù hợp | Offline processing, highlight reel |

---

## 🟩 Phương pháp 2: YOLOv11-seg + SigLIP2 ⭐

> **Ưu tiên tốc độ — Phù hợp xử lý realtime, phát sóng trực tiếp.**

### Quy trình riêng PP2

PP2 yêu cầu **2 bước training bổ sung** sau Bước 2 (Quy trình Chung) để tạo model YOLO-seg:

```
Sau Bước 2 (Train YOLOv11 Detect)
    │
    ▼
┌──────────────────────────────────────────────────────────┐
│  PP2 Bước A: Tạo Polygon Mask tự động bằng SAM          │
│              run_sam.py                                   │
│              (Chỉ chạy 1 lần để tạo dataset)            │
│                     │                                     │
│                     ▼                                     │
│  PP2 Bước B: Train YOLOv11-seg trên polygon dataset     │
│              train_yolo_seg.py                            │
│              → Output: best.pt (model seg)               │
└──────────────────────────────────────────────────────────┘
    │
    ▼
Tiếp tục Bước 3, 4 (Quy trình Chung) → Chạy Pipeline
```

#### PP2 Bước A: Tạo Polygon Mask tự động bằng SAM

> **Script:** [`run_sam.py`](scripts/training/run_sam.py)

```bash
python scripts/training/run_sam.py
```

**Pipeline tự động:**
1. Đọc bbox từ XML annotation (VOC format)
2. Dùng SAM sinh segmentation mask từ bbox prompt
3. Kiểm tra chất lượng mask (fill ratio 30%-150%)
4. Chuyển đổi mask → polygon → YOLO-seg format
5. Tự động chia train/val (80/20)

> ⚠️ Bước này chỉ chạy **1 lần** để tạo polygon dataset. Sau đó SAM không cần thiết khi inference.

#### PP2 Bước B: Train YOLOv11-seg

> **Script:** [`train_yolo_seg.py`](scripts/training/train_yolo_seg.py)

```bash
python scripts/training/train_yolo_seg.py
```

| Cấu hình | Giá trị |
|---|---|
| Model base | `yolo11s-seg.pt` |
| Image size | 640 |
| Epochs | 100 |
| Output | `weights/yolov11s_seg.pt` |

### Luồng xử lý PP2 (per frame)

```
Frame ──▶ YOLOv11-seg ──▶ Bounding Box + Polygon Mask (cùng lúc)
                                      │
                                      ▼
                              Crop mỗi Box ──▶ SigLIP2 Embedding (512-d)
                                                        │
                                                        ▼
                                                FAISS Cosine Search
                                                        │
                                          ┌─────────────┴──────────────┐
                                          ▼                            ▼
                                  Cosine ≥ 0.91                Cosine < 0.91
                                  (Logo hợp lệ)               (Loại bỏ)
                                          │
                                          ▼
                                  Pixelate vùng Mask (có sẵn từ YOLO-seg)
                                  → Không cần gọi SAM
                                          │
                                          ▼
                                     Frame Output
```

### Chạy PP2

> **Script:** [`pixelate_yolo_siglip2.py`](scripts/inference/pixelate_yolo_siglip2.py) ⭐

```bash
# Chạy cơ bản
python scripts/inference/pixelate_yolo_siglip2.py \
    --src input_video.mp4 \
    --dst output/result.mp4

# Chạy với cấu hình tùy chỉnh
python scripts/inference/pixelate_yolo_siglip2.py \
    --src video.mp4 \
    --dst output/result.mp4 \
    --yolo weights/yolov11s_model.pt \
    --model weights/siglip2_model.pth \
    --index logo_index.faiss \
    --meta logo_meta.npy \
    --sim-threshold 0.91 \
    --yolo-conf 0.25 \
    --imgsz 640 \
    --save-crops

# Chỉ pixelate một số brand cụ thể
python scripts/inference/pixelate_yolo_siglip2.py \
    --src video.mp4 --dst out.mp4 \
    --blocklist fun88 1xbet melbet
```

**Tham số quan trọng:**

| Tham số | Mặc định | Mô tả |
|---|---|---|
| `--sim-threshold` | 0.9 | Ngưỡng cosine similarity (khuyến nghị 0.91) |
| `--yolo-conf` | 0.25 | YOLO confidence threshold |
| `--top-k` | 3 | Top-K FAISS neighbors để vote |
| `--min-area` | 600 | Diện tích bbox tối thiểu (px²) |
| `--pixel-size` | 14 | Kích thước pixel cho pixelate |
| `--imgsz` | 640 | Input size cho YOLO |
| `--save-crops` | false | Lưu debug crop trước/sau pixelate |
| `--blocklist` | None (all) | Danh sách brand cần che |

### Models cần cho PP2

| Model | File | Kích thước | Vai trò |
|---|---|---|---|
| YOLOv11-seg | `weights/yolov11s_model.pt` | ~39MB | Detect + Segment |
| SigLIP2 | `weights/siglip2_model.pth` | ~357MB | Embedding xác minh |
| FAISS Index | `logo_index.faiss` (sinh từ Bước 4) | ~750MB | Vector database |

### Đặc điểm PP2

| Chỉ số | Giá trị |
|---|---|
| FPS (RTX 3050) | ~6.6 FPS |
| VRAM tối thiểu | ≥ 4GB |
| Chất lượng Mask | ⭐⭐⭐ Phụ thuộc polygon data |
| Cần SAM khi inference? | ❌ Không |
| Phù hợp | **Realtime, livestream, phát sóng** |

---

### 📊 Tổng hợp Quy trình 2 Phương pháp

```
                        ┌──────────────────────┐
                        │  Bước 1: Dataset     │
                        │  (SMVL_dataset)      │
                        └──────────┬───────────┘
                                   │
                        ┌──────────▼───────────┐
                        │  Bước 2: Train       │
                        │  YOLOv11 Detect      │
                        └──────────┬───────────┘
                                   │
               ┌───────────────────┼───────────────────┐
               │                   │                   │
       ┌───────▼────────┐         │          ┌────────▼────────┐
       │  PP2 Bước A:   │         │          │                 │
       │  SAM → Polygon │         │          │  (PP1 không     │
       │  (1 lần)       │         │          │   cần bước này) │
       └───────┬────────┘         │          └─────────────────┘
               │                  │
       ┌───────▼────────┐         │
       │  PP2 Bước B:   │         │
       │  Train         │         │
       │  YOLOv11-seg   │         │
       └───────┬────────┘         │
               │                  │
               └────────┬─────────┘
                        │
             ┌──────────▼───────────┐
             │  Bước 3: Crop +     │
             │  Train SigLIP2      │
             └──────────┬───────────┘
                        │
             ┌──────────▼───────────┐
             │  Bước 4: Build      │
             │  FAISS Index        │
             └──────────┬───────────┘
                        │
           ┌────────────┴────────────┐
           │                         │
   ┌───────▼────────┐      ┌────────▼────────┐
   │  PP1: Chạy     │      │  PP2: Chạy      │
   │  match_logo_   │      │  pixelate_yolo_ │
   │  in_video.py   │      │  siglip2.py ⭐  │
   └────────────────┘      └─────────────────┘
```

---

## 📊 Kết quả Đánh giá

### Benchmark trên 4.467 Frames Video bóng đá

| Chỉ số | YOLOv11-seg (Không lọc) | YOLOv11-seg + SigLIP2 |
|---|:---:|:---:|
| Số khung hình | 4.467 | 4.467 |
| Tổng thời gian | 538,13s | 675,44s |
| FPS | ≈ 8.30 | ≈ 6.61 |
| Box/Mask hợp lệ | 17.785 | **11.661** *(chuẩn 100%)* |
| FAISS overhead | — | ~0.35ms/crop |
| False Positive bị loại | — | **6.124 (34.4%)** |

> **Nhận xét:** SigLIP2 loại bỏ **6.124 khung lỗi (34.4%)** — đây là các vật thể bị YOLO nhận nhầm (biển quảng cáo, bóng phản quang, áo cầu thủ...). Chi phí overhead chỉ ~0.35ms/crop.

### SigLIP2 Embedding Quality

**Kết quả Retrieval:**
- Top-1 Accuracy: **99.95%**
- 222/227 brands đạt F1 ≥ 95%
- Ngưỡng cosine tối ưu: **0.91**

---

## 🔧 Công cụ Hỗ trợ

### Phân tích Misclassified Crops

> **Script:** [`inspect_misclassified.py`](scripts/evaluation/inspect_misclassified.py)

```bash
python scripts/evaluation/inspect_misclassified.py
# Output: output/misclassified/ (panel ảnh query vs top-K gallery)
```

### Detect Objects trong Video (YOLO only)

> **Script:** [`detect_video.py`](scripts/inference/detect_video.py)

```bash
python scripts/inference/detect_video.py --video test.mp4 --conf 0.5 --save-crops --save-video
```

---

## ⚡ Tối ưu Hiệu năng

### Export TensorRT (tăng tốc 3-5x)

> **Script:** [`export_yolo_tensorrt.py`](scripts/utils/export_yolo_tensorrt.py)

```bash
python scripts/utils/export_yolo_tensorrt.py \
    --model weights/yolov11s_model.pt \
    --half \
    --imgsz 640
```

> ⚠️ File `.engine` phụ thuộc GPU cụ thể. Nếu đổi GPU hoặc driver cần export lại.

### FP16 Inference

Module `SigLIP2Verifier` trong [`siglip2_verifier.py`](scripts/inference/siglip2_verifier.py) hỗ trợ FP16 tự động trên CUDA:

```python
from scripts.inference.siglip2_verifier import SigLIP2Verifier

verifier = SigLIP2Verifier(
    model_path="weights/siglip2_model.pth",
    device="cuda",
    use_fp16=True,  # Mặc định True trên CUDA
)
```

### Hướng mở rộng tối ưu

1. **Batch-Embedding Stream:** Encode tất cả crop trong 1 frame cùng lúc (đã implement trong `get_embeddings_from_bgr_list()`)
2. **Smooth Polygon:** Làm mượt đường polygon trong dataset gốc trước khi train YOLO-seg
3. **TensorRT FP16:** Hạ model `.pt` xuống `.engine` để đạt realtime trên Edge GPU

---

## 🖥 Cấu hình Hệ thống

### Hardware đã kiểm thử

| Component | Spec |
|---|---|
| GPU | NVIDIA GeForce RTX 3050 (4GB VRAM) |
| CPU | Intel Core i7 |
| RAM | 16GB |

### Cấu hình Pipeline

Chỉnh sửa file [`blocklist.yaml`](blocklist.yaml) để quản lý danh sách brand cần che:

```yaml
blocklist:
  enabled: true
  classes:
    - id: 192
      name: "fun88"
      pixelate: true
      pixel_size: 14
    - id: 105
      name: "1xbet"
      pixelate: true
      pixel_size: 14
```

---

## 📄 Tham khảo

- [Ultralytics YOLOv11](https://docs.ultralytics.com/)
- [open_clip — SigLIP2](https://github.com/mlfoundations/open_clip)
- [FAISS — Facebook AI Similarity Search](https://github.com/facebookresearch/faiss)
- [Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything)

---

## 📝 License

Dự án này được phát triển cho mục đích nghiên cứu và ứng dụng nội bộ trong giám sát bản quyền truyền hình.

---

> **Lưu ý:** Các file weights (`.pt`, `.pth`) đã bao gồm trong thư mục `weights/`. File FAISS index (`.faiss`) được sinh ra từ Bước 4 trong quy trình training.
