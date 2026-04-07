"""
Export YOLOv11s sang TensorRT (.engine) để tăng tốc inference ~3-5x.

LƯU Ý:
- File .engine phụ thuộc GPU (chỉ chạy trên GPU đã build).
- Nếu đổi GPU hoặc driver, cần export lại.
- Cần cài: pip install tensorrt (hoặc cài TensorRT từ NVIDIA)

Usage:
    python export_yolo_tensorrt.py
    python export_yolo_tensorrt.py --model model/yolov11s_model.pt --half
"""

import argparse
from pathlib import Path

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser(description="Export YOLO to TensorRT engine")
    ap.add_argument("--model", type=str, default="model/yolov11s_model.pt",
                    help="Path to YOLO .pt checkpoint")
    ap.add_argument("--imgsz", type=int, default=640,
                    help="Input size (640 = YOLO default)")
    ap.add_argument("--half", action="store_true", default=True,
                    help="FP16 half precision (faster, recommended)")
    ap.add_argument("--no-half", action="store_true",
                    help="Disable FP16, use FP32")
    ap.add_argument("--workspace", type=float, default=4.0,
                    help="TensorRT workspace size in GB")
    ap.add_argument("--device", type=str, default="0",
                    help="GPU device for export (0, 1, ...)")
    args = ap.parse_args()

    model_path = Path(args.model)
    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}")

    use_half = args.half and not args.no_half
    print(f"\n{'='*60}")
    print("YOLO TensorRT Export")
    print(f"{'='*60}")
    print(f"  Input:      {model_path}")
    print(f"  Output:     {model_path.with_suffix('.engine')}")
    print(f"  imgsz:      {args.imgsz}")
    print(f"  FP16:       {use_half}")
    print(f"  Workspace:  {args.workspace} GB")
    print(f"  Device:     {args.device}")
    print(f"{'='*60}\n")

    model = YOLO(str(model_path))
    model.export(
        format="engine",
        imgsz=args.imgsz,
        half=use_half,
        workspace=args.workspace,
        device=args.device,
    )

    # Ultralytics lưu .engine cùng thư mục với .pt
    engine_path = model_path.with_suffix(".engine")
    if not engine_path.is_file():
        engine_path = model_path.parent / (model_path.stem + ".engine")
    print(f"\n  Engine file: {engine_path}")

    print(f"\n✅ Done. Chạy với TensorRT:")
    print(f"   python test_sam3_segment.py --video <video> --sample <logo> --yolo-model {engine_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
