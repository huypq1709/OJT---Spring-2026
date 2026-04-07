"""
Phase 1: Train YOLO11 Detection — Chạy trên server (Linux/Windows).

Usage:
    python train_yolo_detect_server.py --data /path/to/data.yaml
    python train_yolo_detect_server.py --data data.yaml --epochs 100 --batch 8 --device 0
    python train_yolo_detect_server.py --data data.yaml --resume

    # Chạy nền (Linux)
    nohup python train_yolo_detect_server.py --data data.yaml > train.log 2>&1 &
"""

import argparse
import os
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO


def fix_data_yaml(path: str, output_dir: Path) -> str:
    """Sửa path Windows trong data.yaml khi chạy trên server."""
    p = Path(path).resolve()
    if not p.exists():
        return path

    with open(p, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not cfg:
        return path

    changed = False
    root = p.parent

    if "path" in cfg:
        v = str(cfg["path"])
        if "\\" in v or (len(v) >= 2 and v[1] == ":"):
            cfg["path"] = str(root)
            changed = True

    def _fix(key: str, default: str) -> None:
        nonlocal changed
        if key in cfg:
            v = str(cfg[key])
            if "\\" in v or (len(v) >= 2 and v[1] == ":"):
                cfg[key] = default
                changed = True

    _fix("train", "images/train")
    _fix("val", "images/val")
    _fix("test", "images/test")

    if not changed:
        return path

    out = output_dir / f"{p.stem}_fixed.yaml"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, default_flow_style=False, allow_unicode=True)
    print(f"  Fixed data.yaml -> {out}")
    return str(out)


def main():
    ap = argparse.ArgumentParser(description="Train YOLO11 Detection trên server")
    ap.add_argument("--data", "-d", type=str, required=True, help="Path to data.yaml")
    ap.add_argument("--model", "-m", type=str, default="yolo11s.pt",
                    help="Base model: yolo11n.pt, yolo11s.pt, yolo11m.pt")
    ap.add_argument("--epochs", "-e", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=1280, help="1280 cho TV/HD, 640 cho generic")
    ap.add_argument("--batch", "-b", type=int, default=8,
                    help="Batch size (multi-GPU: bội số của số GPU)")
    ap.add_argument("--device", type=str, default=None,
                    help="GPU: 0, 0,1 hoặc cpu (mặc định: auto)")
    ap.add_argument("--project", type=str, default="runs/detect")
    ap.add_argument("--name", type=str, default="smvl_logo_detect")
    ap.add_argument("--resume", action="store_true", help="Resume từ last.pt")
    ap.add_argument("--fix-yaml", action="store_true",
                    help="Sửa path Windows trong data.yaml")
    args = ap.parse_args()

    data_yaml = Path(args.data)
    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml not found: {data_yaml}")

    if args.fix_yaml:
        output_dir = Path(args.project) / args.name
        data_yaml = fix_data_yaml(str(data_yaml), output_dir)

    n_gpu = torch.cuda.device_count()
    device = args.device
    if device is None:
        device = "0" if n_gpu >= 1 else "cpu"
    if n_gpu >= 2 and device != "cpu" and "," not in device:
        device = "0,1"

    # Multi-GPU: batch phải là bội số của số GPU
    if "," in device and args.batch > 0:
        n = len(device.split(","))
        if args.batch % n != 0:
            args.batch = (args.batch // n) * n or n
            print(f"  Adjusted batch to {args.batch} (multiple of {n} GPUs)")

    print("=" * 60)
    print("GPU INFO")
    print("=" * 60)
    if n_gpu > 0:
        for i in range(n_gpu):
            name = torch.cuda.get_device_name(i)
            props = torch.cuda.get_device_properties(i)
            mem = getattr(props, "total_memory", getattr(props, "total_mem", 0)) / 1024**3
            print(f"  GPU {i}: {name} ({mem:.1f} GB)")
        print(f"  Total: {n_gpu} GPU(s)")
    else:
        print("  No GPU, using CPU")
    print("=" * 60)

    if args.resume:
        ckpt = Path(args.project) / args.name / "weights" / "last.pt"
        if not ckpt.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
        print(f"\nResume from: {ckpt}")
        model = YOLO(str(ckpt))
    else:
        print(f"\nBase model: {args.model}")
        model = YOLO(args.model)

    print("\n" + "=" * 60)
    print("TRAIN YOLO11 DETECTION")
    print("=" * 60)
    print(f"  data:    {data_yaml}")
    print(f"  epochs:  {args.epochs}")
    print(f"  imgsz:   {args.imgsz}")
    print(f"  batch:   {args.batch}")
    print(f"  device:  {device}")
    print("=" * 60)

    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        warmup_epochs=3,
        patience=20,
        save_period=10,
        project=args.project,
        name=args.name,
        exist_ok=True,
        resume=args.resume,
        amp=True,
        workers=4,
        mosaic=1.0,
        mixup=0.1,
        scale=0.5,
        fliplr=0.5,
        flipud=0.0,
        verbose=True,
    )

    metrics = model.val()
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  mAP50:    {metrics.box.map50:.4f}")
    print(f"  mAP50-95: {metrics.box.map:.4f}")
    print("=" * 60)
    print("\nDone!")


if __name__ == "__main__":
    main()
