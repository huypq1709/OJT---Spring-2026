# -*- coding: utf-8 -*-
"""
Train SigLIP2 Classifier — Single GPU, optimized for speed.

Architecture:
  - Backbone: open_clip ViT-B-16-SigLIP2-256 (frozen → unfreeze later)
  - Head: LN(768) → Linear(768→512) → ReLU → Dropout → Linear(512→num_classes)

Dataset: cropped_classes/ (223 classes, ~380k images)

Usage:
    python train_siglip2_server.py
    python train_siglip2_server.py --epochs 30 --batch-size 192 --unfreeze-epoch 10
"""

import argparse
import os
import gc
import json
import time
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torchvision import transforms
from torchvision.transforms import functional as TF
from PIL import Image

torch.cuda.empty_cache()
gc.collect()


# ═══════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════
def parse_args():
    parser = argparse.ArgumentParser(description="Train SigLIP2 — Single GPU")
    parser.add_argument("--data-dir", default="cropped_classes",
                        help="Root folder with class subfolders")
    parser.add_argument("--output-dir", default="train_output",
                        help="Directory to save checkpoints and logs")
    parser.add_argument("--resume", default=None,
                        help="Path to checkpoint to resume training")
    parser.add_argument("--pretrained-weights", default=None,
                        help="Path to existing siglip2_model.pth to init head")

    # Training
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=192,
                        help="Batch size for single GPU")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lr-backbone", type=float, default=1e-5,
                        help="LR for backbone when unfrozen")
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--val-ratio", type=float, default=0.05,
                        help="Fraction of data used for validation")

    # Backbone unfreezing
    parser.add_argument("--unfreeze-epoch", type=int, default=8,
                        help="Epoch to unfreeze backbone (0=never)")

    # Early stopping
    parser.add_argument("--patience", type=int, default=7,
                        help="Early stopping patience")

    # GPU
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU ID to use")

    # Compile (PyTorch 2.0+)
    parser.add_argument("--compile", action="store_true",
                        help="Use torch.compile for faster training")

    return parser.parse_args()


# ═══════════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════════
class LogoCropDataset(Dataset):
    """
    Load images from class subfolders with letterbox padding.
    Optimized: pure PIL pipeline, no cv2 conversion overhead.
    """
    EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(self, root_dir, classes=None, img_size=256, augment=False):
        self.root_dir = root_dir
        self.img_size = img_size
        self.augment = augment

        # Discover classes
        if classes is None:
            self.classes = sorted([
                d for d in os.listdir(root_dir)
                if os.path.isdir(os.path.join(root_dir, d))
            ])
        else:
            self.classes = classes

        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        # Collect samples — store as list of (path, label)
        self.samples = []
        for cls_name in self.classes:
            cls_dir = os.path.join(root_dir, cls_name)
            if not os.path.isdir(cls_dir):
                continue
            label = self.class_to_idx[cls_name]
            for fname in os.listdir(cls_dir):
                if Path(fname).suffix.lower() in self.EXTS:
                    self.samples.append((
                        os.path.join(cls_dir, fname),
                        label
                    ))

        # Augmentation transforms
        if augment:
            self.aug = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomAffine(
                    degrees=10, translate=(0.05, 0.05),
                    scale=(0.9, 1.1), fill=128
                ),
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.2,
                    saturation=0.15, hue=0.05
                ),
            ])
        else:
            self.aug = None

        # Normalize (SigLIP2 standard)
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def letterbox_padding(self, image: Image.Image) -> Image.Image:
        """Letterbox: resize keeping aspect ratio + neutral gray padding. Pure PIL."""
        w, h = image.size
        ratio = self.img_size / max(h, w)
        new_w, new_h = int(w * ratio), int(h * ratio)

        # Use LANCZOS for downscale, BICUBIC for upscale
        resample = Image.LANCZOS if ratio < 1.0 else Image.BICUBIC
        resized = image.resize((new_w, new_h), resample)

        # Create gray canvas and paste centered
        canvas = Image.new("RGB", (self.img_size, self.img_size), (128, 128, 128))
        left = (self.img_size - new_w) // 2
        top = (self.img_size - new_h) // 2
        canvas.paste(resized, (left, top))
        return canvas

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (self.img_size, self.img_size), (128, 128, 128))

        img = self.letterbox_padding(img)

        if self.aug is not None:
            img = self.aug(img)

        return self.norm(img), label


# ═══════════════════════════════════════════════════════════════════
# Model
# ═══════════════════════════════════════════════════════════════════
class SigLIP2Classifier(nn.Module):
    def __init__(self, num_classes: int, dropout: float = 0.3):
        super().__init__()
        import open_clip

        # Load pretrained backbone
        model, _, _ = open_clip.create_model_and_transforms(
            "ViT-B-16-SigLIP2-256", pretrained="webli"
        )
        self.backbone = model.visual

        # Freeze backbone initially
        for p in self.backbone.parameters():
            p.requires_grad = False

        # Classification head
        self.head = nn.Sequential(
            nn.LayerNorm(768),
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        with torch.set_grad_enabled(self.backbone.training):
            features = self.backbone(x)  # (B, 768)
        return self.head(features)       # (B, num_classes)

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()
        print("  🔒 Backbone FROZEN")

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = True
        self.backbone.train()
        print("  🔓 Backbone UNFROZEN")


# ═══════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════
def train_one_epoch(model, loader, criterion, optimizer, scaler, device, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    from tqdm import tqdm
    pbar = tqdm(loader, desc=f"  Train Epoch {epoch}", leave=False)
    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda"):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        if batch_idx % 10 == 0:
            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                acc=f"{100. * correct / total:.1f}%"
            )

    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


@torch.no_grad()
def validate(model, loader, criterion, device, collect_predictions=False):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    from tqdm import tqdm
    pbar = tqdm(loader, desc="  Validate", leave=False)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast("cuda"):
            outputs = model(images)
            loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        if collect_predictions:
            all_preds.extend(predicted.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    val_loss = running_loss / total
    val_acc = 100. * correct / total

    if collect_predictions:
        return val_loss, val_acc, all_preds, all_labels
    return val_loss, val_acc


def main():
    args = parse_args()

    # ── GPU setup (single GPU) ─────────────────────────────────
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Performance flags
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    print(f"{'='*60}")
    print(f"  SigLIP2 Training — Single GPU")
    print(f"{'='*60}")
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_mem / 1e9
        print(f"  GPU: {name} ({mem:.0f} GB)")
    else:
        print("  ⚠️  No GPU found, using CPU!")

    # Output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Dataset ──────────────────────────────────────────────────
    print(f"\n📂 Loading dataset from: {args.data_dir}")
    full_dataset = LogoCropDataset(
        args.data_dir, img_size=args.img_size, augment=True
    )
    num_classes = len(full_dataset.classes)
    total_samples = len(full_dataset)
    print(f"   Classes: {num_classes}")
    print(f"   Total samples: {total_samples:,}")

    # Save class list
    class_list_path = os.path.join(args.output_dir, "classes.json")
    with open(class_list_path, "w") as f:
        json.dump(full_dataset.classes, f, indent=2)
    print(f"   Saved class list → {class_list_path}")

    # Train/Val split — NO deepcopy, create separate dataset for val
    val_size = int(total_samples * args.val_ratio)
    train_size = total_samples - val_size

    generator = torch.Generator().manual_seed(42)
    indices = torch.randperm(total_samples, generator=generator).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    # Train subset uses the augmented dataset directly
    train_dataset = Subset(full_dataset, train_indices)

    # Val dataset: create a NEW lightweight dataset without augmentation
    val_base_dataset = LogoCropDataset(
        args.data_dir, classes=full_dataset.classes,
        img_size=args.img_size, augment=False
    )
    val_dataset = Subset(val_base_dataset, val_indices)

    print(f"   Train: {train_size:,} | Val: {val_size:,}")

    # Adjust num_workers based on platform
    num_workers = args.num_workers
    if os.name == 'nt' and num_workers > 4:
        num_workers = 4  # Windows multiprocessing limit
        print(f"   ⚠️  Windows detected, capping num_workers to {num_workers}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size * 2,  # Larger batch for val (no grad)
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
    )

    # ── Model ────────────────────────────────────────────────────
    print(f"\n🏗️  Building SigLIP2Classifier (num_classes={num_classes})")
    model = SigLIP2Classifier(num_classes=num_classes)

    # Optionally load pretrained head weights
    if args.pretrained_weights and os.path.isfile(args.pretrained_weights):
        print(f"   Loading pretrained weights: {args.pretrained_weights}")
        ckpt = torch.load(args.pretrained_weights, map_location="cpu", weights_only=False)
        from collections import OrderedDict
        clean = OrderedDict()
        for k, v in ckpt.items():
            clean[k.replace("module.", "")] = v
        missing, unexpected = model.load_state_dict(clean, strict=False)
        print(f"   Loaded (missing={len(missing)}, unexpected={len(unexpected)})")

    model = model.to(device)

    # Optional: torch.compile for PyTorch 2.0+
    if args.compile and hasattr(torch, "compile"):
        print("   ⚡ Using torch.compile() for acceleration")
        model = torch.compile(model)

    # ── Optimizer ────────────────────────────────────────────────
    # Initially only train head
    optimizer = optim.AdamW(
        model.head.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Cosine annealing scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.amp.GradScaler("cuda")

    # ── Resume ───────────────────────────────────────────────────
    start_epoch = 0
    best_val_acc = 0.0
    if args.resume and os.path.isfile(args.resume):
        print(f"\n🔄 Resuming from: {args.resume}")
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)

        # Handle DataParallel keys
        state_dict = ckpt["model_state_dict"]
        cleaned = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(cleaned)

        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_acc = ckpt.get("best_val_acc", 0.0)
        print(f"   Resuming from epoch {start_epoch}, best_val_acc={best_val_acc:.2f}%")

    # ── Training Loop ────────────────────────────────────────────
    patience_counter = 0
    log_path = os.path.join(args.output_dir, "training_log.csv")

    with open(log_path, "w") as f:
        f.write("epoch,train_loss,train_acc,val_loss,val_acc,lr,backbone_status\n")

    print(f"\n{'='*60}")
    print(f"  🚀 Starting training: {args.epochs} epochs")
    print(f"     Batch size: {args.batch_size} | LR: {args.lr}")
    print(f"     Unfreeze backbone at epoch: {args.unfreeze_epoch}")
    print(f"     Patience: {args.patience}")
    print(f"     Num workers: {num_workers}")
    print(f"     cudnn.benchmark: {torch.backends.cudnn.benchmark}")
    print(f"     TF32: {torch.backends.cuda.matmul.allow_tf32}")
    print(f"{'='*60}\n")

    t_start = time.time()

    # Access raw model (handles torch.compile wrapper)
    raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model

    for epoch in range(start_epoch, args.epochs):
        t_epoch = time.time()

        # Unfreeze backbone at specified epoch
        backbone_status = "frozen"
        if args.unfreeze_epoch > 0 and epoch == args.unfreeze_epoch:
            print(f"\n🔓 Epoch {epoch}: Unfreezing backbone!")
            raw_model.unfreeze_backbone()

            # Rebuild optimizer with backbone params at lower LR
            optimizer = optim.AdamW([
                {"params": raw_model.backbone.parameters(), "lr": args.lr_backbone},
                {"params": raw_model.head.parameters(), "lr": args.lr * 0.5},
            ], weight_decay=args.weight_decay)

            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.epochs - epoch, eta_min=1e-7
            )

        if args.unfreeze_epoch > 0 and epoch >= args.unfreeze_epoch:
            backbone_status = "unfrozen"

        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, epoch + 1
        )

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Step scheduler
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        elapsed = time.time() - t_epoch

        print(
            f"  Epoch {epoch+1:02d}/{args.epochs} | "
            f"Train: loss={train_loss:.4f} acc={train_acc:.1f}% | "
            f"Val: loss={val_loss:.4f} acc={val_acc:.1f}% | "
            f"LR={current_lr:.2e} | {backbone_status} | "
            f"{elapsed:.0f}s"
        )

        # Log
        with open(log_path, "a") as f:
            f.write(f"{epoch+1},{train_loss:.5f},{train_acc:.2f},"
                    f"{val_loss:.5f},{val_acc:.2f},"
                    f"{current_lr:.2e},{backbone_status}\n")

        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0

            # Save full checkpoint (for resume)
            ckpt_path = os.path.join(args.output_dir, "best_checkpoint.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": raw_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_acc": best_val_acc,
                "num_classes": num_classes,
                "classes": full_dataset.classes,
            }, ckpt_path)

            # Save model-only weights (for deployment)
            weights_path = os.path.join(args.output_dir, "siglip2_model.pth")
            torch.save(raw_model.state_dict(), weights_path)

            print(f"  ✅ New best! Val Acc={val_acc:.2f}% → saved to {weights_path}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\n  ⏹️  Early stopping at epoch {epoch+1} (patience={args.patience})")
                break

        # Save periodic checkpoint
        if (epoch + 1) % 5 == 0:
            periodic_path = os.path.join(args.output_dir, f"checkpoint_epoch{epoch+1}.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": raw_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_acc": best_val_acc,
                "num_classes": num_classes,
                "classes": full_dataset.classes,
            }, periodic_path)

        # Clear cache between epochs
        torch.cuda.empty_cache()

    total_time = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"  ✅ Training complete!")
    print(f"     Best Val Accuracy: {best_val_acc:.2f}%")
    print(f"     Total time: {total_time/60:.1f} min")
    print(f"     Output dir: {args.output_dir}")
    print(f"     Model weights: {args.output_dir}/siglip2_model.pth")
    print(f"{'='*60}")

    # ── Generate Report ──────────────────────────────────────────
    print(f"\n📊 Generating training report...")

    # Load best model for final evaluation
    best_weights = os.path.join(args.output_dir, "siglip2_model.pth")
    if os.path.isfile(best_weights):
        raw_model.load_state_dict(
            torch.load(best_weights, map_location=device, weights_only=False)
        )
        print("   Loaded best model for final evaluation")

    # Collect predictions on val set
    val_loss_final, val_acc_final, all_preds, all_labels = validate(
        model, val_loader, criterion, device, collect_predictions=True
    )

    generate_report(
        log_path=log_path,
        output_dir=args.output_dir,
        classes=full_dataset.classes,
        all_preds=all_preds,
        all_labels=all_labels,
        best_val_acc=best_val_acc,
        val_loss_final=val_loss_final,
        val_acc_final=val_acc_final,
        total_time=total_time,
        num_classes=num_classes,
        total_samples=total_samples,
        train_size=train_size,
        val_size=val_size,
        args=args,
    )
    print(f"   📁 All reports saved to: {args.output_dir}/")


# ═══════════════════════════════════════════════════════════════════
# Report Generation
# ═══════════════════════════════════════════════════════════════════
def generate_report(
    log_path, output_dir, classes, all_preds, all_labels,
    best_val_acc, val_loss_final, val_acc_final,
    total_time, num_classes, total_samples,
    train_size, val_size, args,
):
    """Generate comprehensive training report with charts and statistics."""
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    from sklearn.metrics import (
        confusion_matrix, classification_report,
        precision_recall_fscore_support,
    )

    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "figure.dpi": 150,
    })

    report_dir = os.path.join(output_dir, "report")
    os.makedirs(report_dir, exist_ok=True)

    # ── 1. Parse training log ────────────────────────────────────
    epochs_data = []
    with open(log_path, "r") as f:
        header = f.readline()  # skip header
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 6:
                epochs_data.append({
                    "epoch": int(parts[0]),
                    "train_loss": float(parts[1]),
                    "train_acc": float(parts[2]),
                    "val_loss": float(parts[3]),
                    "val_acc": float(parts[4]),
                    "lr": float(parts[5]),
                    "backbone": parts[6] if len(parts) > 6 else "frozen",
                })

    if not epochs_data:
        print("   ⚠️  No training data found in log, skipping report.")
        return

    ep = [d["epoch"] for d in epochs_data]
    train_losses = [d["train_loss"] for d in epochs_data]
    val_losses = [d["val_loss"] for d in epochs_data]
    train_accs = [d["train_acc"] for d in epochs_data]
    val_accs = [d["val_acc"] for d in epochs_data]
    lrs = [d["lr"] for d in epochs_data]

    # ── 2. Training Curves (Loss + Accuracy) ─────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle("SigLIP2 Training Report", fontsize=18, fontweight="bold", y=0.98)

    # 2a. Loss curves
    ax1 = axes[0, 0]
    ax1.plot(ep, train_losses, "o-", color="#2196F3", linewidth=2, markersize=4, label="Train Loss")
    ax1.plot(ep, val_losses, "s-", color="#F44336", linewidth=2, markersize=4, label="Val Loss")
    # Mark unfreeze epoch
    if args.unfreeze_epoch > 0 and args.unfreeze_epoch <= max(ep):
        ax1.axvline(x=args.unfreeze_epoch, color="#9C27B0", linestyle="--",
                    alpha=0.7, label=f"Unfreeze (epoch {args.unfreeze_epoch})")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss Curves")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    # 2b. Accuracy curves
    ax2 = axes[0, 1]
    ax2.plot(ep, train_accs, "o-", color="#4CAF50", linewidth=2, markersize=4, label="Train Acc")
    ax2.plot(ep, val_accs, "s-", color="#FF9800", linewidth=2, markersize=4, label="Val Acc")
    if args.unfreeze_epoch > 0 and args.unfreeze_epoch <= max(ep):
        ax2.axvline(x=args.unfreeze_epoch, color="#9C27B0", linestyle="--",
                    alpha=0.7, label=f"Unfreeze (epoch {args.unfreeze_epoch})")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Accuracy Curves")
    ax2.legend(loc="lower right")
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    # 2c. Learning Rate schedule
    ax3 = axes[1, 0]
    ax3.plot(ep, lrs, "D-", color="#673AB7", linewidth=2, markersize=4)
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Learning Rate")
    ax3.set_title("Learning Rate Schedule")
    ax3.set_yscale("log")
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    # 2d. Train vs Val gap (overfitting indicator)
    ax4 = axes[1, 1]
    gap = [t - v for t, v in zip(train_accs, val_accs)]
    colors = ["#4CAF50" if g < 5 else "#FF9800" if g < 15 else "#F44336" for g in gap]
    ax4.bar(ep, gap, color=colors, alpha=0.8)
    ax4.axhline(y=0, color="black", linewidth=0.5)
    ax4.set_xlabel("Epoch")
    ax4.set_ylabel("Train - Val Acc (%)")
    ax4.set_title("Overfitting Gap")
    ax4.grid(True, alpha=0.3, axis="y")
    ax4.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    curves_path = os.path.join(report_dir, "training_curves.png")
    fig.savefig(curves_path, bbox_inches="tight")
    plt.close(fig)
    print(f"   ✅ Training curves → {curves_path}")

    # ── 3. Per-class Accuracy ────────────────────────────────────
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, average=None, zero_division=0
    )
    per_class_acc = []
    for i in range(num_classes):
        class_mask = [l == i for l in all_labels]
        class_total = sum(class_mask)
        if class_total > 0:
            class_correct = sum(1 for l, p in zip(all_labels, all_preds) if l == i and p == i)
            per_class_acc.append(100.0 * class_correct / class_total)
        else:
            per_class_acc.append(0.0)

    # Sort by accuracy for the bar chart
    sorted_indices = np.argsort(per_class_acc)

    # Bar chart: show bottom-20 and top-20 classes (or all if <= 40)
    if num_classes > 40:
        show_indices = list(sorted_indices[:20]) + list(sorted_indices[-20:])
        chart_title = "Per-class Accuracy (Bottom 20 + Top 20)"
    else:
        show_indices = list(sorted_indices)
        chart_title = "Per-class Accuracy"

    fig2, ax_pc = plt.subplots(figsize=(14, max(6, len(show_indices) * 0.28)))
    show_names = [classes[i] for i in show_indices]
    show_accs = [per_class_acc[i] for i in show_indices]
    bar_colors = ["#F44336" if a < 50 else "#FF9800" if a < 80 else "#4CAF50" for a in show_accs]

    bars = ax_pc.barh(range(len(show_indices)), show_accs, color=bar_colors, height=0.7)
    ax_pc.set_yticks(range(len(show_indices)))
    ax_pc.set_yticklabels(show_names, fontsize=8)
    ax_pc.set_xlabel("Accuracy (%)")
    ax_pc.set_title(chart_title, fontweight="bold")
    ax_pc.set_xlim(0, 105)
    ax_pc.axvline(x=50, color="red", linestyle=":", alpha=0.4)
    ax_pc.axvline(x=80, color="orange", linestyle=":", alpha=0.4)
    ax_pc.grid(True, alpha=0.2, axis="x")

    # Add value labels on bars
    for bar, acc in zip(bars, show_accs):
        ax_pc.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                   f"{acc:.1f}%", va="center", fontsize=7)

    plt.tight_layout()
    perclass_path = os.path.join(report_dir, "per_class_accuracy.png")
    fig2.savefig(perclass_path, bbox_inches="tight")
    plt.close(fig2)
    print(f"   ✅ Per-class accuracy → {perclass_path}")

    # ── 4. Confusion Matrix (top confused pairs) ─────────────────
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))

    # If too many classes, show top-N most confused class pairs instead of full matrix
    if num_classes > 30:
        # Find the most confused pairs
        confused_pairs = []
        for i in range(num_classes):
            for j in range(num_classes):
                if i != j and cm[i][j] > 0:
                    confused_pairs.append((i, j, cm[i][j]))
        confused_pairs.sort(key=lambda x: -x[2])
        top_confused = confused_pairs[:30]

        # Get unique classes involved
        confused_classes = sorted(set(
            [p[0] for p in top_confused] + [p[1] for p in top_confused]
        ))

        if len(confused_classes) > 1:
            sub_cm = cm[np.ix_(confused_classes, confused_classes)]
            sub_names = [classes[i][:15] for i in confused_classes]

            fig3, ax_cm = plt.subplots(
                figsize=(max(8, len(confused_classes)*0.5),
                         max(6, len(confused_classes)*0.45))
            )
            im = ax_cm.imshow(sub_cm, cmap="YlOrRd", aspect="auto")
            ax_cm.set_xticks(range(len(confused_classes)))
            ax_cm.set_yticks(range(len(confused_classes)))
            ax_cm.set_xticklabels(sub_names, rotation=45, ha="right", fontsize=7)
            ax_cm.set_yticklabels(sub_names, fontsize=7)
            ax_cm.set_xlabel("Predicted")
            ax_cm.set_ylabel("True")
            ax_cm.set_title("Confusion Matrix (Most Confused Classes)", fontweight="bold")
            plt.colorbar(im, ax=ax_cm, shrink=0.8)

            # Add text annotations for non-zero cells
            for ii in range(len(confused_classes)):
                for jj in range(len(confused_classes)):
                    val = sub_cm[ii, jj]
                    if val > 0:
                        ax_cm.text(jj, ii, str(val), ha="center", va="center",
                                   fontsize=6, color="black" if val < sub_cm.max()/2 else "white")

            plt.tight_layout()
            cm_path = os.path.join(report_dir, "confusion_matrix.png")
            fig3.savefig(cm_path, bbox_inches="tight")
            plt.close(fig3)
            print(f"   ✅ Confusion matrix → {cm_path}")
    else:
        # Small enough for full matrix
        fig3, ax_cm = plt.subplots(figsize=(max(10, num_classes*0.4),
                                            max(8, num_classes*0.35)))
        im = ax_cm.imshow(cm, cmap="YlOrRd", aspect="auto")
        ax_cm.set_xticks(range(num_classes))
        ax_cm.set_yticks(range(num_classes))
        ax_cm.set_xticklabels(classes, rotation=45, ha="right", fontsize=7)
        ax_cm.set_yticklabels(classes, fontsize=7)
        ax_cm.set_xlabel("Predicted")
        ax_cm.set_ylabel("True")
        ax_cm.set_title("Confusion Matrix", fontweight="bold")
        plt.colorbar(im, ax=ax_cm, shrink=0.8)
        plt.tight_layout()
        cm_path = os.path.join(report_dir, "confusion_matrix.png")
        fig3.savefig(cm_path, bbox_inches="tight")
        plt.close(fig3)
        print(f"   ✅ Confusion matrix → {cm_path}")

    # ── 5. Classification Report (text) ──────────────────────────
    cls_report = classification_report(
        all_labels, all_preds,
        target_names=classes,
        digits=3,
        zero_division=0,
    )
    report_txt_path = os.path.join(report_dir, "classification_report.txt")
    with open(report_txt_path, "w", encoding="utf-8") as f:
        f.write(cls_report)
    print(f"   ✅ Classification report → {report_txt_path}")

    # ── 6. Summary Report ────────────────────────────────────────
    # Find worst / best classes
    worst_5 = sorted_indices[:5]
    best_5 = sorted_indices[-5:][::-1]

    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="weighted", zero_division=0
    )

    # Count per-class samples
    label_counts = Counter(all_labels)

    summary_path = os.path.join(report_dir, "summary_report.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("  SigLIP2 TRAINING SUMMARY REPORT\n")
        f.write("=" * 70 + "\n\n")

        f.write("📋 DATASET\n")
        f.write(f"   Total samples:  {total_samples:,}\n")
        f.write(f"   Train samples:  {train_size:,}\n")
        f.write(f"   Val samples:    {val_size:,}\n")
        f.write(f"   Num classes:    {num_classes}\n\n")

        f.write("⚙️  HYPERPARAMETERS\n")
        f.write(f"   Epochs trained: {len(epochs_data)}\n")
        f.write(f"   Batch size:     {args.batch_size}\n")
        f.write(f"   Learning rate:  {args.lr}\n")
        f.write(f"   LR backbone:    {args.lr_backbone}\n")
        f.write(f"   Weight decay:   {args.weight_decay}\n")
        f.write(f"   Image size:     {args.img_size}\n")
        f.write(f"   Unfreeze epoch: {args.unfreeze_epoch}\n")
        f.write(f"   Label smooth:   0.1\n\n")

        f.write("📈 RESULTS\n")
        f.write(f"   Best Val Accuracy:     {best_val_acc:.2f}%\n")
        f.write(f"   Final Val Loss:        {val_loss_final:.4f}\n")
        f.write(f"   Final Val Accuracy:    {val_acc_final:.2f}%\n")
        f.write(f"   Macro Precision:       {macro_p:.4f}\n")
        f.write(f"   Macro Recall:          {macro_r:.4f}\n")
        f.write(f"   Macro F1-score:        {macro_f1:.4f}\n")
        f.write(f"   Weighted Precision:    {weighted_p:.4f}\n")
        f.write(f"   Weighted Recall:       {weighted_r:.4f}\n")
        f.write(f"   Weighted F1-score:     {weighted_f1:.4f}\n")
        f.write(f"   Total training time:   {total_time/60:.1f} min\n\n")

        f.write("🏆 TOP 5 BEST CLASSES\n")
        for rank, idx in enumerate(best_5, 1):
            f.write(f"   {rank}. {classes[idx]:30s}  "
                    f"Acc={per_class_acc[idx]:5.1f}%  "
                    f"F1={f1[idx]:.3f}  "
                    f"Samples={label_counts.get(idx, 0)}\n")

        f.write(f"\n⚠️  TOP 5 WORST CLASSES\n")
        for rank, idx in enumerate(worst_5, 1):
            f.write(f"   {rank}. {classes[idx]:30s}  "
                    f"Acc={per_class_acc[idx]:5.1f}%  "
                    f"F1={f1[idx]:.3f}  "
                    f"Samples={label_counts.get(idx, 0)}\n")

        # Distribution of accuracy ranges
        acc_ranges = {
            "≥ 95%": sum(1 for a in per_class_acc if a >= 95),
            "80-95%": sum(1 for a in per_class_acc if 80 <= a < 95),
            "50-80%": sum(1 for a in per_class_acc if 50 <= a < 80),
            "< 50%": sum(1 for a in per_class_acc if a < 50),
        }
        f.write(f"\n📊 ACCURACY DISTRIBUTION\n")
        for range_name, count in acc_ranges.items():
            bar = "█" * count + "░" * (num_classes - count)
            f.write(f"   {range_name:8s}: {count:3d}/{num_classes} classes "
                    f"({100*count/num_classes:.0f}%)\n")

        f.write(f"\n📁 OUTPUT FILES\n")
        f.write(f"   Model weights:       {args.output_dir}/siglip2_model.pth\n")
        f.write(f"   Best checkpoint:     {args.output_dir}/best_checkpoint.pth\n")
        f.write(f"   Training log:        {args.output_dir}/training_log.csv\n")
        f.write(f"   Training curves:     {report_dir}/training_curves.png\n")
        f.write(f"   Per-class accuracy:  {report_dir}/per_class_accuracy.png\n")
        f.write(f"   Confusion matrix:    {report_dir}/confusion_matrix.png\n")
        f.write(f"   Classification rpt:  {report_dir}/classification_report.txt\n")
        f.write(f"   This summary:        {report_dir}/summary_report.txt\n")
        f.write("\n" + "=" * 70 + "\n")

    print(f"   ✅ Summary report → {summary_path}")

    # ── 7. Per-class detail CSV ───────────────────────────────────
    detail_csv_path = os.path.join(report_dir, "per_class_detail.csv")
    with open(detail_csv_path, "w", encoding="utf-8") as f:
        f.write("class_name,accuracy,precision,recall,f1,val_samples\n")
        for i in range(num_classes):
            f.write(f"{classes[i]},{per_class_acc[i]:.2f},"
                    f"{precision[i]:.4f},{recall[i]:.4f},{f1[i]:.4f},"
                    f"{label_counts.get(i, 0)}\n")
    print(f"   ✅ Per-class detail CSV → {detail_csv_path}")


if __name__ == "__main__":
    main()
