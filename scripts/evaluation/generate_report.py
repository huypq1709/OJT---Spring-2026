# -*- coding: utf-8 -*-
"""
Generate training report from existing train_output/ without retraining.

Mode 1 (CSV only — no GPU needed):
    python generate_report.py --output-dir train_output

Mode 2 (Full report — needs GPU + dataset):
    python generate_report.py --output-dir train_output --data-dir cropped_classes --full

Outputs saved to: train_output/report/
"""

import argparse
import os
import json
import numpy as np
from pathlib import Path
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def parse_args():
    parser = argparse.ArgumentParser(description="Generate training report")
    parser.add_argument("--output-dir", default="train_output",
                        help="Directory containing training outputs")
    parser.add_argument("--data-dir", default="cropped_classes",
                        help="Dataset root (only needed with --full)")
    parser.add_argument("--full", action="store_true",
                        help="Full report with confusion matrix (needs GPU + dataset)")
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=384,
                        help="Batch size for evaluation")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--gpu", type=int, default=0)
    return parser.parse_args()


# ═══════════════════════════════════════════════════════════════════
# 1. Parse CSV
# ═══════════════════════════════════════════════════════════════════
def parse_training_log(csv_path):
    """Parse training_log.csv into structured data."""
    epochs_data = []
    with open(csv_path, "r") as f:
        header = f.readline()
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
                    "backbone": parts[6].strip() if len(parts) > 6 else "frozen",
                })
    return epochs_data


# ═══════════════════════════════════════════════════════════════════
# 2. Training Curves
# ═══════════════════════════════════════════════════════════════════
def plot_training_curves(epochs_data, report_dir):
    """Generate 4-panel training curves chart."""
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "figure.dpi": 150,
    })

    ep = [d["epoch"] for d in epochs_data]
    train_losses = [d["train_loss"] for d in epochs_data]
    val_losses = [d["val_loss"] for d in epochs_data]
    train_accs = [d["train_acc"] for d in epochs_data]
    val_accs = [d["val_acc"] for d in epochs_data]
    lrs = [d["lr"] for d in epochs_data]

    # Find unfreeze epoch
    unfreeze_epoch = None
    for d in epochs_data:
        if d["backbone"] == "unfrozen":
            unfreeze_epoch = d["epoch"]
            break

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle("SigLIP2 Training Report", fontsize=18, fontweight="bold", y=0.98)

    # ── Loss curves ──
    ax1 = axes[0, 0]
    ax1.plot(ep, train_losses, "o-", color="#2196F3", linewidth=2,
             markersize=5, label="Train Loss", zorder=3)
    ax1.plot(ep, val_losses, "s-", color="#F44336", linewidth=2,
             markersize=5, label="Val Loss", zorder=3)
    if unfreeze_epoch:
        ax1.axvline(x=unfreeze_epoch, color="#9C27B0", linestyle="--",
                    alpha=0.7, linewidth=1.5, label=f"Unfreeze (epoch {unfreeze_epoch})")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("📉 Loss Curves")
    ax1.legend(loc="upper right", framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    # ── Accuracy curves ──
    ax2 = axes[0, 1]
    ax2.plot(ep, train_accs, "o-", color="#4CAF50", linewidth=2,
             markersize=5, label="Train Acc", zorder=3)
    ax2.plot(ep, val_accs, "s-", color="#FF9800", linewidth=2,
             markersize=5, label="Val Acc", zorder=3)
    if unfreeze_epoch:
        ax2.axvline(x=unfreeze_epoch, color="#9C27B0", linestyle="--",
                    alpha=0.7, linewidth=1.5, label=f"Unfreeze (epoch {unfreeze_epoch})")
    # Shade frozen vs unfrozen
    if unfreeze_epoch:
        ax2.axvspan(ep[0], unfreeze_epoch, alpha=0.05, color="blue", label="Frozen phase")
        ax2.axvspan(unfreeze_epoch, ep[-1], alpha=0.05, color="green", label="Unfrozen phase")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("📈 Accuracy Curves")
    ax2.legend(loc="lower right", framealpha=0.9, fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    # ── Learning Rate schedule ──
    ax3 = axes[1, 0]
    ax3.plot(ep, lrs, "D-", color="#673AB7", linewidth=2, markersize=5)
    if unfreeze_epoch:
        ax3.axvline(x=unfreeze_epoch, color="#9C27B0", linestyle="--",
                    alpha=0.7, linewidth=1.5, label=f"Unfreeze (epoch {unfreeze_epoch})")
        ax3.legend(framealpha=0.9)
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Learning Rate")
    ax3.set_title("⚡ Learning Rate Schedule")
    ax3.set_yscale("log")
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    # ── Overfitting gap ──
    ax4 = axes[1, 1]
    gap = [t - v for t, v in zip(train_accs, val_accs)]
    colors = ["#4CAF50" if g < 2 else "#FF9800" if g < 5 else "#F44336" for g in gap]
    ax4.bar(ep, gap, color=colors, alpha=0.85, edgecolor="white", linewidth=0.5)
    ax4.axhline(y=0, color="black", linewidth=0.5)
    ax4.axhline(y=2, color="#4CAF50", linewidth=0.8, linestyle=":", alpha=0.5, label="Good (<2%)")
    ax4.axhline(y=5, color="#FF9800", linewidth=0.8, linestyle=":", alpha=0.5, label="Warning (<5%)")
    ax4.set_xlabel("Epoch")
    ax4.set_ylabel("Train - Val Acc (%)")
    ax4.set_title("🔍 Overfitting Gap")
    ax4.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax4.grid(True, alpha=0.3, axis="y")
    ax4.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(report_dir, "training_curves.png")
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"   ✅ Training curves → {path}")
    return path


# ═══════════════════════════════════════════════════════════════════
# 3. Epoch Detail Chart
# ═══════════════════════════════════════════════════════════════════
def plot_epoch_detail(epochs_data, report_dir):
    """Detailed epoch-by-epoch comparison table as image."""
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis("off")

    headers = ["Epoch", "Train Loss", "Train Acc", "Val Loss", "Val Acc", "LR", "Backbone"]
    cell_data = []
    for d in epochs_data:
        cell_data.append([
            str(d["epoch"]),
            f"{d['train_loss']:.4f}",
            f"{d['train_acc']:.2f}%",
            f"{d['val_loss']:.4f}",
            f"{d['val_acc']:.2f}%",
            f"{d['lr']:.2e}",
            d["backbone"],
        ])

    table = ax.table(
        cellText=cell_data,
        colLabels=headers,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.4)

    # Style header
    for j, header in enumerate(headers):
        cell = table[0, j]
        cell.set_facecolor("#2196F3")
        cell.set_text_props(color="white", fontweight="bold")

    # Color-code rows
    for i, d in enumerate(cell_data, start=1):
        backbone = d[-1]
        bg_color = "#E3F2FD" if backbone == "frozen" else "#E8F5E9"
        for j in range(len(headers)):
            table[i, j].set_facecolor(bg_color)

    # Highlight best val acc row
    best_idx = max(range(len(epochs_data)), key=lambda i: epochs_data[i]["val_acc"])
    for j in range(len(headers)):
        table[best_idx + 1, j].set_facecolor("#FFF9C4")
        table[best_idx + 1, j].set_text_props(fontweight="bold")

    plt.title("Epoch-by-Epoch Details (yellow = best val acc)", fontsize=14,
              fontweight="bold", pad=20)
    plt.tight_layout()
    path = os.path.join(report_dir, "epoch_details.png")
    fig.savefig(path, bbox_inches="tight", facecolor="white", dpi=150)
    plt.close(fig)
    print(f"   ✅ Epoch details → {path}")
    return path


# ═══════════════════════════════════════════════════════════════════
# 4. Summary Text Report
# ═══════════════════════════════════════════════════════════════════
def write_summary_report(epochs_data, report_dir, classes=None,
                         per_class_stats=None):
    """Write comprehensive text summary."""
    num_classes = len(classes) if classes else "?"
    best_epoch = max(epochs_data, key=lambda d: d["val_acc"])
    last_epoch = epochs_data[-1]
    first_epoch = epochs_data[0]

    # Find unfreeze epoch
    unfreeze_epoch = None
    for d in epochs_data:
        if d["backbone"] == "unfrozen":
            unfreeze_epoch = d["epoch"]
            break

    path = os.path.join(report_dir, "summary_report.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("  SigLIP2 TRAINING SUMMARY REPORT\n")
        f.write("=" * 70 + "\n\n")

        f.write("📋 MODEL INFO\n")
        f.write(f"   Architecture:   ViT-B-16-SigLIP2-256 + Classification Head\n")
        f.write(f"   Num classes:    {num_classes}\n")
        f.write(f"   Epochs trained: {len(epochs_data)}\n\n")

        f.write("📈 TRAINING RESULTS\n")
        f.write(f"   ┌───────────────────────────────────────────────┐\n")
        f.write(f"   │  Best Val Accuracy:  {best_epoch['val_acc']:.2f}%  (epoch {best_epoch['epoch']})  │\n")
        f.write(f"   │  Best Val Loss:      {best_epoch['val_loss']:.4f}              │\n")
        f.write(f"   └───────────────────────────────────────────────┘\n\n")

        f.write(f"   Final epoch ({last_epoch['epoch']}):\n")
        f.write(f"     Train: loss={last_epoch['train_loss']:.4f}  acc={last_epoch['train_acc']:.2f}%\n")
        f.write(f"     Val:   loss={last_epoch['val_loss']:.4f}  acc={last_epoch['val_acc']:.2f}%\n")
        f.write(f"     Gap:   {last_epoch['train_acc'] - last_epoch['val_acc']:.2f}%\n\n")

        f.write("📊 TRAINING PHASES\n")
        frozen_epochs = [d for d in epochs_data if d["backbone"] == "frozen"]
        unfrozen_epochs = [d for d in epochs_data if d["backbone"] == "unfrozen"]

        if frozen_epochs:
            f.write(f"   🔒 Frozen phase (epochs {frozen_epochs[0]['epoch']}-{frozen_epochs[-1]['epoch']}):\n")
            f.write(f"      Start: Train={frozen_epochs[0]['train_acc']:.2f}%  Val={frozen_epochs[0]['val_acc']:.2f}%\n")
            f.write(f"      End:   Train={frozen_epochs[-1]['train_acc']:.2f}%  Val={frozen_epochs[-1]['val_acc']:.2f}%\n")
            f.write(f"      Improvement: +{frozen_epochs[-1]['val_acc'] - frozen_epochs[0]['val_acc']:.2f}% val acc\n\n")

        if unfrozen_epochs:
            f.write(f"   🔓 Unfrozen phase (epochs {unfrozen_epochs[0]['epoch']}-{unfrozen_epochs[-1]['epoch']}):\n")
            f.write(f"      Start: Train={unfrozen_epochs[0]['train_acc']:.2f}%  Val={unfrozen_epochs[0]['val_acc']:.2f}%\n")
            f.write(f"      End:   Train={unfrozen_epochs[-1]['train_acc']:.2f}%  Val={unfrozen_epochs[-1]['val_acc']:.2f}%\n")
            f.write(f"      Improvement: +{unfrozen_epochs[-1]['val_acc'] - unfrozen_epochs[0]['val_acc']:.2f}% val acc\n\n")

        f.write("📉 CONVERGENCE ANALYSIS\n")
        val_accs = [d["val_acc"] for d in epochs_data]
        # Check last 5 epochs variance
        if len(val_accs) >= 5:
            last5_std = np.std(val_accs[-5:])
            f.write(f"   Val acc std (last 5 epochs): {last5_std:.4f}%\n")
            if last5_std < 0.1:
                f.write(f"   → Model has CONVERGED (very stable)\n")
            elif last5_std < 0.5:
                f.write(f"   → Model is NEARLY converged\n")
            else:
                f.write(f"   → Model may benefit from more training\n")

        # Overfitting check
        gap = last_epoch["train_acc"] - last_epoch["val_acc"]
        f.write(f"\n   Train-Val gap (final): {gap:.2f}%\n")
        if gap < 1:
            f.write(f"   → ✅ No overfitting (gap < 1%)\n")
        elif gap < 5:
            f.write(f"   → ⚠️ Minor overfitting (gap < 5%)\n")
        else:
            f.write(f"   → ❌ Significant overfitting (gap ≥ 5%)\n")

        # Per-class stats if available
        if per_class_stats:
            f.write(f"\n\n🏆 TOP 5 BEST CLASSES\n")
            sorted_by_acc = sorted(per_class_stats, key=lambda x: -x["accuracy"])
            for rank, s in enumerate(sorted_by_acc[:5], 1):
                f.write(f"   {rank}. {s['name']:30s}  Acc={s['accuracy']:5.1f}%  "
                        f"F1={s['f1']:.3f}  Samples={s['samples']}\n")

            f.write(f"\n⚠️  TOP 5 WORST CLASSES\n")
            for rank, s in enumerate(sorted_by_acc[-5:][::-1], 1):
                f.write(f"   {rank}. {s['name']:30s}  Acc={s['accuracy']:5.1f}%  "
                        f"F1={s['f1']:.3f}  Samples={s['samples']}\n")

            # Distribution
            accs = [s["accuracy"] for s in per_class_stats]
            f.write(f"\n📊 ACCURACY DISTRIBUTION\n")
            ranges = [
                ("≥ 95%", lambda a: a >= 95),
                ("80-95%", lambda a: 80 <= a < 95),
                ("50-80%", lambda a: 50 <= a < 80),
                ("< 50%", lambda a: a < 50),
            ]
            nc = len(per_class_stats)
            for name, fn in ranges:
                count = sum(1 for a in accs if fn(a))
                bar = "█" * int(count / nc * 40) + "░" * (40 - int(count / nc * 40))
                f.write(f"   {name:8s}: {count:3d}/{nc} ({100*count/nc:4.0f}%) {bar}\n")

        f.write("\n\n📁 OUTPUT FILES\n")
        f.write(f"   training_curves.png    — Loss, accuracy, LR, overfitting charts\n")
        f.write(f"   epoch_details.png      — Epoch-by-epoch table\n")
        f.write(f"   summary_report.txt     — This file\n")
        if per_class_stats:
            f.write(f"   per_class_accuracy.png — Per-class accuracy bar chart\n")
            f.write(f"   confusion_matrix.png   — Confusion matrix heatmap\n")
            f.write(f"   classification_report.txt — Full sklearn report\n")
            f.write(f"   per_class_detail.csv   — CSV with all per-class metrics\n")

        f.write("\n" + "=" * 70 + "\n")

    print(f"   ✅ Summary report → {path}")
    return path


# ═══════════════════════════════════════════════════════════════════
# 5. Full Evaluation (needs model + dataset)
# ═══════════════════════════════════════════════════════════════════
def run_full_evaluation(args, classes, report_dir):
    """Load best model, run on val set, generate confusion matrix + per-class stats."""
    import torch
    from torch.utils.data import DataLoader, Subset
    from sklearn.metrics import (
        confusion_matrix, classification_report,
        precision_recall_fscore_support,
    )

    # Import dataset and model from training script
    from train_siglip2_server import LogoCropDataset, SigLIP2Classifier

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n   GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    # Load dataset (no augmentation for eval)
    print(f"   Loading dataset from: {args.data_dir}")
    dataset = LogoCropDataset(
        args.data_dir, classes=classes,
        img_size=args.img_size, augment=False
    )
    total_samples = len(dataset)

    # Same split as training
    val_size = int(total_samples * 0.05)
    train_size = total_samples - val_size
    generator = torch.Generator().manual_seed(42)
    indices = torch.randperm(total_samples, generator=generator).tolist()
    val_indices = indices[train_size:]

    val_dataset = Subset(dataset, val_indices)
    print(f"   Val samples: {len(val_dataset):,}")

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Load model
    num_classes = len(classes)
    model = SigLIP2Classifier(num_classes=num_classes)
    weights_path = os.path.join(args.output_dir, "siglip2_model.pth")
    state = torch.load(weights_path, map_location="cpu", weights_only=False)
    cleaned = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(cleaned)
    model = model.to(device)
    model.eval()
    print(f"   Loaded model from: {weights_path}")

    # Inference
    all_preds = []
    all_labels = []

    from tqdm import tqdm
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="   Evaluating"):
            images = images.to(device, non_blocking=True)
            with torch.amp.autocast("cuda"):
                outputs = model(images)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().tolist())
            all_labels.extend(labels.tolist())

    overall_acc = 100.0 * sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    print(f"   Val Accuracy: {overall_acc:.2f}%")

    # Per-class stats
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, labels=list(range(num_classes)),
        average=None, zero_division=0
    )
    label_counts = Counter(all_labels)

    per_class_stats = []
    for i in range(num_classes):
        class_total = sum(1 for l in all_labels if l == i)
        if class_total > 0:
            class_correct = sum(1 for l, p in zip(all_labels, all_preds) if l == i and p == i)
            acc = 100.0 * class_correct / class_total
        else:
            acc = 0.0
        per_class_stats.append({
            "name": classes[i],
            "accuracy": acc,
            "precision": precision[i],
            "recall": recall[i],
            "f1": f1[i],
            "samples": label_counts.get(i, 0),
        })

    # ── Per-class accuracy bar chart ──
    sorted_stats = sorted(per_class_stats, key=lambda x: x["accuracy"])
    if num_classes > 40:
        show_stats = sorted_stats[:20] + sorted_stats[-20:]
        chart_title = "Per-class Accuracy (Bottom 20 + Top 20)"
    else:
        show_stats = sorted_stats
        chart_title = "Per-class Accuracy"

    fig, ax = plt.subplots(figsize=(14, max(6, len(show_stats) * 0.28)))
    names = [s["name"] for s in show_stats]
    accs = [s["accuracy"] for s in show_stats]
    bar_colors = ["#F44336" if a < 50 else "#FF9800" if a < 80 else "#4CAF50" for a in accs]

    bars = ax.barh(range(len(show_stats)), accs, color=bar_colors, height=0.7,
                   edgecolor="white", linewidth=0.3)
    ax.set_yticks(range(len(show_stats)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Accuracy (%)")
    ax.set_title(chart_title, fontweight="bold", fontsize=14)
    ax.set_xlim(0, 105)
    ax.axvline(x=50, color="red", linestyle=":", alpha=0.3)
    ax.axvline(x=80, color="orange", linestyle=":", alpha=0.3)
    ax.axvline(x=95, color="green", linestyle=":", alpha=0.3)
    ax.grid(True, alpha=0.2, axis="x")

    for bar, acc in zip(bars, accs):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f"{acc:.1f}%", va="center", fontsize=7)

    plt.tight_layout()
    path = os.path.join(report_dir, "per_class_accuracy.png")
    fig.savefig(path, bbox_inches="tight", facecolor="white", dpi=150)
    plt.close(fig)
    print(f"   ✅ Per-class accuracy → {path}")

    # ── Confusion matrix ──
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))

    if num_classes > 30:
        confused_pairs = []
        for i in range(num_classes):
            for j in range(num_classes):
                if i != j and cm[i][j] > 0:
                    confused_pairs.append((i, j, cm[i][j]))
        confused_pairs.sort(key=lambda x: -x[2])
        top_confused = confused_pairs[:30]

        confused_classes = sorted(set(
            [p[0] for p in top_confused] + [p[1] for p in top_confused]
        ))

        if len(confused_classes) > 1:
            sub_cm = cm[np.ix_(confused_classes, confused_classes)]
            sub_names = [classes[i][:18] for i in confused_classes]

            fig, ax = plt.subplots(
                figsize=(max(10, len(confused_classes)*0.55),
                         max(8, len(confused_classes)*0.5))
            )
            im = ax.imshow(sub_cm, cmap="YlOrRd", aspect="auto")
            ax.set_xticks(range(len(confused_classes)))
            ax.set_yticks(range(len(confused_classes)))
            ax.set_xticklabels(sub_names, rotation=45, ha="right", fontsize=7)
            ax.set_yticklabels(sub_names, fontsize=7)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            ax.set_title("Confusion Matrix (Most Confused Classes)", fontweight="bold")
            plt.colorbar(im, ax=ax, shrink=0.8)

            for ii in range(len(confused_classes)):
                for jj in range(len(confused_classes)):
                    val = sub_cm[ii, jj]
                    if val > 0:
                        ax.text(jj, ii, str(val), ha="center", va="center",
                                fontsize=6,
                                color="black" if val < sub_cm.max()/2 else "white")

            plt.tight_layout()
            cm_path = os.path.join(report_dir, "confusion_matrix.png")
            fig.savefig(cm_path, bbox_inches="tight", facecolor="white", dpi=150)
            plt.close(fig)
            print(f"   ✅ Confusion matrix → {cm_path}")
    else:
        fig, ax = plt.subplots(figsize=(max(10, num_classes*0.4),
                                        max(8, num_classes*0.35)))
        im = ax.imshow(cm, cmap="YlOrRd", aspect="auto")
        ax.set_xticks(range(num_classes))
        ax.set_yticks(range(num_classes))
        ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(classes, fontsize=7)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Confusion Matrix", fontweight="bold")
        plt.colorbar(im, ax=ax, shrink=0.8)
        plt.tight_layout()
        cm_path = os.path.join(report_dir, "confusion_matrix.png")
        fig.savefig(cm_path, bbox_inches="tight", facecolor="white", dpi=150)
        plt.close(fig)
        print(f"   ✅ Confusion matrix → {cm_path}")

    # ── Classification report ──
    cls_report = classification_report(
        all_labels, all_preds,
        labels=list(range(num_classes)),
        target_names=classes,
        digits=3,
        zero_division=0,
    )
    cls_path = os.path.join(report_dir, "classification_report.txt")
    with open(cls_path, "w", encoding="utf-8") as f:
        f.write(cls_report)
    print(f"   ✅ Classification report → {cls_path}")

    # ── Per-class CSV ──
    csv_path = os.path.join(report_dir, "per_class_detail.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("class_name,accuracy,precision,recall,f1,val_samples\n")
        for s in per_class_stats:
            f.write(f"{s['name']},{s['accuracy']:.2f},"
                    f"{s['precision']:.4f},{s['recall']:.4f},{s['f1']:.4f},"
                    f"{s['samples']}\n")
    print(f"   ✅ Per-class detail CSV → {csv_path}")

    return per_class_stats


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════
def main():
    args = parse_args()

    print(f"{'='*60}")
    print(f"  📊 SigLIP2 Report Generator")
    print(f"{'='*60}")

    # Validate inputs
    log_path = os.path.join(args.output_dir, "training_log.csv")
    classes_path = os.path.join(args.output_dir, "classes.json")

    if not os.path.isfile(log_path):
        print(f"  ❌ training_log.csv not found in {args.output_dir}")
        return

    # Parse CSV
    epochs_data = parse_training_log(log_path)
    print(f"\n  📂 Found {len(epochs_data)} epochs in training log")

    # Load classes
    classes = None
    if os.path.isfile(classes_path):
        with open(classes_path, "r") as f:
            classes = json.load(f)
        print(f"  📂 Found {len(classes)} classes")

    # Create report directory
    report_dir = os.path.join(args.output_dir, "report")
    os.makedirs(report_dir, exist_ok=True)

    # ── Generate charts from CSV ──
    print(f"\n  🎨 Generating charts...")
    plot_training_curves(epochs_data, report_dir)
    plot_epoch_detail(epochs_data, report_dir)

    # ── Full evaluation if requested ──
    per_class_stats = None
    if args.full:
        if not os.path.isfile(os.path.join(args.output_dir, "siglip2_model.pth")):
            print(f"\n  ❌ siglip2_model.pth not found, cannot run full evaluation")
        elif not os.path.isdir(args.data_dir):
            print(f"\n  ❌ Dataset not found: {args.data_dir}")
        else:
            print(f"\n  🔬 Running full evaluation...")
            per_class_stats = run_full_evaluation(args, classes, report_dir)

    # ── Summary report ──
    print(f"\n  📝 Writing summary report...")
    write_summary_report(epochs_data, report_dir, classes, per_class_stats)

    print(f"\n{'='*60}")
    print(f"  ✅ Report generated → {report_dir}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
