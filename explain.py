"""Explainability for the pneumonia ResNet-18 and ResNet-50.

Runs Grad-CAM on every image of a split, groups the results into true/false
positives/negatives, and saves:
  - reports/figures/explainability/<run>/<group>.png   Grad-CAM grids per outcome group
  - reports/figures/explainability/<run>/occlusion_<group>.png   Grad-CAM vs. occlusion check
  - reports/explainability_summary_<run>.csv   per-image probability, prediction and border attention
<run> is the model name plus any --image-size, so one model's report never overwrites another's.

Usage (from the repo root):
  python explain.py                         # ResNet-18, whole test split
  python explain.py --arch resnet50         # ResNet-50 from models/baseline_resnet50.pth
  python explain.py --image path/to/xray.jpeg
  python explain.py --image-size 150        # model trained at another size than src/dataset.py uses
"""
import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets

from src.dataset import DATA_DIR, get_test_transform
from src.explainability.gradcam import (CLASS_NAMES, LOADERS, GradCAM, border_fraction, denormalize,
                                        occlusion_sensitivity)

# Default weights and decision threshold for each architecture
ARCH_DEFAULTS = {
    "resnet18": ("pneumonia_resnet18_recall_optimized.pth", 0.4),  # the threshold the weights in the repo were reported at
    "resnet50": (os.path.join("models", "baseline_resnet50.pth"), 0.5),  # argmax in src/modeling/train.py
}

GROUPS = {
    "true_positive": "Pneumonia, predicted pneumonia",
    "true_negative": "Normal, predicted normal",
    "false_positive": "Normal, predicted pneumonia",
    "false_negative": "Pneumonia, predicted normal",
}


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def outcome(label, pred):
    if label == 1:
        return "true_positive" if pred == 1 else "false_negative"
    return "false_positive" if pred == 1 else "true_negative"


def plot_overlay(ax, image, cam, title):
    ax.imshow(image, cmap="gray")
    ax.imshow(cam, cmap="jet", alpha=0.4)
    ax.set_title(title, fontsize=8)
    ax.axis("off")


def save_group_grid(records, group, path):
    """One row per image: original X-ray and Grad-CAM overlay."""
    rows = len(records)
    fig, axes = plt.subplots(rows, 2, figsize=(6, 3 * rows), squeeze=False)
    for row, rec in enumerate(records):
        axes[row, 0].imshow(rec["image"], cmap="gray")
        axes[row, 0].set_title(rec["filename"], fontsize=8)
        axes[row, 0].axis("off")
        plot_overlay(axes[row, 1], rec["image"], rec["cam"],
                     f"P(pneumonia)={rec['prob']:.2f}  border={rec['border']:.2f}")
    fig.suptitle(GROUPS[group], fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_occlusion_grid(model, records, tensors, group, path):
    """Grad-CAM next to occlusion sensitivity, to check both methods agree."""
    rows = len(records)
    fig, axes = plt.subplots(rows, 3, figsize=(9, 3 * rows), squeeze=False)
    for row, (rec, tensor) in enumerate(zip(records, tensors)):
        occ = occlusion_sensitivity(model, tensor)
        axes[row, 0].imshow(rec["image"], cmap="gray")
        axes[row, 0].set_title(rec["filename"], fontsize=8)
        plot_overlay(axes[row, 1], rec["image"], rec["cam"], "Grad-CAM")
        limit = max(abs(occ.min()), abs(occ.max()), 1e-6)
        axes[row, 2].imshow(rec["image"], cmap="gray")
        axes[row, 2].imshow(occ, cmap="bwr", alpha=0.5, vmin=-limit, vmax=limit)
        axes[row, 2].set_title("Occlusion (red = supports pneumonia)", fontsize=8)
        for ax in axes[row]:
            ax.axis("off")
    fig.suptitle(f"Grad-CAM vs. occlusion: {GROUPS[group]}", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def explain_split(args, model, device):
    split_dir = os.path.join(args.data_dir, args.split)
    data = datasets.ImageFolder(split_dir, transform=get_test_transform(args.image_size))
    loader = DataLoader(data, batch_size=32, shuffle=False)
    gradcam = GradCAM(model, model.layer4)

    rows, examples = [], {group: [] for group in GROUPS}
    index = 0
    for images, labels in loader:
        images = images.to(device)
        cams, probs, preds = gradcam(images, threshold=args.threshold)
        for i in range(len(labels)):
            path = data.samples[index][0]
            label, pred = int(labels[i]), int(preds[i])
            group = outcome(label, pred)
            row = {
                "filename": os.path.basename(path),
                "label": CLASS_NAMES[label],
                "prob_pneumonia": round(float(probs[i]), 4),
                "prediction": CLASS_NAMES[pred],
                "outcome": group,
                "cam_border_fraction": round(border_fraction(cams[i]), 4),
            }
            rows.append(row)
            if len(examples[group]) < args.num_examples:
                examples[group].append((index, {
                    "filename": row["filename"], "image": denormalize(images[i]),
                    "cam": cams[i], "prob": row["prob_pneumonia"],
                    "border": row["cam_border_fraction"],
                }))
            index += 1
    gradcam.remove()

    os.makedirs(args.out_dir, exist_ok=True)
    summary_path = args.summary
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    for group, items in examples.items():
        if not items:
            continue
        records = [rec for _, rec in items]
        save_group_grid(records, group, os.path.join(args.out_dir, f"{group}.png"))
        tensors = [data[i][0].to(device) for i, _ in items[:args.num_occlusion]]
        save_occlusion_grid(model, records[:args.num_occlusion], tensors, group,
                            os.path.join(args.out_dir, f"occlusion_{group}.png"))

    print(f"Explained {len(rows)} images from {split_dir}")
    print(f"{'outcome':<16}{'count':>7}{'mean border attention':>24}")
    for group in GROUPS:
        values = [r["cam_border_fraction"] for r in rows if r["outcome"] == group]
        mean = f"{np.mean(values):.3f}" if values else "-"
        print(f"{group:<16}{len(values):>7}{mean:>24}")
    print(f"Saved summary to {summary_path} and figures to {args.out_dir}")


def explain_image(args, model, device):
    image = Image.open(args.image).convert("RGB")
    tensor = get_test_transform(args.image_size)(image).unsqueeze(0).to(device)
    gradcam = GradCAM(model, model.layer4)
    cams, probs, preds = gradcam(tensor, threshold=args.threshold)
    gradcam.remove()
    occ = occlusion_sensitivity(model, tensor[0])

    gray = denormalize(tensor[0])
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.2))
    axes[0].imshow(gray, cmap="gray")
    axes[0].set_title(os.path.basename(args.image), fontsize=8)
    plot_overlay(axes[1], gray, cams[0], "Grad-CAM")
    limit = max(abs(occ.min()), abs(occ.max()), 1e-6)
    axes[2].imshow(gray, cmap="gray")
    axes[2].imshow(occ, cmap="bwr", alpha=0.5, vmin=-limit, vmax=limit)
    axes[2].set_title("Occlusion (red = supports pneumonia)", fontsize=8)
    for ax in axes:
        ax.axis("off")
    fig.suptitle(f"Prediction: {CLASS_NAMES[preds[0]]}  (P(pneumonia)={probs[0]:.2f})", fontsize=11)
    fig.tight_layout()

    os.makedirs(args.out_dir, exist_ok=True)
    name = os.path.splitext(os.path.basename(args.image))[0]
    out_path = os.path.join(args.out_dir, f"single_{name}.png")
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Prediction: {CLASS_NAMES[preds[0]]}, P(pneumonia)={probs[0]:.4f}, "
          f"border attention={border_fraction(cams[0]):.3f}")
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arch", default="resnet18", choices=list(LOADERS))
    parser.add_argument("--weights", help="default: the usual weights file for --arch")
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--image", help="explain a single X-ray instead of a whole split")
    parser.add_argument("--threshold", type=float,
                        help="P(pneumonia) decision threshold (default: 0.4 for resnet18, 0.5 for resnet50)")
    parser.add_argument("--image-size", type=int,
                        help="input size the model was trained at (default: src/dataset.py test_transform; "
                             "the ResNet-18 weights were trained at 150)")
    parser.add_argument("--num-examples", type=int, default=4, help="images per outcome group in the grids")
    parser.add_argument("--num-occlusion", type=int, default=2, help="images per group for the occlusion check")
    parser.add_argument("--out-dir", help="default: reports/figures/explainability/<run>")
    parser.add_argument("--summary", help="default: reports/explainability_summary_<run>.csv")
    args = parser.parse_args()

    run = f"{args.arch}_{args.image_size}" if args.image_size else args.arch
    args.out_dir = args.out_dir or os.path.join("reports", "figures", "explainability", run)
    args.summary = args.summary or os.path.join("reports", f"explainability_summary_{run}.csv")

    device = get_device()
    default_weights, default_threshold = ARCH_DEFAULTS[args.arch]
    args.weights = args.weights or default_weights
    args.threshold = default_threshold if args.threshold is None else args.threshold
    model = LOADERS[args.arch](args.weights, device)
    if args.image:
        explain_image(args, model, device)
    else:
        explain_split(args, model, device)


if __name__ == "__main__":
    main()
