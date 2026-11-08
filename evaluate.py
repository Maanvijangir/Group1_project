"""Full evaluation for the pneumonia ResNet-18 and ResNet-50.

Reports accuracy, recall (sensitivity), specificity, precision, F1 and ROC-AUC
with a confusion matrix, so results never rest on recall alone. The threshold
can be fixed, or chosen on the validation split as the highest threshold that
keeps validation recall at --min-recall, and then applied unchanged to test.

Saves:
  - reports/evaluation_<arch>.csv   metrics per split
  - reports/figures/evaluation/<arch>_confusion_matrix.png
  - reports/figures/evaluation/<arch>_roc.png

Usage (from the repo root):
  python evaluate.py                              # ResNet-18, threshold from its training log
  python evaluate.py --arch resnet50              # ResNet-50, threshold 0.5
  python evaluate.py --arch resnet50 --min-recall 0.95
"""
import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, roc_auc_score, roc_curve

from src.dataset import get_dataloaders, get_test_transform
from src.explainability.gradcam import CLASS_NAMES, LOADERS, pneumonia_probability, saved_resnet18_threshold

# Default weights and decision threshold for each architecture
ARCH_DEFAULTS = {
    "resnet18": ("pneumonia_resnet18_recall_optimized.pth", None),  # threshold read from performance_logs_optimization.csv
    "resnet50": (os.path.join("models", "baseline_resnet50.pth"), 0.5),  # argmax in src/modeling/train.py
}


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def predict(model, loader, device):
    """Return (labels, P(pneumonia)) for every image in the loader."""
    labels, probs = [], []
    with torch.no_grad():
        for images, targets in loader:
            probs.append(pneumonia_probability(model(images.to(device))).cpu().numpy())
            labels.append(targets.numpy())
    return np.concatenate(labels), np.concatenate(probs)


def compute_metrics(labels, probs, threshold):
    preds = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    recall = tp / (tp + fn) if tp + fn else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    return {
        "images": len(labels),
        "threshold": round(float(threshold), 4),
        "accuracy": (tp + tn) / len(labels),
        "recall": recall,
        "specificity": tn / (tn + fp) if tn + fp else 0.0,
        "precision": precision,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "roc_auc": roc_auc_score(labels, probs) if len(set(labels)) == 2 else float("nan"),
        "tp": int(tp), "fn": int(fn), "tn": int(tn), "fp": int(fp),
    }


def pick_threshold(labels, probs, min_recall):
    """Highest threshold whose recall is still at least min_recall."""
    candidates = np.unique(probs)[::-1]
    for threshold in candidates:
        if ((probs >= threshold) & (labels == 1)).sum() / (labels == 1).sum() >= min_recall:
            return float(threshold)
    return float(candidates[-1])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arch", default="resnet18", choices=list(LOADERS))
    parser.add_argument("--weights", help="default: the usual weights file for --arch")
    parser.add_argument("--threshold", type=float,
                        help="fixed P(pneumonia) threshold (default: from the training log for resnet18, 0.5 for resnet50)")
    parser.add_argument("--min-recall", type=float,
                        help="instead of --threshold, pick the threshold on validation that keeps this recall")
    parser.add_argument("--image-size", type=int,
                        help="input size the model was trained at (default: src/dataset.py test_transform; "
                             "the ResNet-18 weights in the repo were trained at 150)")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    default_weights, default_threshold = ARCH_DEFAULTS[args.arch]
    if default_threshold is None:
        default_threshold = saved_resnet18_threshold()
    weights = args.weights or default_weights
    device = get_device()
    model = LOADERS[args.arch](weights, device)

    # The same validation split and preprocessing as training
    _, val_loader, test_loader = get_dataloaders(batch_size=args.batch_size)
    transform = get_test_transform(args.image_size)
    val_loader.dataset.transform = transform
    test_loader.dataset.transform = transform
    loaders = {"val": val_loader, "test": test_loader}
    results = {split: predict(model, loader, device) for split, loader in loaders.items()}

    if args.min_recall is not None:
        threshold = pick_threshold(*results["val"], args.min_recall)
        print(f"Threshold chosen on validation for recall >= {args.min_recall}: {threshold:.4f}")
    else:
        threshold = default_threshold if args.threshold is None else args.threshold

    rows = []
    print(f"\n{args.arch} ({weights}), threshold {threshold:.4f}, image size {args.image_size or 'from src/dataset.py'}")
    print(f"{'split':<6}{'acc':>8}{'recall':>8}{'spec':>8}{'prec':>8}{'f1':>8}{'auc':>8}   TP  FN  TN  FP")
    for split, (labels, probs) in results.items():
        metrics = compute_metrics(labels, probs, threshold)
        rows.append({"split": split, **{k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()}})
        print(f"{split:<6}{metrics['accuracy']:>8.3f}{metrics['recall']:>8.3f}{metrics['specificity']:>8.3f}"
              f"{metrics['precision']:>8.3f}{metrics['f1']:>8.3f}{metrics['roc_auc']:>8.3f}"
              f"  {metrics['tp']:>4}{metrics['fn']:>4}{metrics['tn']:>4}{metrics['fp']:>4}")

    os.makedirs("reports", exist_ok=True)
    csv_path = os.path.join("reports", f"evaluation_{args.arch}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    fig_dir = os.path.join("reports", "figures", "evaluation")
    os.makedirs(fig_dir, exist_ok=True)
    labels, probs = results["test"]
    ConfusionMatrixDisplay(confusion_matrix(labels, (probs >= threshold).astype(int), labels=[0, 1]),
                           display_labels=CLASS_NAMES).plot(cmap="Blues", colorbar=False)
    plt.title(f"{args.arch} on test (threshold {threshold:.2f})")
    plt.savefig(os.path.join(fig_dir, f"{args.arch}_confusion_matrix.png"), dpi=120, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(5, 5))
    for split, (labels, probs) in results.items():
        fpr, tpr, _ = roc_curve(labels, probs)
        plt.plot(fpr, tpr, label=f"{split} (AUC {roc_auc_score(labels, probs):.3f})")
    plt.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    plt.xlabel("False positive rate (1 - specificity)")
    plt.ylabel("True positive rate (recall)")
    plt.title(f"{args.arch} ROC")
    plt.legend()
    plt.savefig(os.path.join(fig_dir, f"{args.arch}_roc.png"), dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\nSaved {csv_path} and figures to {fig_dir}")


if __name__ == "__main__":
    main()
