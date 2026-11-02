import os
import csv
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score

from src.dataset import get_dataloaders

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

MODEL_SAVE_PATH = "pneumonia_resnet18_recall_optimized.pth"
LOG_CSV_PATH = "performance_logs_optimization.csv"


def build_resnet18(device):
    """Builds ResNet-18 maintaining original state_dict key names (model.fc)."""
    try:
        weights = models.ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
    except AttributeError:
        model = models.resnet18(pretrained=True)

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 1)
    return model.to(device)


def evaluate(model, loader, criterion, device, threshold=0.5):
    """Evaluates loss, Recall, Specificity, Precision, F1, and ROC-AUC."""
    model.eval()
    running_loss = 0.0
    all_probs = []
    all_targets = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels_f = labels.float().unsqueeze(1).to(device)

            logits = model(images)
            loss = criterion(logits, labels_f)
            running_loss += loss.item() * images.size(0)

            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            all_probs.extend(probs)
            all_targets.extend(labels.numpy().flatten())

    avg_loss = running_loss / len(loader.dataset)
    all_probs = np.array(all_probs)
    all_targets = np.array(all_targets)

    preds = (all_probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(all_targets, preds, labels=[0, 1]).ravel()

    acc = (tp + tn) / (tp + tn + fp + fn)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = f1_score(all_targets, preds, zero_division=0)
    auc = roc_auc_score(all_targets, all_probs)

    return {
        "loss": avg_loss,
        "acc": acc,
        "recall": recall,
        "specificity": specificity,
        "precision": precision,
        "f1": f1,
        "auc": auc,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "probs": all_probs,
        "targets": all_targets
    }


def find_optimal_threshold(probs, targets, min_recall=0.92):
    """Finds decision threshold achieving >= min_recall while maximizing Specificity."""
    best_thresh = 0.50
    best_spec = -1.0

    for t in np.arange(0.30, 0.85, 0.02):
        preds = (probs >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(targets, preds, labels=[0, 1]).ravel()
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        if rec >= min_recall and spec > best_spec:
            best_spec = spec
            best_thresh = round(float(t), 2)

    return best_thresh


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=32, val_split=0.15, return_val=True
    )

    # Calculate mathematically balanced pos_weight = N_normal / N_pneumonia (~0.35)
    targets = train_loader.dataset.targets
    num_normal = sum(1 for y in targets if y == 0)
    num_pneumonia = sum(1 for y in targets if y == 1)
    pos_weight_val = num_normal / num_pneumonia
    print(f"Train Counts -> Normal: {num_normal} | Pneumonia: {num_pneumonia} | Balanced pos_weight: {pos_weight_val:.4f}")

    pos_weight = torch.tensor([pos_weight_val], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    model = build_resnet18(device)

    # Stage 1: Freeze backbone for 3 epochs to warm up classification head
    warmup_epochs = 3
    total_epochs = 15
    for name, param in model.named_parameters():
        param.requires_grad = ("fc" in name)

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=1e-4)
    scheduler = None

    best_balanced_score = -1.0
    best_weights = copy.deepcopy(model.state_dict())
    best_threshold = 0.50
    patience = 5
    patience_counter = 0

    with open(LOG_CSV_PATH, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch", "train_loss", "val_loss", "val_acc",
            "val_recall", "val_specificity", "val_precision",
            "val_f1", "val_auc", "optimal_threshold"
        ])

    for epoch in range(1, total_epochs + 1):
        # Stage 2: Unfreeze entire ResNet-18 backbone at epoch 4
        if epoch == warmup_epochs + 1:
            print("\n--- Unfreezing full ResNet-18 backbone for fine-tuning ---")
            for param in model.parameters():
                param.requires_grad = True
            optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=(total_epochs - warmup_epochs), eta_min=1e-6
            )

        model.train()
        running_train_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels_f = labels.float().unsqueeze(1).to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels_f)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * images.size(0)

        if scheduler is not None:
            scheduler.step()

        train_loss = running_train_loss / len(train_loader.dataset)

        # Evaluate on validation set & tune threshold
        val_raw = evaluate(model, val_loader, criterion, device, threshold=0.50)
        opt_t = find_optimal_threshold(val_raw["probs"], val_raw["targets"], min_recall=0.92)
        val_metrics = evaluate(model, val_loader, criterion, device, threshold=opt_t)

        # Balanced clinical score: geometric mean of Recall and Specificity
        balanced_score = np.sqrt(val_metrics["recall"] * val_metrics["specificity"])

        print(
            f"Epoch {epoch:02d}/{total_epochs:02d} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_metrics['loss']:.4f} | "
            f"Val Acc: {val_metrics['acc']*100:.2f}% | Recall: {val_metrics['recall']*100:.2f}% | "
            f"Specificity: {val_metrics['specificity']*100:.2f}% | Thresh: {opt_t:.2f}"
        )

        with open(LOG_CSV_PATH, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch, f"{train_loss:.4f}", f"{val_metrics['loss']:.4f}",
                f"{val_metrics['acc']:.4f}", f"{val_metrics['recall']:.4f}",
                f"{val_metrics['specificity']:.4f}", f"{val_metrics['precision']:.4f}",
                f"{val_metrics['f1']:.4f}", f"{val_metrics['auc']:.4f}", opt_t
            ])

        if balanced_score > best_balanced_score:
            best_balanced_score = balanced_score
            best_weights = copy.deepcopy(model.state_dict())
            best_threshold = opt_t
            torch.save(best_weights, MODEL_SAVE_PATH)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}.")
                break

    # Final Evaluation on Untouched 624-Image Test Set
    print("\n================ FINAL TEST SET EVALUATION ================")
    model.load_state_dict(best_weights)
    test_metrics = evaluate(model, test_loader, criterion, device, threshold=best_threshold)
    print(f"Optimal Threshold Used : {best_threshold:.2f}")
    print(f"Test Accuracy          : {test_metrics['acc']*100:.2f}%")
    print(f"Test Recall (Sens.)    : {test_metrics['recall']*100:.2f}%")
    print(f"Test Specificity       : {test_metrics['specificity']*100:.2f}%")
    print(f"Test Precision         : {test_metrics['precision']*100:.2f}%")
    print(f"Test F1-Score          : {test_metrics['f1']:.4f}")
    print(f"Test ROC-AUC           : {test_metrics['auc']:.4f}")
    print(f"Confusion Matrix       : TP={test_metrics['tp']}, TN={test_metrics['tn']}, FP={test_metrics['fp']}, FN={test_metrics['fn']}")
    print(f"Saved weights to '{MODEL_SAVE_PATH}' and logs to '{LOG_CSV_PATH}'.")


if __name__ == "__main__":
    main()