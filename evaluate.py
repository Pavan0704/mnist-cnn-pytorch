"""
Evaluation & Visualisation Script — MNIST CNN
==============================================
Loads the best checkpoint, prints a full classification report,
plots the confusion matrix and training curves, and saves a
sample predictions grid — all to the results/ folder.
"""

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")           # headless backend (no display needed)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

from model import MnistCNN


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def load_model(checkpoint_path: str, device):
    model = MnistCNN().to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"  Loaded checkpoint from epoch {ckpt['epoch']}  "
          f"(val_acc={ckpt['val_acc']:.2f}%)")
    return model


def get_test_loader(data_dir: str, batch_size: int = 256):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset = datasets.MNIST(
        root=data_dir, train=False, download=True, transform=transform
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


@torch.no_grad()
def predict_all(model, loader, device):
    all_preds, all_labels, all_probs, all_images = [], [], [], []
    for images, labels in loader:
        images = images.to(device)
        logits = model(images)
        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)

        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.numpy())
        all_probs.append(probs.cpu().numpy())
        all_images.append(images.cpu().numpy())

    return (
        np.concatenate(all_preds),
        np.concatenate(all_labels),
        np.concatenate(all_probs),
        np.concatenate(all_images),
    )


# ─────────────────────────────────────────────
# Plot utilities
# ─────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=list(range(10)),
        yticklabels=list(range(10))
    )
    plt.xlabel("Predicted Label", fontsize=13)
    plt.ylabel("True Label", fontsize=13)
    plt.title("Confusion Matrix — MNIST CNN", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Confusion matrix saved → {save_path}")


def plot_training_curves(log_path: str, save_path: str):
    import csv
    epochs, train_loss, val_loss, train_acc, val_acc = [], [], [], [], []
    with open(log_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            train_loss.append(float(row["train_loss"]))
            val_loss.append(float(row["val_loss"]))
            train_acc.append(float(row["train_acc"]))
            val_acc.append(float(row["val_acc"]))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    ax1.plot(epochs, train_loss, "o-", label="Train Loss", color="#2196F3")
    ax1.plot(epochs, val_loss,   "s--", label="Val Loss",  color="#FF5722")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.set_title("Loss per Epoch", fontweight="bold")
    ax1.legend(); ax1.grid(alpha=0.3)

    # Accuracy
    ax2.plot(epochs, train_acc, "o-", label="Train Acc", color="#4CAF50")
    ax2.plot(epochs, val_acc,   "s--", label="Val Acc",  color="#9C27B0")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Accuracy per Epoch", fontweight="bold")
    ax2.legend(); ax2.grid(alpha=0.3)

    fig.suptitle("MNIST CNN — Training Curves", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Training curves saved  → {save_path}")


def plot_sample_predictions(images, y_true, y_pred, y_prob, save_path, n=36):
    """Plot a 6×6 grid of test images with predicted vs true label."""
    # pick some correct and some wrong predictions for variety
    wrong_idx = np.where(y_pred != y_true)[0][:9]
    right_idx = np.where(y_pred == y_true)[0][:27]
    idx = np.concatenate([right_idx, wrong_idx])[:n]

    fig = plt.figure(figsize=(14, 14))
    gs = gridspec.GridSpec(6, 6, figure=fig, hspace=0.5, wspace=0.3)

    for pos, i in enumerate(idx):
        ax = fig.add_subplot(gs[pos // 6, pos % 6])
        img = images[i, 0]                    # (28, 28)
        ax.imshow(img, cmap="gray")
        ax.axis("off")
        correct = y_pred[i] == y_true[i]
        conf = y_prob[i, y_pred[i]] * 100
        color = "#2e7d32" if correct else "#c62828"
        ax.set_title(
            f"P:{y_pred[i]}  T:{y_true[i]}\n{conf:.1f}%",
            fontsize=8, color=color
        )

    fig.suptitle(
        "Sample Predictions  (green=correct, red=wrong)",
        fontsize=14, fontweight="bold"
    )
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Sample predictions saved → {save_path}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main(args):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else
        "cpu"
    )
    print(f"\n{'='*55}")
    print(f"  MNIST CNN Evaluation")
    print(f"{'='*55}")
    print(f"  Device: {device}")

    os.makedirs(args.results_dir, exist_ok=True)

    # Load model & data
    ckpt_path = os.path.join(args.checkpoint_dir, "best_model.pth")
    model = load_model(ckpt_path, device)
    loader = get_test_loader(args.data_dir)

    # Run inference
    y_pred, y_true, y_prob, images = predict_all(model, loader, device)

    # Classification report
    report = classification_report(
        y_true, y_pred,
        target_names=[str(i) for i in range(10)]
    )
    print(f"\n  Classification Report\n  {'─'*45}")
    for line in report.split("\n"):
        print(f"  {line}")

    report_path = os.path.join(args.results_dir, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write("MNIST CNN — Classification Report\n")
        f.write("="*45 + "\n")
        f.write(report)
    print(f"\n  Report saved → {report_path}")

    # Plots
    plot_confusion_matrix(
        y_true, y_pred,
        os.path.join(args.results_dir, "confusion_matrix.png")
    )
    log_path = os.path.join(args.results_dir, "training_log.csv")
    if os.path.exists(log_path):
        plot_training_curves(
            log_path,
            os.path.join(args.results_dir, "training_curves.png")
        )
    plot_sample_predictions(
        images, y_true, y_pred, y_prob,
        os.path.join(args.results_dir, "sample_predictions.png")
    )

    overall_acc = 100.0 * (y_pred == y_true).sum() / len(y_true)
    print(f"\n  Overall Test Accuracy : {overall_acc:.2f}%")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate CNN on MNIST")
    parser.add_argument("--data-dir",       type=str, default="data")
    parser.add_argument("--results-dir",    type=str, default="results")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    args = parser.parse_args()
    main(args)
