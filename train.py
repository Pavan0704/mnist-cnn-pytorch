"""
Training Script — MNIST CNN
============================
Trains MnistCNN, saves the best model checkpoint,
and writes epoch metrics to results/training_log.csv.
"""

import os
import csv
import time
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from model import MnistCNN


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def get_data_loaders(data_dir: str, batch_size: int, num_workers: int = 2):
    """Download MNIST and return train / test DataLoaders."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        # Normalise with MNIST mean/std
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(
        root=data_dir, train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root=data_dir, train=False, download=True, transform=transform
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True, num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=True
    )
    return train_loader, test_loader


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, 100.0 * correct / total


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main(args):
    # ── Reproducibility ──
    torch.manual_seed(args.seed)

    # ── Device ──
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else
        "cpu"
    )
    print(f"\n{'='*55}")
    print(f"  MNIST CNN Training — PyTorch")
    print(f"{'='*55}")
    print(f"  Device      : {device}")
    print(f"  Epochs      : {args.epochs}")
    print(f"  Batch size  : {args.batch_size}")
    print(f"  LR          : {args.lr}")
    print(f"{'='*55}\n")

    # ── Data ──
    train_loader, test_loader = get_data_loaders(
        args.data_dir, args.batch_size
    )

    # ── Model / Loss / Optimiser ──
    model = MnistCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=3, gamma=0.5)

    # ── Output dirs ──
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    log_path = os.path.join(args.results_dir, "training_log.csv")

    best_val_acc = 0.0
    history = []

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr"]
        )

        # ── Training loop ──
        for epoch in range(1, args.epochs + 1):
            t0 = time.time()

            train_loss, train_acc = train_one_epoch(
                model, train_loader, optimizer, criterion, device
            )
            val_loss, val_acc = evaluate(
                model, test_loader, criterion, device
            )
            scheduler.step()

            elapsed = time.time() - t0
            current_lr = optimizer.param_groups[0]["lr"]

            print(
                f"  Epoch [{epoch:02d}/{args.epochs}]  "
                f"Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.2f}%  "
                f"Val Loss: {val_loss:.4f}  Val Acc: {val_acc:.2f}%  "
                f"({elapsed:.1f}s)"
            )

            writer.writerow(
                [epoch, f"{train_loss:.4f}", f"{train_acc:.2f}",
                 f"{val_loss:.4f}", f"{val_acc:.2f}", f"{current_lr:.6f}"]
            )

            history.append(dict(
                epoch=epoch,
                train_loss=train_loss, train_acc=train_acc,
                val_loss=val_loss,   val_acc=val_acc
            ))

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                ckpt_path = os.path.join(
                    args.checkpoint_dir, "best_model.pth"
                )
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_acc,
                }, ckpt_path)
                print(f"    ✔ New best model saved  (val_acc={val_acc:.2f}%)")

    print(f"\n{'='*55}")
    print(f"  Training Complete!")
    print(f"  Best Validation Accuracy : {best_val_acc:.2f}%")
    print(f"  Log saved to             : {log_path}")
    print(f"{'='*55}\n")
    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CNN on MNIST")
    parser.add_argument("--epochs",         type=int,   default=10)
    parser.add_argument("--batch-size",     type=int,   default=64)
    parser.add_argument("--lr",             type=float, default=1e-3)
    parser.add_argument("--seed",           type=int,   default=42)
    parser.add_argument("--data-dir",       type=str,   default="data")
    parser.add_argument("--results-dir",    type=str,   default="results")
    parser.add_argument("--checkpoint-dir", type=str,   default="checkpoints")
    args = parser.parse_args()
    main(args)
