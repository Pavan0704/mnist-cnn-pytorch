"""
CNN Model Architecture for MNIST Classification
================================================
A clean, modular CNN definition using PyTorch nn.Module.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MnistCNN(nn.Module):
    """
    Convolutional Neural Network for MNIST digit classification.

    Architecture:
        Input  : (B, 1, 28, 28)  — grayscale image
        Conv1  : 32 filters, 3×3, ReLU → (B, 32, 26, 26)
        Conv2  : 64 filters, 3×3, ReLU → (B, 64, 24, 24)
        MaxPool: 2×2              → (B, 64, 12, 12)
        Dropout: 25 %
        Flatten: 64 × 12 × 12 = 9216
        FC1    : 128 neurons, ReLU
        Dropout: 50 %
        FC2    : 10 neurons (logits)
    """

    def __init__(self):
        super(MnistCNN, self).__init__()

        # --- Convolutional Layers ---
        self.conv1 = nn.Conv2d(
            in_channels=1,   # grayscale
            out_channels=32,
            kernel_size=3,
            padding=0        # valid conv → 26×26
        )
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            padding=0        # valid conv → 24×24
        )

        # --- Regularisation ---
        self.dropout1 = nn.Dropout(p=0.25)
        self.dropout2 = nn.Dropout(p=0.50)

        # --- Fully-Connected Layers ---
        self.fc1 = nn.Linear(64 * 12 * 12, 128)
        self.fc2 = nn.Linear(128, 10)          # 10 classes (0-9)

    def forward(self, x):
        # Block 1
        x = F.relu(self.conv1(x))              # (B, 32, 26, 26)
        x = F.relu(self.conv2(x))              # (B, 64, 24, 24)
        x = F.max_pool2d(x, kernel_size=2)     # (B, 64, 12, 12)
        x = self.dropout1(x)

        # Flatten
        x = torch.flatten(x, start_dim=1)      # (B, 9216)

        # FC block
        x = F.relu(self.fc1(x))                # (B, 128)
        x = self.dropout2(x)
        x = self.fc2(x)                        # (B, 10)  — raw logits
        return x
