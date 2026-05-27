# MNIST CNN — PyTorch 🧠

A clean, well-documented Convolutional Neural Network trained on the classic **MNIST** handwritten-digit dataset.  
Achieves **~99 % test accuracy** in ≈ 10 epochs on CPU/GPU.

---

## 📁 Project Structure

```
mnist-cnn-pytorch/
├── model.py          # CNN architecture (MnistCNN)
├── train.py          # Training loop + checkpoint saving
├── evaluate.py       # Metrics, confusion matrix, plots
├── main.py           # One-shot runner (train → evaluate)
├── requirements.txt
└── results/          # Generated after training
    ├── training_log.csv
    ├── training_curves.png
    ├── confusion_matrix.png
    ├── sample_predictions.png
    └── classification_report.txt
```

---

## 🏗️ CNN Architecture

```
Input  (1 × 28 × 28)
  │
  ▼
Conv2d(1→32, 3×3) + ReLU   →  32 × 26 × 26
  │
  ▼
Conv2d(32→64, 3×3) + ReLU  →  64 × 24 × 24
  │
  ▼
MaxPool2d(2×2) + Dropout(25%)  →  64 × 12 × 12
  │
  ▼
Flatten  →  9216
  │
  ▼
Linear(9216 → 128) + ReLU + Dropout(50%)
  │
  ▼
Linear(128 → 10)   →  class logits
```

| Layer         | Output Shape      | Parameters |
|---------------|-------------------|------------|
| Conv1 (3×3)   | 32 × 26 × 26      | 320        |
| Conv2 (3×3)   | 64 × 24 × 24      | 18 496     |
| MaxPool       | 64 × 12 × 12      | —          |
| FC1           | 128               | 1 179 776  |
| FC2 (output)  | 10                | 1 290      |
| **Total**     |                   | **~1.2 M** |

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train + Evaluate (one command)
```bash
python main.py
```

### 3. Train only
```bash
python train.py --epochs 10 --batch-size 64 --lr 0.001
```

### 4. Evaluate only (requires a checkpoint)
```bash
python evaluate.py
```

---

## ⚙️ Hyperparameters

| Parameter   | Default | Description               |
|-------------|---------|---------------------------|
| `--epochs`  | 10      | Number of training epochs |
| `--batch-size` | 64  | Mini-batch size           |
| `--lr`      | 0.001   | Adam learning rate        |
| `--seed`    | 42      | Random seed               |

---

## 📊 Results

| Metric             | Value   |
|--------------------|---------|
| Test Accuracy      | ~99 %   |
| Best Val Loss      | ~0.03   |
| Model Parameters   | ~1.2 M  |
| Training Time (CPU)| ~15 min |

### Training Curves
![Training Curves](results/training_curves.png)

### Confusion Matrix
![Confusion Matrix](results/confusion_matrix.png)

### Sample Predictions
![Predictions](results/sample_predictions.png)

---

## 🗂️ Dataset

**MNIST** (Modified National Institute of Standards and Technology)
- **60 000** training images, **10 000** test images
- Grayscale, 28 × 28 pixels, 10 classes (digits 0–9)
- Auto-downloaded by `torchvision.datasets.MNIST` on first run

---

## 📚 Learning Resources

- [PyTorch Docs — nn.Module](https://pytorch.org/docs/stable/generated/torch.nn.Module.html)
- [CS231n CNN Guide](https://cs231n.github.io/convolutional-networks/)
- [MNIST Dataset Info](http://yann.lecun.com/exdb/mnist/)

---

## 📄 License
MIT
