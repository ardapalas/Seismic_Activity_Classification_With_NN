# Seismic Activity Binary Classification

A simple neural network built with PyTorch to classify seismic events based on underground wave energy and vibration axis variation.

## Overview

This project implements a binary classification model to predict whether a seismic event has been detected using two numerical features. Built as a hands-on exercise to practice core deep learning concepts including custom model architecture, BCEWithLogitsLoss, and the PyTorch training loop.

## Dataset

**File:** `seismic_activity_svm.csv` — 400 samples, 3 columns

| Feature | Description |
|---------|-------------|
| `underground_wave_energy` | Underground wave energy measurement (float) |
| `vibration_axis_variation` | Vibration axis variation (float) |
| `seismic_event_detected` | Target — 0: not detected, 1: detected |

## Model Architecture

```
Input (2) → Linear(2, 10) → ReLU → Linear(10, 1) → BCEWithLogitsLoss
```

- **Input:** 2 features
- **Hidden layer:** 10 neurons with ReLU activation
- **Output:** 1 neuron (raw logit — sigmoid applied internally by loss function)

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Loss function | `BCEWithLogitsLoss` |
| Optimizer | Adam (lr=0.001) |
| Epochs | 500 |
| Train/Test split | 80% / 20% |
| Random seed | 42 |

## Setup

```bash
pip install torch pandas numpy matplotlib seaborn scikit-learn
```

```bash
jupyter notebook Untitled.ipynb
```

## Project Structure

```
├── Seismic_Activity_Classification.ipynb                # Main notebook
├── seismic_activity_svm.csv      # Dataset
└── README.md
```

## Lessons Learned

- **Never use `torch.round()` in loss computation.** It kills gradients (zero derivative), preventing the model from learning. Use it only for accuracy calculation.
- **`torch.compile()` can crash the kernel on macOS.** Not worth it for small datasets on CPU — skip it unless you're on CUDA with large-scale training.
- **`BCEWithLogitsLoss` includes sigmoid internally.** No need to apply sigmoid before passing logits to the loss function.

## Tech Stack

Python 3.13 · PyTorch · Pandas · NumPy · Matplotlib · Seaborn · scikit-learn
