# Deep Learning-Based Facial Recognition Model for Security Applications

MSc Advanced Computer Science Dissertation — Leeds Beckett University, 2026

**Student:** Anuj Gautam | C7644819  
**Supervisor:** Andrew Scholey  

---

## Overview

This repository contains the complete implementation of a deep learning facial recognition system comparing a custom Baseline CNN against a MobileNetV2 transfer learning model. Models are trained on the Pins Face Recognition dataset and cross-evaluated on the Labelled Faces in the Wild (LFW) dataset using cosine similarity-based face verification.

---

## Key Results

| Model | Pins Accuracy | LFW Accuracy | Generalisation Gap |
|---|---|---|---|
| Baseline CNN | 32.80% | 66.60% | -33.80% |
| MobileNetV2 | 83.39% | 79.90% | +3.49% |

---

## Datasets

- **Pins Face Recognition** — 105 classes, 17,534 images: [Kaggle](https://www.kaggle.com/datasets/hereisburak/pins-face-recognition)
- **LFW Deep-Funneled** — 13,233 images, 1,000 verification pairs: [Kaggle](https://www.kaggle.com/datasets/jessicali9530/lfw-dataset)

---

## Repository Structure

```
├── notebook/
│   └── facial_recognition_dissertation.ipynb
├── requirements.txt
└── README.md
```

---

## Requirements

```
tensorflow>=2.20.0
scikit-learn
numpy
pandas
matplotlib
seaborn
Pillow
```

---

## Methodology

CRISP-DM six-phase framework. Two separate preprocessing pipelines: `rescale=1/255` for Baseline CNN and `preprocess_input` for MobileNetV2. Three-phase MobileNetV2 training with cosine decay and label smoothing.
