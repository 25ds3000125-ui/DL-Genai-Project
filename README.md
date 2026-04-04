# 🎵 Messy Mashup – Genre Classification

**Author:** Prathap Manohar Joshi
**ID:** 25DS3000125

---

## 📌 Overview

*Messy Mashup* is a robust music genre classification project that leverages **synthetic mashups** and the **Audio Spectrogram Transformer (AST)** to improve generalization and performance on complex audio data.

---

## 🚀 Key Features

* 🎧 **Synthetic Mashup Generation** – Enhances dataset diversity by blending multiple audio samples
* 🎚️ **SpecAugment** – Improves model robustness through spectrogram augmentation
* 🔀 **Mixup Augmentation** – Regularizes training and reduces overfitting
* 🤖 **AST (Audio Spectrogram Transformer)** – State-of-the-art transformer-based audio model
* 📊 **Weights & Biases Tracking** – Experiment monitoring and logging

---

## 🏋️ Training

Run the training pipeline using:

```bash
python train.py
```

---

## 🌐 Demo

Try out the live demo hosted on Hugging Face Spaces:
👉 [https://huggingface.co/spaces/pmjoshiBS/AST-messy-mashup](https://huggingface.co/spaces/pmjoshiBS/AST-messy-mashup)

---

## 📂 Project Structure

```bash
.
├── train.py
├── dataset/
├── models/
├── utils/
└── README.md
```

---

## ✨ Future Improvements

* Real-time audio inference
* Expanded genre coverage
* Model optimization for edge deployment
