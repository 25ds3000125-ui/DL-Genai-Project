# 🎵 Messy Mashup – Genre Classification

**Author:** Prathap Manohar Joshi  
**ID:** 25DS3000125

---

## 📌 Overview

*Messy Mashup* is a robust music genre classification project that leverages **synthetic mashups** and the **Audio Spectrogram Transformer (AST)** to improve generalization and performance on complex audio data.

## 🌐 Demo

Try out the live demo hosted on Hugging Face Spaces:  
👉 [https://huggingface.co/spaces/pmjoshiBS/AST-messy-mashup](https://huggingface.co/spaces/pmjoshiBS/AST-messy-mashup)

---

## 📂 Project Structure

The project is modularized into a `src/` directory for better maintainability and scalability.

```text
.
├── src/
│   ├── config.py         # Hyperparameters, paths, and global settings
│   ├── utils.py          # Audio loading, Mel-spectrogram conversion, and metrics
│   ├── dataset.py        # Custom PyTorch Datasets (CNNDataset, ASTDataset)
│   ├── models.py         # Model architectures (Simple CNN & AST Transformer)
│   ├── train.py          # Training loops, Mixup logic, and validation
│   └── inference.py      # Test-Time Augmentation (TTA) and submission logic
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation

---
```

## 🧠 Methodology

### 1. Convolutional Neural Network (CNN)
A lightweight 3-layer CNN designed to process 2D Mel-Spectrograms. This serves as a baseline for performance and speed.

### 2. Audio Spectrogram Transformer (AST)
Utilizes the **MIT AST (Audio Spectrogram Transformer)** finetuned on AudioSet. This model treats audio as a sequence of patches, similar to Vision Transformers (ViT), allowing for superior global context understanding.

### 🧬 Data Augmentation & Robustness
* **Dynamic Stem Mixing:** Training samples are created by randomly mixing stems from different tracks to simulate "mashup" conditions.
* **Noise Injection:** Integration of the ESC-50 dataset to add realistic environmental noise.
* **SpecAugment:** Random time and frequency masking on the spectrogram.
* **Mixup:** Blending two training samples and their labels to improve model generalization.
* **TTA (Test-Time Augmentation):** Averaging predictions across multiple random 10-second crops during inference.

---

## 🚀 Getting Started

### Prerequisites
* Python 3.8+
* CUDA-enabled GPU (Recommended)

### Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/25ds3000125-ui/DL-Genai-Project.git
    cd DL-Genai-Project
    ```
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Training & Logging
The project uses **Weights & Biases (W&B)** for experiment tracking. Ensure you are logged in:
```bash
wandb login
```
To start the training pipeline, ensure your data paths are set in `src/config.py` and run your training script or notebook.

---

## 📊 Evaluation

Performance is measured using the **Macro F1-Score** to ensure the model performs consistently across all 10 target genres.

| Model | Technique | Best Macro F1 |
| :--- | :--- | :--- |
| CNN | Baseline | 0.14 |
| AST | Baseline (v1) | 0.78 |
| AST | Mixup + SpecAug (v2) | 0.86 |

---

## ✨ Future Improvements
* Real-time audio inference.
* Expanded genre coverage.
* Model optimization for edge deployment.

## 📝 License
This project is for educational and research purposes as part of the DL-GenAI Project.
