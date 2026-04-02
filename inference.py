import os
import numpy as np
import pandas as pd
import torch
import librosa

from transformers import AutoFeatureExtractor
from src.config import *
from src.model import get_model

device = "cuda" if torch.cuda.is_available() else "cpu"

model = get_model()
model.load_state_dict(torch.load("outputs/models/best_model.pt", map_location=device))
model.to(device)
model.eval()

extractor = AutoFeatureExtractor.from_pretrained(
    "MIT/ast-finetuned-audioset-10-10-0.4593"
)

def load_audio(path):
    y, _ = librosa.load(path, sr=SR)

    if len(y) < SAMPLES:
        y = np.pad(y, (0, SAMPLES - len(y)))
    else:
        y = y[:SAMPLES]

    # normalization
    y = y / (np.max(np.abs(y)) + 1e-9)

    return y

def predict_tta(path, n=5):
    y = load_audio(path)
    preds = []

    for _ in range(n):
        start = np.random.randint(0, len(y) - SAMPLES + 1)
        crop = y[start:start + SAMPLES]

        inputs = extractor(
            crop,
            sampling_rate=SR,
            return_tensors="pt"
        )

        x = inputs["input_values"].to(device)

        with torch.no_grad():
            logits = model(x).logits
            probs = torch.softmax(logits, dim=1).cpu().numpy()

        preds.append(probs)

    preds = np.mean(preds, axis=0)
    return np.argmax(preds)

test_df = pd.read_csv(TEST_CSV)

ids = []
predictions = []

for _, row in test_df.iterrows():
    file_path = os.path.join(TEST_PATH, f"song{row['id']:04d}.wav")

    if not os.path.exists(file_path):
        print("Missing:", file_path)
        continue

    pred_idx = predict_tta(file_path)
    pred_label = GENRES[pred_idx]

    ids.append(row['id'])
    predictions.append(pred_label)

submission = pd.DataFrame({
    "id": ids,
    "genre": predictions
})

os.makedirs("outputs/submissions", exist_ok=True)
submission.to_csv("outputs/submissions/submission.csv", index=False)

print("Submission saved at outputs/submissions/submission.csv")