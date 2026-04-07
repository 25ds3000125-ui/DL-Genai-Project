import torch
import numpy as np
import random
from utils import load_audio
from config import CFG, DEVICE

# Final Inference & Submission
# Uses 'Test-Time Augmentation' (TTA) by averaging predictions from multiple audio crops
def predict_tta(path, model, extractor, n=5):
    y = load_audio(path, CFG.AST_SR, CFG.AST_SAMPLES)
    y = y / (np.max(np.abs(y)) + 1e-9)
    preds = []
    for _ in range(n):
        start = random.randint(0, len(y) - CFG.AST_SAMPLES)
        crop = y[start:start + CFG.AST_SAMPLES]
        x = extractor(crop, sampling_rate=CFG.AST_SR, return_tensors="pt")["input_values"].to(DEVICE)
        with torch.no_grad():
            logits = model(x).logits
            probs = torch.softmax(logits, dim=1)
            preds.append(probs.cpu().numpy())
    return np.argmax(np.mean(preds, axis=0))
