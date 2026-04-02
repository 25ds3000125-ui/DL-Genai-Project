import numpy as np
import librosa
from .config import SR, DURATION, SAMPLES

def load_audio(path):
    y, _ = librosa.load(path, sr=SR, duration=DURATION)
    if len(y) < SAMPLES:
        y = np.pad(y, (0, SAMPLES - len(y)))
    return y[:SAMPLES]

def fix_length(y):
    if len(y) < SAMPLES:
        y = np.pad(y, (0, SAMPLES - len(y)))
    return y[:SAMPLES]