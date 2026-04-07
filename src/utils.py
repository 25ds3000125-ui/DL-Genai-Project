#!/usr/bin/env python
# coding: utf-8

import os
import random
import numpy as np
import librosa
from config import CFG

# Audio Helper Functions
# Standardizes audio length (padding/cropping) and converts waveforms to Mel-Spectrograms
def load_audio(path, sr, samples):
    y, _ = librosa.load(path, sr=sr)
    if len(y) < samples:
        y = np.pad(y, (0, samples - len(y)))
    return y[:samples]

def get_mel(y):
    mel = librosa.feature.melspectrogram(y=y, sr=CFG.SR, n_mels=CFG.N_MELS)
    mel = librosa.power_to_db(mel).astype(np.float32)
    if mel.shape[1] < CFG.MEL_WIDTH:
        mel = np.pad(mel, ((0, 0), (0, CFG.MEL_WIDTH - mel.shape[1])))
    return mel[:, :CFG.MEL_WIDTH]

# Data Augmentation: Randomly masks parts of the spectrogram to prevent overfitting
def spec_aug(x):
    if random.random() < 0.5:
        t = random.randint(0, 100)
        t0 = random.randint(0, x.shape[0] - t)
        x[t0:t0+t, :] = 0
    if random.random() < 0.5:
        f = random.randint(0, 20)
        f0 = random.randint(0, x.shape[1] - f)
        x[:, f0:f0+f] = 0
    return x
