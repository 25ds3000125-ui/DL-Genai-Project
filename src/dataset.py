import os
import torch
import random
import numpy as np
from torch.utils.data import Dataset
from config import CFG, LABEL2IDX, GENRES
from utils import load_audio, get_mel

# CNN Approach
# Dataset class that randomly mixes stems drums, bass, etc. and adds noise for training
class CNNDataset(Dataset):
    def __init__(self, df, noise_files, train=True):
        self.df = df.reset_index(drop=True)
        self.noise_files = noise_files
        self.train = train
        self.stems = ["drums.wav", "bass.wav", "vocals.wav", "others.wav"]

    def __len__(self):
        return len(self.df)

    def random_stem(self, genre):
        songs = self.df[self.df.genre == genre]["path"].tolist()
        for _ in range(10):
            song = random.choice(songs)
            stem = random.choice(self.stems)
            path = os.path.join(song, stem)
            if os.path.exists(path):
                return load_audio(path, CFG.SR, CFG.SAMPLES)
        return np.zeros(CFG.SAMPLES)

    def __getitem__(self, idx):
        genre = self.df.iloc[idx]["genre"]
        y = sum([self.random_stem(genre) for _ in range(4)]) / 4
        if self.train:
            try: y = librosa.effects.time_stretch(y, random.uniform(0.8, 1.2))
            except: pass
        y = y[:CFG.SAMPLES] if len(y) >= CFG.SAMPLES else np.pad(y, (0, CFG.SAMPLES - len(y)))
        if self.train:
            noise = load_audio(random.choice(self.noise_files), CFG.SR, CFG.SAMPLES)
            y = y + random.uniform(0.1, 0.5) * noise
        mel = np.expand_dims(get_mel(y), 0)
        return torch.tensor(mel), torch.tensor(LABEL2IDX[genre])

# Data Augmentation: Randomly masks parts of the spectrogram to prevent overfitting
def spec_aug(x):
    if random.random() < 0.5:
        t = random.randint(0, 100)
        t0 = random.randint(0, x.shape[0] - t)
        x[t0:t0 + t, :] = 0
    if random.random() < 0.5:
        f = random.randint(0, 20)
        f0 = random.randint(0, x.shape[1] - f)
        x[:, f0:f0 + f] = 0
    return x

class ASTDataset(Dataset):
    def __init__(self, df, extractor, noise_files, train=True):
        self.df = df.reset_index(drop=True)
        self.extractor = extractor
        self.noise_files = noise_files
        self.train = train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        genre = self.df.iloc[idx]["genre"]
        paths = self.df[self.df.genre == genre]["path"].tolist()
        
        # Audio creation logic
        mix = np.zeros(CFG.AST_SAMPLES)
        for stem in ["drums.wav", "vocals.wav", "bass.wav", "others.wav"]:
            song = random.choice(paths)
            y = load_audio(os.path.join(song, stem), CFG.AST_SR, CFG.AST_SAMPLES)
            mix += y
        
        feat = self.extractor(mix, sampling_rate=CFG.AST_SR, return_tensors="pt")["input_values"].squeeze(0)
        if self.train:
            feat = spec_aug(feat)
        return feat, GENRES.index(genre)
