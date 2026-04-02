import os
import random
import numpy as np
from torch.utils.data import Dataset

from .audio import load_audio, fix_length
from .config import GENRES
from .augment import spec_augment

from transformers import AutoFeatureExtractor

extractor = AutoFeatureExtractor.from_pretrained(
    "MIT/ast-finetuned-audioset-10-10-0.4593"
)

def create_mashup(song_paths, noise_paths):
    stems = ['drums.wav', 'vocals.wav', 'bass.wav', 'others.wav']
    mix = np.zeros(len(load_audio(os.path.join(song_paths[0], stems[0]))))

    for stem in stems:
        song = random.choice(song_paths)
        path = os.path.join(song, stem)

        y = load_audio(path)

        try:
            import librosa
            y = librosa.effects.time_stretch(y, random.uniform(0.8, 1.2))
        except:
            pass

        mix += fix_length(y)

    noise = load_audio(random.choice(noise_paths))
    mix += random.uniform(0.05, 0.3) * noise

    mix = mix / (np.max(np.abs(mix)) + 1e-9)

    return mix

class MashupDataset(Dataset):
    def __init__(self, df, noise_paths, train=True):
        self.df = df.reset_index(drop=True)
        self.noise_paths = noise_paths
        self.train = train
        self.genre_to_paths = df.groupby('genre')['path'].apply(list).to_dict()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        genre = self.df.iloc[idx]['genre']
        genre_paths = self.genre_to_paths[genre]

        audio = create_mashup(genre_paths, self.noise_paths)

        features = extractor(
            audio,
            sampling_rate=16000,
            return_tensors="pt"
        )["input_values"].squeeze(0)

        if self.train:
            features = spec_augment(features)

        label = GENRES.index(genre)

        return features, label