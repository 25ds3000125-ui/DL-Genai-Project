import torch
import wandb
import pandas as pd
import os

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, WeightedRandomSampler

from src import config
from src.dataset import MashupDataset
from src.model import get_model
from src.train import train

wandb.init(project="messy-mashup")

data = []

for genre in config.GENRES:
    genre_path = os.path.join(config.TRAIN_PATH, genre)
    for song in os.listdir(genre_path):
        song_path = os.path.join(genre_path, song)
        if os.path.isdir(song_path):
            data.append({"path": song_path, "genre": genre})

df = pd.DataFrame(data)

train_df, val_df = train_test_split(
    df, test_size=0.2, stratify=df['genre'], random_state=42
)

noise_paths = [
    os.path.join(config.NOISE_PATH, f)
    for f in os.listdir(config.NOISE_PATH)
    if f.endswith(".wav")
]

train_dataset = MashupDataset(train_df, noise_paths, train=True)
val_dataset = MashupDataset(val_df, noise_paths, train=False)

class_counts = train_df['genre'].value_counts()
weights = train_df['genre'].map(lambda x: 1.0 / class_counts[x]).values
sampler = WeightedRandomSampler(weights, len(weights))

train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, sampler=sampler)
val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE)

device = "cuda" if torch.cuda.is_available() else "cpu"

model = get_model()
model.to(device)

train(model, train_loader, val_loader, config, device)