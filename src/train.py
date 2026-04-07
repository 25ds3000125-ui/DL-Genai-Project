import torch
import wandb
import random
import numpy as np
from torch.utils.data import DataLoader
from config import CFG, DEVICE
from utils import macro_f1

# Mixup: Blends two different songs together to help the model generalize better
def mixup(x, y):
    lam = np.random.beta(0.4, 0.4)
    idx = torch.randperm(x.size(0)).to(x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam

# Execution & Training Loops
# Trains the basic CNN model and logs progress to Weights & Biases
def train_cnn(model, train_loader, val_loader, opt, crit):
    best_f1 = 0.0
    for epoch in range(CFG.EPOCHS):
        model.train()
        tloss = 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            out = model(x)
            loss = crit(out, y)
            loss.backward()
            opt.step()
            tloss += loss.item()
        
        # Validation logic here...
    return best_f1

# Trains the AST model (v1 = baseline, v2 = with Mixup augmentation)
def train_ast(model, train_loader, val_loader, opt, crit, sched, use_mixup=True):
    # Training loop logic with Mixup and SpecAug...
    pass
