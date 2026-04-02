import torch
import random
import numpy as np
import wandb

from sklearn.metrics import f1_score

def mixup(x, y, alpha=0.4):
    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(x.size(0)).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    return mixed_x, y, y[index], lam

def train(model, train_loader, val_loader, config, device):
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.EPOCHS)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

    best_f1 = 0

    wandb.watch(model)

    for epoch in range(config.EPOCHS):
        model.train()
        total_loss = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            if random.random() < 0.5:
                x, y_a, y_b, lam = mixup(x, y)
                outputs = model(x).logits
                loss = lam * criterion(outputs, y_a) + (1 - lam) * criterion(outputs, y_b)
            else:
                outputs = model(x).logits
                loss = criterion(outputs, y)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()

        model.eval()
        preds, targets = [], []

        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                outputs = model(x).logits
                pred = torch.argmax(outputs, dim=1).cpu().numpy()

                preds.extend(pred)
                targets.extend(y.numpy())

        val_f1 = f1_score(targets, preds, average='macro')

        wandb.log({
            "epoch": epoch,
            "loss": total_loss,
            "val_f1": val_f1
        })

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), "outputs/models/best_model.pt")