from __future__ import annotations

import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_classifier(
    model: nn.Module,
    train_loader: DataLoader,
    device: str,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
) -> None:
    if len(train_loader) == 0:
        return

    model.to(device)
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=0.9,
        nesterov=True,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))

    for _ in range(epochs):
        for images, labels, _ in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
        scheduler.step()


@torch.no_grad()
def evaluate_classifier(model: nn.Module, data_loader: DataLoader, device: str) -> float:
    model.to(device)
    model.eval()

    total = 0
    correct = 0
    for images, labels, _ in data_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        preds = torch.argmax(logits, dim=1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()
    return float(correct) / float(total) if total > 0 else 0.0


@torch.no_grad()
def predict_entropy(model: nn.Module, data_loader: DataLoader, device: str) -> dict[int, float]:
    model.to(device)
    model.eval()

    entropy_by_index: dict[int, float] = {}
    for images, _, indices in data_loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        probs = torch.softmax(logits, dim=1)
        entropy = -(probs * torch.log(probs.clamp(min=1e-12))).sum(dim=1).detach().cpu().numpy()
        for idx, value in zip(indices.tolist(), entropy.tolist()):
            entropy_by_index[int(idx)] = float(value)
    return entropy_by_index
