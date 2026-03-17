from __future__ import annotations

from typing import Iterable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights, resnet18


class _ResNet18Features(nn.Module):
    def __init__(self, weights: ResNet18_Weights | None) -> None:
        super().__init__()
        model = resnet18(weights=weights)
        self.stem = nn.Sequential(*(list(model.children())[:-1]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.stem(x)
        return torch.flatten(feat, 1)


def build_resnet18_classifier(num_classes: int = 10, pretrained: bool = False) -> nn.Module:
    if pretrained:
        try:
            model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        except Exception:
            model = resnet18(weights=None)
    else:
        model = resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def build_feature_extractor(pretrained: bool = True) -> nn.Module:
    if pretrained:
        try:
            return _ResNet18Features(ResNet18_Weights.IMAGENET1K_V1)
        except Exception:
            return _ResNet18Features(None)
    return _ResNet18Features(None)


@torch.no_grad()
def extract_embeddings(
    feature_extractor: nn.Module,
    loader: DataLoader,
    device: str,
    n_samples: int,
) -> np.ndarray:
    feature_extractor = feature_extractor.to(device)
    feature_extractor.eval()

    emb_dim = 512
    embeddings = np.zeros((n_samples, emb_dim), dtype=np.float32)
    for images, _, indices in loader:
        images = images.to(device, non_blocking=True)
        feats = feature_extractor(images).detach().cpu().numpy().astype(np.float32)
        embeddings[np.asarray(indices, dtype=np.int64)] = feats
    return embeddings
