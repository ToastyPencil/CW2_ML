from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


def nt_xent_loss(z_i: torch.Tensor, z_j: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be > 0")

    reps = torch.cat([z_i, z_j], dim=0)
    reps = nn.functional.normalize(reps, dim=1)
    similarity = reps @ reps.T
    similarity = similarity / temperature

    mask = torch.eye(similarity.size(0), device=similarity.device, dtype=torch.bool)
    similarity = similarity.masked_fill(mask, float("-inf"))

    batch_size = z_i.size(0)
    targets = torch.cat(
        [
            torch.arange(batch_size, 2 * batch_size, device=similarity.device),
            torch.arange(0, batch_size, device=similarity.device),
        ],
        dim=0,
    )
    return nn.functional.cross_entropy(similarity, targets)


def train_contrastive_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    temperature: float,
) -> float:
    model = model.to(device)
    model.train()

    total_loss = 0.0
    total_samples = 0
    for x_i, x_j, *_ in train_loader:
        x_i = x_i.to(device, non_blocking=True)
        x_j = x_j.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        z_i = model(x_i)
        z_j = model(x_j)
        loss = nt_xent_loss(z_i, z_j, temperature=temperature)
        loss.backward()
        optimizer.step()

        batch_size = int(x_i.size(0))
        total_loss += float(loss.detach().item()) * batch_size
        total_samples += batch_size

    if total_samples == 0:
        return 0.0
    return total_loss / total_samples


@torch.no_grad()
def extract_normalized_embeddings(
    feature_extractor: nn.Module,
    loader: DataLoader,
    device: str,
    n_samples: int,
) -> np.ndarray:
    feature_extractor = feature_extractor.to(device)
    feature_extractor.eval()

    embeddings = None

    for images, _, indices in loader:
        images = images.to(device, non_blocking=True)
        feats = feature_extractor(images)
        feats = nn.functional.normalize(feats, dim=1, p=2)
        if embeddings is None:
            emb_dim = int(feats.shape[1])
            embeddings = np.zeros((n_samples, emb_dim), dtype=np.float32)
        embeddings[np.asarray(indices, dtype=np.int64)] = feats.detach().cpu().numpy().astype(np.float32)

    if embeddings is None:
        raise ValueError("extract_normalized_embeddings received an empty loader")

    return embeddings
