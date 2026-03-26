import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from cw2_ml.models.resnet import build_simclr_resnet18
from cw2_ml.train import extract_normalized_embeddings, nt_xent_loss, train_contrastive_epoch


class _IndexedTensorDataset(Dataset):
    def __init__(self, features: torch.Tensor, order: list[int] | None = None) -> None:
        self.features = features
        self.order = list(range(int(features.size(0)))) if order is None else order

    def __len__(self) -> int:
        return len(self.order)

    def __getitem__(self, index: int):
        original_index = int(self.order[index])
        feature = self.features[original_index]
        return feature, torch.tensor(0), torch.tensor(original_index)


class _PairDataset(Dataset):
    def __init__(self, left: torch.Tensor, right: torch.Tensor) -> None:
        self.left = left
        self.right = right

    def __len__(self) -> int:
        return int(self.left.size(0))

    def __getitem__(self, index: int):
        return (
            self.left[index],
            self.right[index],
            torch.tensor(0),
            torch.tensor(index),
        )


def _manual_nt_xent(z_i: torch.Tensor, z_j: torch.Tensor, temperature: float) -> torch.Tensor:
    reps = torch.cat([z_i, z_j], dim=0)
    reps = nn.functional.normalize(reps, dim=1)
    similarity = reps @ reps.T
    similarity = similarity / temperature

    mask = torch.eye(similarity.size(0), dtype=torch.bool)
    similarity = similarity.masked_fill(mask, float("-inf"))

    batch_size = z_i.size(0)
    positives = torch.cat(
        [
            torch.arange(batch_size, 2 * batch_size),
            torch.arange(0, batch_size),
        ]
    )
    return nn.functional.cross_entropy(similarity, positives)


def test_build_simclr_resnet18_exposes_encoder_and_projection_head() -> None:
    model = build_simclr_resnet18(pretrained=False, projection_dim=64, projection_hidden_dim=128)

    images = torch.randn(2, 3, 32, 32)
    embeddings = model.encode(images)
    projections = model(images)

    assert embeddings.shape == (2, 512)
    assert projections.shape == (2, 64)
    assert isinstance(model.projector, nn.Sequential)


def test_nt_xent_loss_matches_manual_symmetric_objective() -> None:
    z_i = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    z_j = torch.tensor(
        [
            [0.9, 0.1, 0.0],
            [0.0, 0.8, 0.2],
        ],
        dtype=torch.float32,
    )

    loss = nt_xent_loss(z_i, z_j, temperature=0.5)
    expected = _manual_nt_xent(z_i, z_j, temperature=0.5)

    assert torch.isclose(loss, expected, atol=1e-6)


def test_nt_xent_loss_raises_for_non_positive_temperature() -> None:
    z_i = torch.randn(2, 4)
    z_j = torch.randn(2, 4)

    with pytest.raises(ValueError, match="temperature"):
        nt_xent_loss(z_i, z_j, temperature=0.0)
    with pytest.raises(ValueError, match="temperature"):
        nt_xent_loss(z_i, z_j, temperature=-1.0)


def test_train_contrastive_epoch_updates_model_parameters() -> None:
    torch.manual_seed(0)
    model = build_simclr_resnet18(pretrained=False, projection_dim=16, projection_hidden_dim=32)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    left = torch.randn(4, 3, 32, 32)
    right = left + 0.05 * torch.randn(4, 3, 32, 32)
    loader = DataLoader(_PairDataset(left, right), batch_size=2, shuffle=False)

    before = model.projector[0].weight.detach().clone()
    loss = train_contrastive_epoch(
        model=model,
        train_loader=loader,
        optimizer=optimizer,
        device="cpu",
        temperature=0.5,
    )

    after = model.projector[0].weight.detach()
    assert loss > 0.0
    assert not torch.allclose(before, after)


def test_train_contrastive_epoch_returns_sample_weighted_loss_with_uneven_batches() -> None:
    class _LinearProjector(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layer = nn.Linear(3, 3, bias=False)
            with torch.no_grad():
                self.layer.weight.copy_(torch.eye(3))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.layer(x)

    left = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    right = left + 0.05 * torch.tensor(
        [
            [0.1, -0.2, 0.3],
            [-0.3, 0.2, -0.1],
            [0.2, 0.1, -0.2],
            [-0.1, -0.3, 0.2],
            [0.3, -0.1, 0.1],
        ],
        dtype=torch.float32,
    )
    loader = DataLoader(_PairDataset(left, right), batch_size=2, shuffle=False)

    model = _LinearProjector()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    temperature = 0.5

    with torch.no_grad():
        expected_weighted_sum = 0.0
        expected_count = 0
        batch_losses = []
        for x_i, x_j, *_ in loader:
            z_i = model(x_i)
            z_j = model(x_j)
            batch_loss = float(nt_xent_loss(z_i, z_j, temperature=temperature).item())
            expected_weighted_sum += batch_loss * int(x_i.size(0))
            expected_count += int(x_i.size(0))
            batch_losses.append(batch_loss)
    expected_sample_weighted = expected_weighted_sum / expected_count
    batch_mean = sum(batch_losses) / len(batch_losses)
    assert not np.isclose(expected_sample_weighted, batch_mean)

    epoch_loss = train_contrastive_epoch(
        model=model,
        train_loader=loader,
        optimizer=optimizer,
        device="cpu",
        temperature=temperature,
    )
    assert np.isclose(epoch_loss, expected_sample_weighted, atol=1e-6)


def test_extract_normalized_embeddings_returns_unit_rows_in_index_order() -> None:
    features = torch.tensor(
        [
            [3.0, 0.0, 4.0],
            [0.0, 5.0, 12.0],
            [8.0, 15.0, 0.0],
        ],
        dtype=torch.float32,
    )
    model = nn.Identity()
    loader = DataLoader(_IndexedTensorDataset(features, order=[2, 0, 1]), batch_size=2, shuffle=False)

    embeddings = extract_normalized_embeddings(
        feature_extractor=model,
        loader=loader,
        device="cpu",
        n_samples=3,
    )

    expected = features.numpy().astype(np.float32)
    expected /= np.linalg.norm(expected, axis=1, keepdims=True)

    assert embeddings.shape == (3, 3)
    assert np.allclose(np.linalg.norm(embeddings, axis=1), np.ones(3), atol=1e-6)
    assert np.allclose(embeddings, expected, atol=1e-6)


def test_extract_normalized_embeddings_raises_on_empty_loader() -> None:
    empty_features = torch.empty((0, 3), dtype=torch.float32)
    loader = DataLoader(_IndexedTensorDataset(empty_features), batch_size=2, shuffle=False)

    with pytest.raises(ValueError, match="empty"):
        extract_normalized_embeddings(
            feature_extractor=nn.Identity(),
            loader=loader,
            device="cpu",
            n_samples=0,
        )
