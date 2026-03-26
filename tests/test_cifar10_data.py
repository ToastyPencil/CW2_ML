import torch
from torch.utils.data import Dataset

import pytest

from cw2_ml.data import ContrastivePairDataset, build_index_loader, get_fake_cifar10_datasets


class TwoTupleDataset(Dataset):
    def __len__(self) -> int:
        return 4

    def __getitem__(self, item: int):
        image = torch.zeros(3, 32, 32, dtype=torch.float32) + float(item)
        label = item % 10
        return image, label


class ThreeTupleDataset(Dataset):
    def __len__(self) -> int:
        return 4

    def __getitem__(self, item: int):
        image = torch.zeros(3, 32, 32, dtype=torch.float32) + float(item)
        label = item % 10
        return image, label, 100 + item


class CountingTransform:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return image + float(self.calls)


def test_contrastive_pair_dataset_applies_transform_and_preserves_index() -> None:
    transform = CountingTransform()
    dataset = ContrastivePairDataset(ThreeTupleDataset(), transform=transform)

    view_a, view_b, label, index = dataset[0]

    assert transform.calls == 2
    assert not torch.equal(view_a, view_b)
    assert view_a.shape == view_b.shape == (3, 32, 32)
    assert isinstance(int(label), int)
    assert index == 100


def test_contrastive_pair_dataset_falls_back_to_local_index() -> None:
    dataset = ContrastivePairDataset(TwoTupleDataset(), transform=CountingTransform())

    _, _, _, index = dataset[2]

    assert index == 2


def test_contrastive_pair_dataset_requires_a_transform() -> None:
    with pytest.raises(ValueError, match="transform"):
        ContrastivePairDataset(TwoTupleDataset())


def test_fake_cifar10_bundle_index_loader_preserves_global_indices() -> None:
    bundle = get_fake_cifar10_datasets(train_size=6, test_size=2)
    loader = build_index_loader(
        bundle.train_eval,
        [1, 4, 5],
        batch_size=3,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    _, _, indices = next(iter(loader))

    assert indices.tolist() == [1, 4, 5]


def test_build_index_loader_allows_explicit_pin_memory_override() -> None:
    bundle = get_fake_cifar10_datasets(train_size=6, test_size=2)
    loader = build_index_loader(
        bundle.train_eval,
        [0, 1, 2],
        batch_size=3,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    assert loader.pin_memory is False
