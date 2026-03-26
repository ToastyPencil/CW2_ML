from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

CIFAR10_NUM_CLASSES = 10
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

@dataclass(slots=True)
class DatasetBundle:
    train: Dataset
    train_eval: Dataset
    test: Dataset


class ContrastivePairDataset(Dataset):
    def __init__(
        self,
        base_dataset: Dataset,
        transform: Callable[[object], object] | None = None,
    ) -> None:
        if transform is None:
            raise ValueError("ContrastivePairDataset requires a transform for two augmented views")
        self.base_dataset = base_dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, item: int):
        sample = self.base_dataset[item]
        if len(sample) == 2:
            image, label = sample
            original_index = int(item)
        else:
            image, label, original_index = sample[:3]
        view_a = self.transform(image)
        view_b = self.transform(image)
        return view_a, view_b, int(label), int(original_index)


class IndexSubset(Dataset):
    def __init__(self, base_dataset: Dataset, indices: Sequence[int]) -> None:
        self.base_dataset = base_dataset
        self.indices = list(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        global_index = int(self.indices[item])
        x, y = self.base_dataset[global_index]
        return x, y, global_index


def _train_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

def _eval_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])


def get_cifar10_datasets(data_dir: str, download: bool = True) -> DatasetBundle:
    train = datasets.CIFAR10(root=data_dir, train=True, transform=_train_transform(), download=download)
    train_eval = datasets.CIFAR10(root=data_dir, train=True, transform=_eval_transform(), download=download)
    test = datasets.CIFAR10(root=data_dir, train=False, transform=_eval_transform(), download=download)
    return DatasetBundle(train=train, train_eval=train_eval, test=test)

def get_fake_cifar10_datasets(train_size: int = 256, test_size: int = 128) -> DatasetBundle:
    train = datasets.FakeData(size=train_size, image_size=(3, 32, 32), num_classes=CIFAR10_NUM_CLASSES, transform=_train_transform(), random_offset=0)
    train_eval = datasets.FakeData(size=train_size, image_size=(3, 32, 32), num_classes=CIFAR10_NUM_CLASSES, transform=_eval_transform(), random_offset=0)
    test = datasets.FakeData(size=test_size, image_size=(3, 32, 32), num_classes=CIFAR10_NUM_CLASSES, transform=_eval_transform(), random_offset=1_000_000)
    return DatasetBundle(train=train, train_eval=train_eval, test=test)


def build_index_loader(
    dataset: Dataset,
    indices: Iterable[int],
    batch_size: int,
    shuffle: bool,
    num_workers: int = 2,
    pin_memory: bool | None = None,
) -> DataLoader:
    subset = IndexSubset(dataset, list(indices))
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available() if pin_memory is None else pin_memory,
    )
