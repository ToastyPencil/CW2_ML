from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from cw2_ml.al.modified import select_typiclust_adaptive_indices
from cw2_ml.al.typiclust import select_typiclust_indices
from cw2_ml.data import ContrastivePairDataset, CIFAR10_NUM_CLASSES, build_index_loader, get_cifar10_datasets, get_fake_cifar10_datasets
from cw2_ml.models import build_feature_extractor, build_resnet18_classifier, extract_embeddings
from cw2_ml.models.resnet import build_simclr_resnet18
from cw2_ml.train import (
    evaluate_classifier,
    extract_normalized_embeddings,
    predict_entropy,
    set_global_seed,
    train_classifier,
    train_contrastive_epoch,
)
from cw2_ml.utils.io import ensure_dir


ALL_STRATEGIES = ("random", "entropy", "typiclust", "typiclust_adaptive")
SUPPORTED_STRATEGIES = ALL_STRATEGIES


@dataclass(slots=True)
class ActiveLearningConfig:
    strategy: str = "typiclust"
    data_dir: str = "data"
    output_dir: str = "outputs/final_submission"
    repeats: int = 3
    rounds: int = 5
    query_batch_size: int = 10
    init_labeled_size: int = 0
    train_epochs: int = 20
    batch_size: int = 128
    embedding_batch_size: int = 256
    learning_rate: float = 0.03
    weight_decay: float = 5e-4
    seed: int = 42
    device: str = "auto"
    max_clusters: int = 500
    knn_k: int = 20
    min_cluster_size: int = 5
    novelty_weight: float = 0.15
    novelty_start_round: int = 3
    uncertainty_weight: float = 0.20
    uncertainty_start_round: int = 4
    ssl_batch_size: int = 256
    ssl_pretrain_epochs: int = 50
    ssl_projection_dim: int = 128
    ssl_projection_hidden_dim: int = 512
    ssl_temperature: float = 0.5
    ssl_learning_rate: float = 3e-4
    ssl_weight_decay: float = 1e-6
    ssl_checkpoint_path: str | None = None
    ssl_embeddings_path: str | None = None
    num_workers: int = 2
    smoke: bool = False
    download: bool = True
    fake_train_size: int = 256
    fake_test_size: int = 128


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return device


def _sample_random(rng: np.random.Generator, indices: Iterable[int], budget: int) -> list[int]:
    candidates = np.array(list(indices), dtype=np.int64)
    if candidates.size == 0:
        return []
    budget = min(budget, candidates.size)
    selected = rng.choice(candidates, size=budget, replace=False)
    return selected.astype(int).tolist()


def _load_bundle(config: ActiveLearningConfig):
    if config.smoke:
        return get_fake_cifar10_datasets(train_size=config.fake_train_size, test_size=config.fake_test_size)
    return get_cifar10_datasets(data_dir=config.data_dir, download=config.download)


def _resolve_ssl_paths(config: ActiveLearningConfig) -> tuple[Path, Path]:
    checkpoint_path = Path(config.ssl_checkpoint_path) if config.ssl_checkpoint_path else Path(config.output_dir) / "ssl" / "simclr_resnet18.pt"
    embeddings_path = Path(config.ssl_embeddings_path) if config.ssl_embeddings_path else Path(config.output_dir) / "ssl" / "cifar10_embeddings.npy"
    return checkpoint_path, embeddings_path


def _embedding_cache_metadata_path(embeddings_path: Path) -> Path:
    return embeddings_path.with_suffix(f"{embeddings_path.suffix}.meta.json")


def _expected_embedding_cache_metadata(
    config: ActiveLearningConfig,
    n_samples: int,
) -> dict[str, object]:
    metadata: dict[str, object] = {
        "mode": "smoke" if config.smoke else "ssl",
        "n_samples": int(n_samples),
    }
    if not config.smoke:
        metadata.update(
            {
                "ssl_pretrain_epochs": int(config.ssl_pretrain_epochs),
                "ssl_projection_dim": int(config.ssl_projection_dim),
                "ssl_projection_hidden_dim": int(config.ssl_projection_hidden_dim),
                "ssl_temperature": float(config.ssl_temperature),
                "ssl_learning_rate": float(config.ssl_learning_rate),
                "ssl_weight_decay": float(config.ssl_weight_decay),
            }
        )
    return metadata


def _load_cached_embeddings(
    embeddings_path: Path,
    expected_rows: int,
    expected_metadata: dict[str, object],
) -> np.ndarray | None:
    if not embeddings_path.exists():
        return None

    metadata_path = _embedding_cache_metadata_path(embeddings_path)
    if not metadata_path.exists():
        return None

    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    if metadata != expected_metadata:
        return None

    try:
        embeddings = np.load(embeddings_path)
    except (OSError, ValueError):
        return None

    if embeddings.ndim != 2 or embeddings.shape[0] != expected_rows or embeddings.shape[1] <= 0:
        return None
    if not np.isfinite(embeddings).all():
        return None
    return embeddings


def _save_embedding_cache(
    embeddings: np.ndarray,
    embeddings_path: Path,
    metadata: dict[str, object],
) -> None:
    ensure_dir(embeddings_path.parent)
    np.save(embeddings_path, embeddings)
    _embedding_cache_metadata_path(embeddings_path).write_text(
        json.dumps(metadata, sort_keys=True),
        encoding="utf-8",
    )


def _contrastive_tensor_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(size=32, scale=(0.7, 1.0), antialias=True),
            transforms.RandomHorizontalFlip(),
        ]
    )


def _build_contrastive_loader(
    dataset,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    contrastive_dataset = ContrastivePairDataset(dataset, transform=_contrastive_tensor_transform())
    return DataLoader(
        contrastive_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def _load_or_train_embeddings(
    config: ActiveLearningConfig,
    dataset_bundle,
    device: str,
    num_workers: int,
) -> np.ndarray:
    checkpoint_path, embeddings_path = _resolve_ssl_paths(config)
    expected_metadata = _expected_embedding_cache_metadata(
        config=config,
        n_samples=len(dataset_bundle.train_eval),
    )
    cached_embeddings = _load_cached_embeddings(
        embeddings_path=embeddings_path,
        expected_rows=len(dataset_bundle.train_eval),
        expected_metadata=expected_metadata,
    )
    if cached_embeddings is not None:
        return cached_embeddings

    all_train_indices = range(len(dataset_bundle.train_eval))
    eval_loader = build_index_loader(
        dataset=dataset_bundle.train_eval,
        indices=all_train_indices,
        batch_size=config.embedding_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device == "cuda",
    )

    if config.smoke:
        feature_extractor = build_feature_extractor(pretrained=False)
        embeddings = extract_normalized_embeddings(
            feature_extractor=feature_extractor,
            loader=eval_loader,
            device=device,
            n_samples=len(dataset_bundle.train_eval),
        )
        _save_embedding_cache(
            embeddings=embeddings,
            embeddings_path=embeddings_path,
            metadata=expected_metadata,
        )
        return embeddings

    ssl_model = build_simclr_resnet18(
        pretrained=False,
        projection_dim=config.ssl_projection_dim,
        projection_hidden_dim=config.ssl_projection_hidden_dim,
    )

    if checkpoint_path.exists():
        state = torch.load(checkpoint_path, map_location="cpu")
        ssl_model.load_state_dict(state["model"])
    else:
        optimizer = torch.optim.Adam(
            ssl_model.parameters(),
            lr=config.ssl_learning_rate,
            weight_decay=config.ssl_weight_decay,
        )
        train_loader = _build_contrastive_loader(
            dataset=dataset_bundle.train,
            batch_size=config.ssl_batch_size,
            num_workers=num_workers,
            pin_memory=device == "cuda",
        )
        for _ in range(config.ssl_pretrain_epochs):
            train_contrastive_epoch(
                model=ssl_model,
                train_loader=train_loader,
                optimizer=optimizer,
                device=device,
                temperature=config.ssl_temperature,
            )
        ensure_dir(checkpoint_path.parent)
        torch.save({"model": ssl_model.state_dict()}, checkpoint_path)

    embeddings = extract_normalized_embeddings(
        feature_extractor=ssl_model.encoder,
        loader=eval_loader,
        device=device,
        n_samples=len(dataset_bundle.train_eval),
    )
    _save_embedding_cache(
        embeddings=embeddings,
        embeddings_path=embeddings_path,
        metadata=expected_metadata,
    )
    return embeddings


def _query_by_entropy(
    model: torch.nn.Module | None,
    unlabeled_indices: list[int],
    dataset,
    budget: int,
    batch_size: int,
    device: str,
    num_workers: int,
    rng: np.random.Generator,
) -> list[int]:
    if model is None:
        return _sample_random(rng, unlabeled_indices, budget)
    unlabeled_loader = build_index_loader(
        dataset=dataset,
        indices=unlabeled_indices,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device == "cuda",
    )
    entropy_scores = predict_entropy(model, unlabeled_loader, device=device)
    ranked = sorted(entropy_scores.keys(), key=lambda idx: entropy_scores[idx], reverse=True)
    return [int(idx) for idx in ranked[:budget]]


def _query_by_strategy(
    strategy: str,
    embeddings: np.ndarray | None,
    labeled_indices: set[int],
    unlabeled_indices: list[int],
    budget: int,
    model: torch.nn.Module | None,
    dataset,
    config: ActiveLearningConfig,
    current_round: int,
    device: str,
    num_workers: int,
    rng: np.random.Generator,
) -> list[int]:
    if strategy == "random":
        return _sample_random(rng, unlabeled_indices, budget)

    if strategy == "entropy":
        return _query_by_entropy(
            model=model,
            unlabeled_indices=unlabeled_indices,
            dataset=dataset,
            budget=budget,
            batch_size=config.batch_size,
            device=device,
            num_workers=num_workers,
            rng=rng,
        )

    if embeddings is None:
        raise ValueError("Embeddings are required for TypiClust strategies.")

    if strategy == "typiclust":
        return select_typiclust_indices(
            embeddings=embeddings,
            labeled_indices=labeled_indices,
            budget=budget,
            max_clusters=config.max_clusters,
            knn_k=config.knn_k,
            min_cluster_size=config.min_cluster_size,
            random_state=int(rng.integers(0, 10_000)),
        )

    if strategy == "typiclust_adaptive":
        uncertainty_scores = None
        if model is not None and config.uncertainty_weight > 0.0 and current_round >= config.uncertainty_start_round:
            unlabeled_loader = build_index_loader(
                dataset=dataset,
                indices=unlabeled_indices,
                batch_size=config.batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=device == "cuda",
            )
            uncertainty_scores = predict_entropy(model, unlabeled_loader, device=device)
        return select_typiclust_adaptive_indices(
            embeddings=embeddings,
            labeled_indices=labeled_indices,
            budget=budget,
            round_idx=max(current_round, 1),
            total_rounds=config.rounds,
            novelty_max_weight=config.novelty_weight,
            novelty_start_round=config.novelty_start_round,
            uncertainty_scores=uncertainty_scores,
            uncertainty_max_weight=config.uncertainty_weight,
            uncertainty_start_round=config.uncertainty_start_round,
            max_clusters=config.max_clusters,
            knn_k=config.knn_k,
            min_cluster_size=config.min_cluster_size,
            random_state=int(rng.integers(0, 10_000)),
        )

    raise ValueError(f"Unsupported strategy: {strategy}")


def _single_run(
    config: ActiveLearningConfig,
    strategy: str,
    repeat_id: int,
) -> list[dict[str, object]]:
    seed = config.seed + repeat_id
    set_global_seed(seed)
    rng = np.random.default_rng(seed)
    device = _resolve_device(config.device)
    num_workers = 0 if config.smoke else config.num_workers

    bundle = _load_bundle(config)
    train_size = len(bundle.train)
    all_train_indices = list(range(train_size))

    labeled_indices: set[int] = set(_sample_random(rng, all_train_indices, config.init_labeled_size))
    embeddings = None
    if strategy in {"typiclust", "typiclust_adaptive"}:
        print(
            f"[{strategy}] repeat={repeat_id}: extracting embeddings on device={device} ...",
            flush=True,
        )
        embeddings = _load_or_train_embeddings(
            config=config,
            dataset_bundle=bundle,
            device=device,
            num_workers=num_workers,
        )

    metrics: list[dict[str, object]] = []
    model: torch.nn.Module | None = None
    for round_idx in range(1, config.rounds + 1):
        print(
            f"[{strategy}] repeat={repeat_id}: round {round_idx}/{config.rounds} starting "
            f"(labeled={len(labeled_indices)})",
            flush=True,
        )
        unlabeled_indices = [idx for idx in all_train_indices if idx not in labeled_indices]
        if not unlabeled_indices:
            break

        budget = min(config.query_batch_size, len(unlabeled_indices))
        queried = _query_by_strategy(
            strategy=strategy,
            embeddings=embeddings,
            labeled_indices=labeled_indices,
            unlabeled_indices=unlabeled_indices,
            budget=budget,
            model=model,
            dataset=bundle.train_eval,
            config=config,
            current_round=round_idx,
            device=device,
            num_workers=num_workers,
            rng=rng,
        )

        for idx in queried:
            if idx in unlabeled_indices:
                labeled_indices.add(int(idx))

        if not labeled_indices:
            continue

        model = build_resnet18_classifier(num_classes=CIFAR10_NUM_CLASSES, pretrained=False)
        train_loader = build_index_loader(
            dataset=bundle.train,
            indices=sorted(labeled_indices),
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=device == "cuda",
        )
        test_loader = build_index_loader(
            dataset=bundle.test,
            indices=range(len(bundle.test)),
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=device == "cuda",
        )

        train_classifier(
            model=model,
            train_loader=train_loader,
            device=device,
            epochs=config.train_epochs,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        test_accuracy = evaluate_classifier(model=model, data_loader=test_loader, device=device)
        print(
            f"[{strategy}] repeat={repeat_id}: round {round_idx} done "
            f"(labeled={len(labeled_indices)}, acc={test_accuracy:.4f})",
            flush=True,
        )

        metrics.append(
            {
                "strategy": strategy,
                "repeat": repeat_id,
                "round": round_idx,
                "labeled_count": len(labeled_indices),
                "test_accuracy": test_accuracy,
            }
        )
    return metrics


def run_experiments(config: ActiveLearningConfig, strategies: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for strategy in strategies:
        if strategy not in SUPPORTED_STRATEGIES:
            raise ValueError(f"Unsupported strategy: {strategy}")
        print(f"=== Strategy: {strategy} ===", flush=True)
        for repeat_id in range(config.repeats):
            print(f"--- Repeat {repeat_id + 1}/{config.repeats} ---", flush=True)
            rows.extend(_single_run(config=config, strategy=strategy, repeat_id=repeat_id))
    return pd.DataFrame(rows)


def _save_outputs(df: pd.DataFrame, output_dir: str | Path) -> Path:
    out_dir = ensure_dir(output_dir)
    metrics_path = out_dir / "metrics.csv"
    df.to_csv(metrics_path, index=False)

    if not df.empty:
        summary = (
            df.groupby(["strategy", "round"], as_index=False)
            .agg(
                labeled_count_mean=("labeled_count", "mean"),
                accuracy_mean=("test_accuracy", "mean"),
                accuracy_std=("test_accuracy", "std"),
            )
            .fillna(0.0)
        )
        summary.to_csv(out_dir / "summary.csv", index=False)

    return metrics_path


def run_smoke_experiment(output_dir: str | Path) -> Path:
    config = ActiveLearningConfig(
        strategy="typiclust",
        output_dir=str(output_dir),
        repeats=1,
        rounds=2,
        query_batch_size=8,
        init_labeled_size=0,
        train_epochs=1,
        batch_size=32,
        embedding_batch_size=64,
        learning_rate=0.03,
        smoke=True,
        download=False,
        fake_train_size=96,
        fake_test_size=48,
        device="cpu",
    )
    df = run_experiments(config=config, strategies=[config.strategy])
    return _save_outputs(df, output_dir=output_dir)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CIFAR-10 active learning experiments.")
    parser.add_argument("--strategy", type=str, default="typiclust", help="random|entropy|typiclust|typiclust_adaptive|all")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--output-dir", type=str, default="outputs/final_submission")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--query-batch-size", type=int, default=10)
    parser.add_argument("--init-labeled-size", type=int, default=0)
    parser.add_argument("--train-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--embedding-batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max-clusters", type=int, default=500)
    parser.add_argument("--knn-k", type=int, default=20)
    parser.add_argument("--min-cluster-size", type=int, default=5)
    parser.add_argument("--novelty-weight", type=float, default=0.15)
    parser.add_argument("--novelty-start-round", type=int, default=3)
    parser.add_argument("--uncertainty-weight", type=float, default=0.20)
    parser.add_argument("--uncertainty-start-round", type=int, default=4)
    parser.add_argument("--ssl-batch-size", type=int, default=256)
    parser.add_argument("--ssl-pretrain-epochs", type=int, default=50)
    parser.add_argument("--ssl-projection-dim", type=int, default=128)
    parser.add_argument("--ssl-projection-hidden-dim", type=int, default=512)
    parser.add_argument("--ssl-temperature", type=float, default=0.5)
    parser.add_argument("--ssl-learning-rate", type=float, default=3e-4)
    parser.add_argument("--ssl-weight-decay", type=float, default=1e-6)
    parser.add_argument("--ssl-checkpoint-path", type=str, default=None)
    parser.add_argument("--ssl-embeddings-path", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-download", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.strategy == "all":
        strategies = list(ALL_STRATEGIES)
    else:
        strategies = [args.strategy]

    cfg = ActiveLearningConfig(
        strategy=args.strategy,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        repeats=args.repeats,
        rounds=args.rounds,
        query_batch_size=args.query_batch_size,
        init_labeled_size=args.init_labeled_size,
        train_epochs=args.train_epochs,
        batch_size=args.batch_size,
        embedding_batch_size=args.embedding_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
        device=args.device,
        max_clusters=args.max_clusters,
        knn_k=args.knn_k,
        min_cluster_size=args.min_cluster_size,
        novelty_weight=args.novelty_weight,
        novelty_start_round=args.novelty_start_round,
        uncertainty_weight=args.uncertainty_weight,
        uncertainty_start_round=args.uncertainty_start_round,
        ssl_batch_size=args.ssl_batch_size,
        ssl_pretrain_epochs=args.ssl_pretrain_epochs,
        ssl_projection_dim=args.ssl_projection_dim,
        ssl_projection_hidden_dim=args.ssl_projection_hidden_dim,
        ssl_temperature=args.ssl_temperature,
        ssl_learning_rate=args.ssl_learning_rate,
        ssl_weight_decay=args.ssl_weight_decay,
        ssl_checkpoint_path=args.ssl_checkpoint_path,
        ssl_embeddings_path=args.ssl_embeddings_path,
        num_workers=args.num_workers,
        smoke=args.smoke,
        download=not args.no_download,
    )

    df = run_experiments(config=cfg, strategies=strategies)
    metrics_path = _save_outputs(df, output_dir=cfg.output_dir)
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
