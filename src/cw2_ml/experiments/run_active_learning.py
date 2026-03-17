from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch

from cw2_ml.al.modified import select_typiclust_novelty_indices
from cw2_ml.al.typiclust import select_typiclust_indices
from cw2_ml.data import CIFAR10_NUM_CLASSES, build_index_loader, get_cifar10_datasets, get_fake_cifar10_datasets
from cw2_ml.models import build_feature_extractor, build_resnet18_classifier, extract_embeddings
from cw2_ml.train import evaluate_classifier, predict_entropy, set_global_seed, train_classifier
from cw2_ml.utils.io import ensure_dir


SUPPORTED_STRATEGIES = ("random", "entropy", "typiclust", "typiclust_novelty")


@dataclass(slots=True)
class ActiveLearningConfig:
    strategy: str = "typiclust"
    data_dir: str = "data"
    output_dir: str = "outputs"
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
    device: str = "cuda"
    max_clusters: int = 500
    knn_k: int = 20
    min_cluster_size: int = 5
    novelty_weight: float = 0.2
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

    if strategy == "typiclust_novelty":
        return select_typiclust_novelty_indices(
            embeddings=embeddings,
            labeled_indices=labeled_indices,
            budget=budget,
            novelty_weight=config.novelty_weight,
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
    num_workers = 0 if config.smoke else 2

    bundle = _load_bundle(config)
    train_size = len(bundle.train)
    all_train_indices = list(range(train_size))

    labeled_indices: set[int] = set(_sample_random(rng, all_train_indices, config.init_labeled_size))
    embeddings = None
    if strategy in {"typiclust", "typiclust_novelty"}:
        feat_model = build_feature_extractor(pretrained=not config.smoke)
        eval_loader = build_index_loader(
            dataset=bundle.train_eval,
            indices=all_train_indices,
            batch_size=config.embedding_batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        embeddings = extract_embeddings(
            feature_extractor=feat_model,
            loader=eval_loader,
            device=device,
            n_samples=train_size,
        )

    metrics: list[dict[str, object]] = []
    model: torch.nn.Module | None = None
    for round_idx in range(1, config.rounds + 1):
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
        )
        test_loader = build_index_loader(
            dataset=bundle.test,
            indices=range(len(bundle.test)),
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=num_workers,
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
        for repeat_id in range(config.repeats):
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
    parser.add_argument("--strategy", type=str, default="typiclust", help="random|entropy|typiclust|typiclust_novelty|all")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--output-dir", type=str, default="outputs/active_learning")
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
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-clusters", type=int, default=500)
    parser.add_argument("--knn-k", type=int, default=20)
    parser.add_argument("--min-cluster-size", type=int, default=5)
    parser.add_argument("--novelty-weight", type=float, default=0.2)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-download", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.strategy == "all":
        strategies = list(SUPPORTED_STRATEGIES)
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
        smoke=args.smoke,
        download=not args.no_download,
    )

    df = run_experiments(config=cfg, strategies=strategies)
    metrics_path = _save_outputs(df, output_dir=cfg.output_dir)
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
