from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import numpy as np

from .typiclust import (
    _choose_cluster,
    _cluster_points,
    _typicality_scores,
    select_typiclust_indices,
)


def _novelty_scores(
    embeddings: np.ndarray,
    candidate_indices: list[int],
    labeled_indices: set[int],
) -> np.ndarray:
    if not labeled_indices:
        return np.zeros(len(candidate_indices), dtype=np.float64)

    labeled_embeddings = embeddings[np.array(sorted(labeled_indices))]
    candidate_embeddings = embeddings[np.array(candidate_indices)]
    distances = np.linalg.norm(
        candidate_embeddings[:, None, :] - labeled_embeddings[None, :, :],
        axis=2,
    )
    return distances.min(axis=1)


def _adaptive_lambda(
    round_idx: int,
    total_rounds: int,
    novelty_max_weight: float,
    novelty_start_round: int,
) -> float:
    if novelty_max_weight <= 0.0 or round_idx < novelty_start_round:
        return 0.0

    denom = max(total_rounds - novelty_start_round, 1)
    progress = (round_idx - novelty_start_round + 1) / denom
    progress = float(min(max(progress, 0.0), 1.0))
    return float(novelty_max_weight * progress)


def _normalize(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values.astype(np.float64, copy=False)

    normalized = np.zeros_like(values, dtype=np.float64)
    finite_mask = np.isfinite(values)
    if finite_mask.any():
        finite_values = values[finite_mask]
        vmin = float(finite_values.min())
        vmax = float(finite_values.max())
        if abs(vmax - vmin) >= 1e-12:
            normalized[finite_mask] = (finite_values - vmin) / (vmax - vmin)
    normalized[np.isposinf(values)] = 1.0
    normalized[np.isneginf(values)] = 0.0
    normalized[np.isnan(values)] = 0.0
    return normalized


def select_typiclust_adaptive_indices(
    embeddings: np.ndarray,
    labeled_indices: Iterable[int],
    budget: int,
    round_idx: int,
    total_rounds: int,
    novelty_max_weight: float = 0.2,
    novelty_start_round: int = 2,
    uncertainty_scores: dict[int, float] | None = None,
    uncertainty_max_weight: float = 0.0,
    uncertainty_start_round: int = 4,
    max_clusters: int = 500,
    knn_k: int = 20,
    min_cluster_size: int = 5,
    random_state: int = 42,
) -> list[int]:
    """TypiClust with a round-dependent novelty term against labeled points.

    ``round_idx`` is interpreted as the 1-based active learning round number.
    """
    if budget <= 0 or embeddings.size == 0:
        return []

    novelty_weight = _adaptive_lambda(
        round_idx=round_idx,
        total_rounds=total_rounds,
        novelty_max_weight=novelty_max_weight,
        novelty_start_round=novelty_start_round,
    )
    uncertainty_weight = _adaptive_lambda(
        round_idx=round_idx,
        total_rounds=total_rounds,
        novelty_max_weight=uncertainty_max_weight,
        novelty_start_round=uncertainty_start_round,
    )
    if novelty_weight <= 0.0 and uncertainty_weight <= 0.0:
        return select_typiclust_indices(
            embeddings=embeddings,
            labeled_indices=labeled_indices,
            budget=budget,
            max_clusters=max_clusters,
            knn_k=knn_k,
            min_cluster_size=min_cluster_size,
            random_state=random_state,
        )

    n = embeddings.shape[0]
    labeled = {int(i) for i in labeled_indices if 0 <= int(i) < n}
    unlabeled = [i for i in range(n) if i not in labeled]
    if not unlabeled:
        return []

    budget = min(budget, len(unlabeled))
    cluster_count = max(1, min(len(labeled) + budget, max_clusters, n))

    labels = _cluster_points(embeddings, cluster_count, random_state)
    cluster_to_indices: dict[int, list[int]] = defaultdict(list)
    for idx, cluster_id in enumerate(labels.tolist()):
        cluster_to_indices[cluster_id].append(idx)

    selected: list[int] = []
    selected_set: set[int] = set()
    effective_labeled = set(labeled)

    while len(selected) < budget:
        chosen_cluster = _choose_cluster(
            cluster_to_indices=cluster_to_indices,
            effective_labeled=effective_labeled,
            selected_set=selected_set,
            min_cluster_size=min_cluster_size,
        )
        if chosen_cluster is None:
            break

        members = cluster_to_indices[chosen_cluster]
        member_embeddings = embeddings[np.array(members)]
        typicality = _typicality_scores(member_embeddings, knn_k=min(knn_k, len(members)))

        available = [idx for idx in members if idx not in effective_labeled and idx not in selected_set]
        if not available:
            break

        member_positions = {idx: pos for pos, idx in enumerate(members)}
        available_positions = [member_positions[idx] for idx in available]
        available_typ = _normalize(np.array([typicality[pos] for pos in available_positions], dtype=np.float64))
        available_nov = _normalize(_novelty_scores(embeddings, available, effective_labeled))
        available_unc = _normalize(
            np.array(
                [float(uncertainty_scores.get(idx, 0.0)) if uncertainty_scores is not None else 0.0 for idx in available],
                dtype=np.float64,
            )
        )

        best_idx = None
        best_score = float("-inf")
        for local_pos, global_idx in enumerate(available):
            score = float(
                available_typ[local_pos]
                + novelty_weight * available_nov[local_pos]
                + uncertainty_weight * available_unc[local_pos]
            )
            if score > best_score or (score == best_score and (best_idx is None or global_idx < best_idx)):
                best_score = score
                best_idx = global_idx

        if best_idx is None:
            break

        selected.append(best_idx)
        selected_set.add(best_idx)
        effective_labeled.add(best_idx)

    if len(selected) < budget:
        remaining = [i for i in unlabeled if i not in selected_set]
        selected.extend(remaining[: budget - len(selected)])

    return selected
