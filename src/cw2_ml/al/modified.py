from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import numpy as np

from .typiclust import _choose_cluster, _cluster_points, _typicality_scores


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


def select_typiclust_novelty_indices(
    embeddings: np.ndarray,
    labeled_indices: Iterable[int],
    budget: int,
    novelty_weight: float = 0.2,
    max_clusters: int = 500,
    knn_k: int = 20,
    min_cluster_size: int = 5,
    random_state: int = 42,
) -> list[int]:
    """TypiClust variant that adds a novelty term against already-labeled points."""
    if budget <= 0 or embeddings.size == 0:
        return []

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
        available_typ = np.array([typicality[pos] for pos in available_positions], dtype=np.float64)
        available_nov = _novelty_scores(embeddings, available, effective_labeled)

        score = available_typ + novelty_weight * available_nov
        best_local = int(np.argmax(score))
        best_idx = available[best_local]

        selected.append(best_idx)
        selected_set.add(best_idx)
        effective_labeled.add(best_idx)

    if len(selected) < budget:
        remaining = [i for i in unlabeled if i not in selected_set]
        selected.extend(remaining[: budget - len(selected)])

    return selected
