from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors


def _cluster_points(
    embeddings: np.ndarray,
    cluster_count: int,
    random_state: int,
) -> np.ndarray:
    if cluster_count <= 50:
        clusterer = KMeans(n_clusters=cluster_count, n_init=10, random_state=random_state)
    else:
        clusterer = MiniBatchKMeans(
            n_clusters=cluster_count,
            n_init=10,
            batch_size=1024,
            random_state=random_state,
        )
    return clusterer.fit_predict(embeddings)


def _typicality_scores(cluster_embeddings: np.ndarray, knn_k: int) -> np.ndarray:
    if cluster_embeddings.shape[0] == 1:
        return np.array([np.inf], dtype=np.float64)

    k = min(max(knn_k, 1), cluster_embeddings.shape[0] - 1)
    neighbors = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    neighbors.fit(cluster_embeddings)
    distances, _ = neighbors.kneighbors(cluster_embeddings)
    mean_dist = distances[:, 1:].mean(axis=1)
    return 1.0 / (mean_dist + 1e-12)


def _choose_cluster(
    cluster_to_indices: dict[int, list[int]],
    effective_labeled: set[int],
    selected_set: set[int],
    min_cluster_size: int,
) -> int | None:
    candidate_meta: list[tuple[int, int, int]] = []
    for cluster_id, members in cluster_to_indices.items():
        if len(members) < min_cluster_size:
            continue
        available = [idx for idx in members if idx not in effective_labeled and idx not in selected_set]
        if not available:
            continue
        labeled_count = sum(idx in effective_labeled or idx in selected_set for idx in members)
        candidate_meta.append((cluster_id, labeled_count, len(members)))

    if not candidate_meta:
        return None

    candidate_meta.sort(key=lambda item: (item[1], -item[2], item[0]))
    return candidate_meta[0][0]


def select_typiclust_indices(
    embeddings: np.ndarray,
    labeled_indices: Iterable[int],
    budget: int,
    max_clusters: int = 500,
    knn_k: int = 20,
    min_cluster_size: int = 5,
    random_state: int = 42,
) -> list[int]:
    """Select query indices following TypiClust (TPC-RP) rules.

    The method clusters the entire pool, repeatedly picks a cluster with the fewest
    labeled points (breaking ties by larger cluster size), then selects the most
    typical point in that cluster using inverse kNN distance.
    """
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
        member_scores = _typicality_scores(member_embeddings, knn_k=min(knn_k, len(members)))

        best_idx = None
        best_score = float("-inf")
        for local_pos, global_idx in enumerate(members):
            if global_idx in effective_labeled or global_idx in selected_set:
                continue
            score = float(member_scores[local_pos])
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
