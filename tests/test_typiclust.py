import numpy as np

from cw2_ml.al.modified import select_typiclust_adaptive_indices
from cw2_ml.al.typiclust import select_typiclust_indices


def test_typiclust_returns_requested_number_without_labeled_overlap() -> None:
    rng = np.random.default_rng(7)
    embeddings = rng.normal(size=(40, 8)).astype(np.float32)
    labeled = {0, 1, 2}

    selected = select_typiclust_indices(
        embeddings=embeddings,
        labeled_indices=labeled,
        budget=5,
        max_clusters=20,
        knn_k=5,
        random_state=7,
    )

    assert len(selected) == 5
    assert labeled.isdisjoint(set(selected))
    assert len(set(selected)) == len(selected)


def test_typiclust_prefers_clusters_with_fewer_labeled_points() -> None:
    rng = np.random.default_rng(0)
    cluster_a = rng.normal(loc=-5.0, scale=0.1, size=(12, 4))
    cluster_b = rng.normal(loc=5.0, scale=0.1, size=(12, 4))
    embeddings = np.vstack([cluster_a, cluster_b]).astype(np.float32)

    labeled = set(range(0, 4))
    selected = select_typiclust_indices(
        embeddings=embeddings,
        labeled_indices=labeled,
        budget=1,
        max_clusters=2,
        knn_k=5,
        random_state=11,
    )

    assert selected[0] >= 12


def test_adaptive_typiclust_matches_typiclust_when_novelty_weight_is_zero() -> None:
    embeddings = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [0.2, 0.0],
            [4.0, 0.0],
            [4.1, 0.0],
            [4.2, 0.0],
        ],
        dtype=np.float32,
    )

    typiclust_selected = select_typiclust_indices(
        embeddings=embeddings,
        labeled_indices={0},
        budget=1,
        max_clusters=1,
        knn_k=2,
        random_state=3,
    )
    adaptive_selected = select_typiclust_adaptive_indices(
        embeddings=embeddings,
        labeled_indices={0},
        budget=1,
        round_idx=1,
        total_rounds=5,
        novelty_max_weight=1.0,
        novelty_start_round=3,
        max_clusters=1,
        knn_k=2,
        random_state=3,
    )

    assert adaptive_selected == typiclust_selected


def test_adaptive_typiclust_prefers_novel_points_later() -> None:
    embeddings = np.array(
        [
            [0.0, 0.0],  # labeled
            [1.0, 0.0],
            [1.0, 0.1],
            [4.0, 0.0],
            [4.0, 0.1],
        ],
        dtype=np.float32,
    )

    selected = select_typiclust_adaptive_indices(
        embeddings=embeddings,
        labeled_indices={0},
        budget=1,
        round_idx=4,
        total_rounds=5,
        novelty_max_weight=2.0,
        novelty_start_round=1,
        max_clusters=1,
        knn_k=2,
        random_state=3,
    )
    assert selected[0] in {3, 4}


def test_adaptive_typiclust_handles_singleton_cluster_without_nan_fallback() -> None:
    embeddings = np.array(
        [
            [0.0, 0.0],  # labeled
            [0.1, 0.0],
            [5.0, 0.0],  # singleton cluster that should be queried
        ],
        dtype=np.float32,
    )

    selected = select_typiclust_adaptive_indices(
        embeddings=embeddings,
        labeled_indices={0},
        budget=1,
        round_idx=4,
        total_rounds=5,
        novelty_max_weight=1.0,
        novelty_start_round=1,
        max_clusters=2,
        knn_k=1,
        min_cluster_size=1,
        random_state=0,
    )

    assert selected == [2]


def test_adaptive_typiclust_prefers_uncertain_points_in_late_rounds() -> None:
    embeddings = np.array(
        [
            [0.0, 0.0],  # labeled
            [2.0, 0.0],
            [2.1, 0.0],
            [2.2, 0.0],
        ],
        dtype=np.float32,
    )

    selected = select_typiclust_adaptive_indices(
        embeddings=embeddings,
        labeled_indices={0},
        budget=1,
        round_idx=5,
        total_rounds=5,
        novelty_max_weight=0.0,
        novelty_start_round=2,
        uncertainty_scores={1: 0.1, 2: 0.9, 3: 0.2},
        uncertainty_max_weight=10.0,
        uncertainty_start_round=4,
        max_clusters=1,
        knn_k=2,
        min_cluster_size=1,
        random_state=3,
    )

    assert selected == [2]
