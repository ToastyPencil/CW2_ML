import numpy as np

from cw2_ml.al.modified import select_typiclust_novelty_indices
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


def test_modified_strategy_prioritizes_far_points_from_labeled() -> None:
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

    selected = select_typiclust_novelty_indices(
        embeddings=embeddings,
        labeled_indices={0},
        budget=1,
        novelty_weight=1.0,
        max_clusters=1,
        knn_k=2,
        random_state=3,
    )
    assert selected[0] in {3, 4}
