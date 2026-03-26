from dataclasses import dataclass


@dataclass(slots=True)
class ExperimentConfig:
    dataset: str = "cifar10"
    data_dir: str = "data"
    outputs_dir: str = "outputs/final_submission"
    seed: int = 42

    init_labeled_size: int = 0
    query_batch_size: int = 10
    al_rounds: int = 5

    train_epochs: int = 40
    batch_size: int = 128
    learning_rate: float = 0.03
    weight_decay: float = 5e-4

    embedding_model: str = "resnet18"
    embedding_batch_size: int = 256
    max_clusters: int = 500
    knn_k: int = 20
    min_cluster_size: int = 5

    strategy: str = "typiclust"
    novelty_weight: float = 0.2
    repeats: int = 3
    device: str = "auto"
    smoke: bool = False
    ssl_checkpoint_path: str | None = None
    ssl_embeddings_path: str | None = None

    def __post_init__(self) -> None:
        base_output = self.outputs_dir.rstrip("/\\")
        if self.ssl_checkpoint_path is None:
            self.ssl_checkpoint_path = f"{base_output}/ssl/simclr_resnet18.pt"
        if self.ssl_embeddings_path is None:
            self.ssl_embeddings_path = f"{base_output}/ssl/cifar10_embeddings.npy"
