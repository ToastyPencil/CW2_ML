from cw2_ml.config import ExperimentConfig


def test_config_defaults_are_valid() -> None:
    cfg = ExperimentConfig()
    assert cfg.dataset == "cifar10"
    assert cfg.query_batch_size > 0
    assert cfg.outputs_dir == "outputs/final_submission"
    assert cfg.device == "auto"
    assert cfg.ssl_checkpoint_path.endswith("outputs/final_submission/ssl/simclr_resnet18.pt")
    assert cfg.ssl_embeddings_path.endswith("outputs/final_submission/ssl/cifar10_embeddings.npy")


def test_config_accepts_ssl_path_overrides() -> None:
    cfg = ExperimentConfig(
        ssl_checkpoint_path="custom/checkpoint.pt",
        ssl_embeddings_path="custom/embeddings.npy",
    )
    assert cfg.ssl_checkpoint_path == "custom/checkpoint.pt"
    assert cfg.ssl_embeddings_path == "custom/embeddings.npy"


def test_config_derives_ssl_defaults_from_outputs_dir() -> None:
    cfg = ExperimentConfig(outputs_dir="outputs/custom_run")
    assert cfg.ssl_checkpoint_path == "outputs/custom_run/ssl/simclr_resnet18.pt"
    assert cfg.ssl_embeddings_path == "outputs/custom_run/ssl/cifar10_embeddings.npy"
