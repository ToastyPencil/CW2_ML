from cw2_ml.config import ExperimentConfig


def test_config_defaults_are_valid() -> None:
    cfg = ExperimentConfig()
    assert cfg.dataset == "cifar10"
    assert cfg.query_batch_size > 0
