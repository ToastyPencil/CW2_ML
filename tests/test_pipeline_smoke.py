from pathlib import Path

import numpy as np

from cw2_ml.experiments.run_active_learning import (
    ALL_STRATEGIES,
    ActiveLearningConfig,
    SUPPORTED_STRATEGIES,
    _query_by_strategy,
    _load_bundle,
    _load_or_train_embeddings,
    _parse_args,
    run_experiments,
    run_smoke_experiment,
)


def test_smoke_experiment_writes_metrics(tmp_path: Path) -> None:
    metrics_path = run_smoke_experiment(tmp_path)
    assert metrics_path.exists()
    assert metrics_path.name == "metrics.csv"
    assert (tmp_path / "summary.csv").exists()


def test_load_or_train_embeddings_smoke_writes_cache(tmp_path: Path) -> None:
    ssl_dir = tmp_path / "ssl"
    cfg = ActiveLearningConfig(
        strategy="typiclust",
        output_dir=str(tmp_path),
        embedding_batch_size=16,
        smoke=True,
        download=False,
        fake_train_size=24,
        fake_test_size=12,
        device="cpu",
        num_workers=0,
        ssl_checkpoint_path=str(ssl_dir / "simclr.pt"),
        ssl_embeddings_path=str(ssl_dir / "embeddings.npy"),
    )
    bundle = _load_bundle(cfg)

    embeddings = _load_or_train_embeddings(cfg, bundle, device="cpu", num_workers=0)

    assert embeddings.shape == (len(bundle.train_eval), 512)
    assert (ssl_dir / "embeddings.npy").exists()


def test_load_or_train_embeddings_smoke_reuses_cached_embeddings(tmp_path: Path, monkeypatch) -> None:
    ssl_dir = tmp_path / "ssl"
    cfg = ActiveLearningConfig(
        strategy="typiclust",
        output_dir=str(tmp_path),
        embedding_batch_size=16,
        smoke=True,
        download=False,
        fake_train_size=24,
        fake_test_size=12,
        device="cpu",
        num_workers=0,
        ssl_checkpoint_path=str(ssl_dir / "simclr.pt"),
        ssl_embeddings_path=str(ssl_dir / "embeddings.npy"),
    )
    bundle = _load_bundle(cfg)

    cached = _load_or_train_embeddings(cfg, bundle, device="cpu", num_workers=0)

    def _unexpected_recompute(*args, **kwargs):
        raise AssertionError("expected cached smoke embeddings to be reused")

    monkeypatch.setattr(
        "cw2_ml.experiments.run_active_learning.extract_normalized_embeddings",
        _unexpected_recompute,
    )

    reused = _load_or_train_embeddings(cfg, bundle, device="cpu", num_workers=0)

    assert np.array_equal(reused, cached)


def test_load_or_train_embeddings_smoke_recomputes_invalid_cache_shape(tmp_path: Path) -> None:
    ssl_dir = tmp_path / "ssl"
    cfg = ActiveLearningConfig(
        strategy="typiclust",
        output_dir=str(tmp_path),
        embedding_batch_size=16,
        smoke=True,
        download=False,
        fake_train_size=24,
        fake_test_size=12,
        device="cpu",
        num_workers=0,
        ssl_checkpoint_path=str(ssl_dir / "simclr.pt"),
        ssl_embeddings_path=str(ssl_dir / "embeddings.npy"),
    )
    bundle = _load_bundle(cfg)
    ssl_dir.mkdir(parents=True, exist_ok=True)
    np.save(ssl_dir / "embeddings.npy", np.zeros((5, 512), dtype=np.float32))

    embeddings = _load_or_train_embeddings(cfg, bundle, device="cpu", num_workers=0)

    assert embeddings.shape == (len(bundle.train_eval), 512)


def test_parse_args_defaults_to_canonical_output_dir(monkeypatch) -> None:
    monkeypatch.setattr("sys.argv", ["run_active_learning"])
    args = _parse_args()
    assert args.output_dir == "outputs/final_submission"


def test_parse_args_accepts_ssl_override_paths(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_active_learning",
            "--ssl-checkpoint-path",
            "custom/checkpoint.pt",
            "--ssl-embeddings-path",
            "custom/embeddings.npy",
        ],
    )
    args = _parse_args()
    assert args.ssl_checkpoint_path == "custom/checkpoint.pt"
    assert args.ssl_embeddings_path == "custom/embeddings.npy"


def test_all_strategies_include_typiclust_adaptive() -> None:
    assert "typiclust_adaptive" in ALL_STRATEGIES


def test_supported_strategies_exclude_fixed_novelty_variant() -> None:
    assert "typiclust_novelty" not in SUPPORTED_STRATEGIES


def test_run_experiments_rejects_removed_fixed_novelty_strategy() -> None:
    cfg = ActiveLearningConfig(repeats=0, smoke=True, download=False)
    try:
        run_experiments(config=cfg, strategies=["typiclust_novelty"])
    except ValueError as exc:
        assert "Unsupported strategy" in str(exc)
    else:
        raise AssertionError("expected typiclust_novelty to be rejected")


def test_query_by_strategy_uses_one_based_round_numbers_for_adaptive(monkeypatch) -> None:
    captured: dict[str, int] = {}

    def _fake_select(**kwargs):
        captured["round_idx"] = kwargs["round_idx"]
        return [2]

    monkeypatch.setattr(
        "cw2_ml.experiments.run_active_learning.select_typiclust_adaptive_indices",
        _fake_select,
    )

    selected = _query_by_strategy(
        strategy="typiclust_adaptive",
        embeddings=np.array([[0.0], [1.0], [2.0]], dtype=np.float32),
        labeled_indices={0},
        unlabeled_indices=[1, 2],
        budget=1,
        model=None,
        dataset=None,
        config=ActiveLearningConfig(rounds=5),
        current_round=2,
        device="cpu",
        num_workers=0,
        rng=np.random.default_rng(0),
    )

    assert selected == [2]
    assert captured["round_idx"] == 2


def test_active_learning_config_defaults_match_final_hybrid_modification() -> None:
    cfg = ActiveLearningConfig()

    assert cfg.novelty_weight == 0.15
    assert cfg.novelty_start_round == 3
    assert cfg.uncertainty_weight == 0.20
    assert cfg.uncertainty_start_round == 4


def test_parse_args_defaults_match_final_hybrid_modification(monkeypatch) -> None:
    monkeypatch.setattr("sys.argv", ["run_active_learning"])
    args = _parse_args()

    assert args.novelty_weight == 0.15
    assert args.novelty_start_round == 3
    assert args.uncertainty_weight == 0.20
    assert args.uncertainty_start_round == 4
