from pathlib import Path

from cw2_ml.experiments.run_active_learning import run_smoke_experiment


def test_smoke_experiment_writes_metrics(tmp_path: Path) -> None:
    metrics_path = run_smoke_experiment(tmp_path)
    assert metrics_path.exists()
    assert metrics_path.name == "metrics.csv"
