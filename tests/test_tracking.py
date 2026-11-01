import mlflow

from fraud.tracking import log_run, start_tracking


def test_run_is_logged_with_params_metrics_and_files(tmp_path):
    db = tmp_path / "mlflow.db"
    note = tmp_path / "note.txt"
    note.write_text("hello")

    start_tracking(db_path=db, artifact_dir=tmp_path / "mlruns", experiment="test-exp")
    log_run("my run", params={"model": "lightgbm"}, metrics={"val_pr_auc": 0.4}, files=[note])

    runs = mlflow.search_runs(experiment_names=["test-exp"])
    assert len(runs) == 1
    run = runs.iloc[0]
    assert run["tags.mlflow.runName"] == "my run"
    assert run["params.model"] == "lightgbm"
    assert run["metrics.val_pr_auc"] == 0.4

    files = [f.path for f in mlflow.MlflowClient().list_artifacts(run["run_id"])]
    assert files == ["note.txt"]


def test_start_tracking_twice_reuses_the_experiment(tmp_path):
    db = tmp_path / "mlflow.db"
    start_tracking(db_path=db, artifact_dir=tmp_path / "mlruns", experiment="same")
    start_tracking(db_path=db, artifact_dir=tmp_path / "mlruns", experiment="same")
    assert mlflow.get_experiment_by_name("same") is not None
