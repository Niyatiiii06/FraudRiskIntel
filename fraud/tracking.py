"""Log runs to MLflow (local and free).

Runs go into mlflow.db and files into mlruns/, both in the project folder.
To browse them:  mlflow ui --backend-store-uri sqlite:///mlflow.db
"""
import mlflow

from fraud.config import PROJECT_ROOT

TRACKING_DB = PROJECT_ROOT / "mlflow.db"
ARTIFACT_DIR = PROJECT_ROOT / "mlruns"
EXPERIMENT = "fraud-risk-intel"


def start_tracking(db_path=TRACKING_DB, artifact_dir=ARTIFACT_DIR, experiment=EXPERIMENT):
    mlflow.set_tracking_uri("sqlite:///" + db_path.as_posix())
    if mlflow.get_experiment_by_name(experiment) is None:
        mlflow.create_experiment(experiment, artifact_location=artifact_dir.as_uri())
    mlflow.set_experiment(experiment)


def log_run(run_name, params, metrics, files=()):
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        for path in files:
            mlflow.log_artifact(str(path))
