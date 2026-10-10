"""MLflow experiment-tracking helper.

Tracking server priority:
  1. MLFLOW_TRACKING_URI environment variable (e.g. DagsHub)   <- recommended for remote
  2. mlflow.tracking_uri in params.yaml (default: local SQLite)
Credentials (MLFLOW_TRACKING_USERNAME / MLFLOW_TRACKING_PASSWORD) are read
from the environment by MLflow itself - never hard-code them in the repo.
"""
import os

import mlflow

from src.config import load_params
from src.exception import handle_errors
from src.logging_config import get_logger

logger = get_logger(__name__)


@handle_errors(logger, "configuring MLflow")
def setup_mlflow() -> None:
    cfg = load_params()["mlflow"]
    if not os.getenv("MLFLOW_TRACKING_URI"):
        mlflow.set_tracking_uri(cfg["tracking_uri"])
    mlflow.set_experiment(cfg["experiment_name"])
    logger.info("MLflow tracking URI: %s | experiment: %s",
                mlflow.get_tracking_uri(), cfg["experiment_name"])
