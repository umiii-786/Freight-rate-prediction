"""Stage 5 - train_model
Fits the model chosen in params.yaml (train.model_name) on ALL training rows,
saves it to models/model.pkl and logs params/metrics/model to MLflow.
"""
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

from src.config import MODELS_DIR, PROCESSED_DIR, REPORTS_DIR, TRAIN_FILE, load_params
from src.exception import handle_errors
from src.logging_config import get_logger
from src.model.cross_validation import regression_metrics
from src.model.model_factory import build_pipeline, ensure_positive_target
from src.tracking import setup_mlflow
from src.utils import read_csv, save_json, save_model

logger = get_logger(__name__)


@handle_errors(logger, "running the train_model stage")
def main() -> None:
    logger.info("=== train_model started ===")
    params = load_params()
    target = params["base"]["target"]
    name = params["train"]["model_name"]
    model_params = params["models"][name]
    feats = params["features"]
    log_target = params["cross_validation"]["log_target"]

    df = read_csv(PROCESSED_DIR / TRAIN_FILE, parse_dates=["date"])
    X, y = df.drop(columns=[target]), df[target]
    if log_target:
        ensure_positive_target(y)

    model = build_pipeline(name, model_params, feats["numeric"], feats["categorical"], log_target)
    logger.info("Training %s on %d rows ...", name, len(X))
    model.fit(X, y)
    train_pred = model.predict(X)
    train_metrics = regression_metrics(y, train_pred, "train_")
    logger.info("Train metrics: %s", {k: round(v, 4) for k, v in train_metrics.items()})

    save_model(model, MODELS_DIR / "model.pkl")

    setup_mlflow()
    with mlflow.start_run(run_name=f"train-{name}") as run:
        mlflow.set_tags({"stage": "train", "model_name": name})
        mlflow.log_params({f"model.{k}": v for k, v in model_params.items()})
        mlflow.log_param("log_target", log_target)
        mlflow.log_metrics({k.replace("%", "pct"): v for k, v in train_metrics.items()})
        signature = infer_signature(X.head(100), train_pred[:100])
        # cloudpickle is needed because the pipeline contains a custom transformer
        # (MarketIndexImputer); MLflow 3.x defaults to skops, which rejects it.
        info = mlflow.sklearn.log_model(
            model, name="model", signature=signature,
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
        )
        run_info = {"run_id": run.info.run_id, "model_name": name,
                    "model_uri": info.model_uri, "model_id": getattr(info, "model_id", None)}

    save_json(run_info, REPORTS_DIR / "train_run_info.json")
    logger.info("=== train_model completed (run_id=%s) ===", run_info["run_id"])


if __name__ == "__main__":
    main()
