"""Stage 4 - evaluate_models
Runs time-based CV for every model in `cross_validation.models_to_compare`,
logs one MLflow run per model, and writes:
  reports/cv_fold_results.csv        per-fold metrics for all models
  reports/model_comparison.csv       average metrics per model
  reports/metrics.json               averages of the model chosen in train.model_name (dvc metrics)
  reports/figures/model_comparison.png
"""
import matplotlib

matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
import mlflow
import pandas as pd

from src.config import FIGURES_DIR, PROCESSED_DIR, REPORTS_DIR, TRAIN_FILE, load_params
from src.exception import handle_errors
from src.logging_config import get_logger
from src.model.cross_validation import time_based_cv
from src.model.model_factory import build_pipeline, ensure_positive_target
from src.tracking import setup_mlflow
from src.utils import read_csv, save_csv, save_json

logger = get_logger(__name__)


@handle_errors(logger, "plotting the model comparison")
def plot_comparison(model_results: pd.DataFrame) -> None:
    metrics = ["test_MAE", "test_RMSE", "test_R2", "test_MAPE_%"]
    ax = model_results.loc[metrics].T.plot(kind="bar", figsize=(12, 6), width=0.8)
    ax.set_title("Model Comparison (time-based CV averages)")
    ax.set_xlabel("Models")
    ax.set_ylabel("Metric value")
    ax.set_xticklabels(model_results.columns, rotation=0)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / "model_comparison.png"
    plt.savefig(out, dpi=120)
    plt.close()
    logger.info("Saved %s", out)


@handle_errors(logger, "running the evaluate_models stage")
def main() -> None:
    logger.info("=== evaluate_models started ===")
    params = load_params()
    target = params["base"]["target"]
    cv = params["cross_validation"]
    feats = params["features"]

    df = read_csv(PROCESSED_DIR / TRAIN_FILE, parse_dates=["date"])
    if cv["log_target"]:
        ensure_positive_target(df[target])

    setup_mlflow()
    averages, all_folds = {}, []

    for name in cv["models_to_compare"]:
        logger.info("---- CV for model: %s ----", name)
        model_params = params["models"].get(name, {})
        pipe = build_pipeline(name, model_params, feats["numeric"], feats["categorical"],
                              cv["log_target"])
        results, avg = time_based_cv(df, pipe, target=target, n_folds=cv["n_folds"],
                                     test_months=cv["test_months"])

        with mlflow.start_run(run_name=f"cv-{name}"):
            mlflow.set_tags({"stage": "cross_validation", "model_name": name})
            mlflow.log_params({f"model.{k}": v for k, v in model_params.items()})
            mlflow.log_params({"log_target": cv["log_target"], "n_folds": cv["n_folds"],
                               "test_months": cv["test_months"]})
            for _, row in results.iterrows():                      # per-fold metrics
                step = int(row["fold"])
                for m in ["train_MAE", "test_MAE", "test_RMSE", "test_R2", "test_MAPE_%"]:
                    mlflow.log_metric(f"fold_{m}".replace("%", "pct"), row[m], step=step)
            mlflow.log_metrics({f"avg_{k}".replace("%", "pct"): v for k, v in avg.items()})

        averages[name] = avg
        all_folds.append(results.assign(model=name))

    model_results = pd.DataFrame(averages)
    logger.info("\n%s", model_results.round(4).to_string())

    save_csv(pd.concat(all_folds, ignore_index=True), REPORTS_DIR / "cv_fold_results.csv")
    save_csv(model_results.T.reset_index().rename(columns={"index": "model"}),
             REPORTS_DIR / "model_comparison.csv")
    chosen = params["train"]["model_name"]
    save_json({"model": chosen, **averages[chosen].to_dict()}, REPORTS_DIR / "metrics.json")
    plot_comparison(model_results)

    logger.info("=== evaluate_models completed ===")


if __name__ == "__main__":
    main()
