"""Expanding-window time-based cross-validation (train on past, test on future)."""
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (mean_absolute_error, mean_absolute_percentage_error,
                             mean_squared_error, r2_score)

from src.exception import handle_errors
from src.logging_config import get_logger

logger = get_logger(__name__)


def regression_metrics(y_true, y_pred, prefix: str) -> dict:
    """MAE / RMSE / R2 / MAPE(%) with a given prefix (e.g. 'test_')."""
    mask = np.asarray(y_true) != 0
    mape = (mean_absolute_percentage_error(np.asarray(y_true)[mask], np.asarray(y_pred)[mask]) * 100
            if mask.any() else np.nan)
    return {
        f"{prefix}MAE": mean_absolute_error(y_true, y_pred),
        f"{prefix}RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        f"{prefix}R2": r2_score(y_true, y_pred),
        f"{prefix}MAPE_%": mape,
    }


@handle_errors(logger, "running time-based cross-validation")
def time_based_cv(df: pd.DataFrame, estimator, target: str, date_col: str = "date",
                  n_folds: int = 3, test_months: int = 2):
    data = df.copy()
    data[date_col] = pd.to_datetime(data[date_col])
    data = data.sort_values(date_col).reset_index(drop=True)
    feature_cols = [c for c in data.columns if c != target]
    end_of_data = data[date_col].max() + pd.Timedelta(days=1)

    rows = []
    for i in range(n_folds):
        test_end = end_of_data - pd.DateOffset(months=(n_folds - 1 - i) * test_months)
        test_start = test_end - pd.DateOffset(months=test_months)

        train = data[data[date_col] < test_start]
        test = data[(data[date_col] >= test_start) & (data[date_col] < test_end)]
        if train.empty or test.empty:
            raise ValueError(f"Fold {i + 1} has an empty train/test set. "
                             "Reduce n_folds/test_months or check your dates.")

        model = clone(estimator)
        model.fit(train[feature_cols], train[target])

        train_pred = model.predict(train[feature_cols])
        test_pred = model.predict(test[feature_cols])

        row = {
            "fold": i + 1,
            "test_period": f"{test_start.date()} to {(test_end - pd.Timedelta(days=1)).date()}",
            "n_train": len(train),
            "n_test": len(test),
            "train_MAE": mean_absolute_error(train[target], train_pred),
        }
        row.update(regression_metrics(test[target], test_pred, "test_"))
        rows.append(row)
        logger.info("Fold %d | test %s | MAE=%.2f R2=%.4f", i + 1, row["test_period"],
                    row["test_MAE"], row["test_R2"])

    results = pd.DataFrame(rows)
    metric_cols = ["train_MAE", "test_MAE", "test_RMSE", "test_R2", "test_MAPE_%"]
    return results, results[metric_cols].mean()
