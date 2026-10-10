"""Builds full model pipelines (preprocessor + estimator [+ log target])."""
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from src.exception import handle_errors
from src.features.preprocessing import build_preprocessor
from src.logging_config import get_logger

logger = get_logger(__name__)

MODEL_REGISTRY = {
    "linear_regression": LinearRegression,
    "random_forest": RandomForestRegressor,
    "gradient_boosting": GradientBoostingRegressor,
}


@handle_errors(logger, "building a model pipeline")
def build_pipeline(model_name: str, model_params: dict, numeric: list,
                   categorical: list, log_target: bool):
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{model_name}'. Choose from {list(MODEL_REGISTRY)}")

    estimator = MODEL_REGISTRY[model_name](**(model_params or {}))
    pipe = Pipeline([
        ("preprocessor", build_preprocessor(numeric, categorical)),
        ("model", estimator),
    ])
    if log_target:
        pipe = TransformedTargetRegressor(regressor=pipe, func=np.log, inverse_func=np.exp)
    return pipe


def ensure_positive_target(y: pd.Series) -> None:
    """log(target) is only valid for strictly positive values."""
    if (y <= 0).any():
        raise ValueError("log_target=true requires all target values to be > 0")
