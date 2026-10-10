"""Builds the sklearn ColumnTransformer used inside every model pipeline."""
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import DATE_COL, MARKET_INDEX_COL, MONTH_COL
from src.features.custom_transformers import MarketIndexImputer


def get_feature_columns(numeric: list, categorical: list) -> list:
    """Every column the model pipeline expects as input."""
    return [DATE_COL] + numeric + categorical + [MARKET_INDEX_COL, MONTH_COL]


def build_preprocessor(numeric: list, categorical: list) -> ColumnTransformer:
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric),
            ("cat", categorical_pipeline, categorical),
            ("market", MarketIndexImputer(), [DATE_COL, MARKET_INDEX_COL]),
            ("month", "passthrough", [MONTH_COL]),
        ],
        remainder="drop",
    )
