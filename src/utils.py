"""Small reusable helpers (I/O + validation) shared by every stage."""
import json
from pathlib import Path

import joblib
import pandas as pd

from src.exception import handle_errors
from src.logging_config import get_logger

logger = get_logger(__name__)


@handle_errors(logger, "reading a CSV file")
def read_csv(path: Path, **kwargs) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_csv(path, **kwargs)
    if df.empty:
        raise ValueError(f"File is empty: {path}")
    logger.info("Loaded %s  shape=%s", path.name, df.shape)
    return df


@handle_errors(logger, "saving a CSV file")
def save_csv(df: pd.DataFrame, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Saved %s  shape=%s", path, df.shape)


@handle_errors(logger, "saving a JSON file")
def save_json(obj: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4, default=str)
    logger.info("Saved %s", path)


@handle_errors(logger, "loading a JSON file")
def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@handle_errors(logger, "saving a model")
def save_model(model, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    logger.info("Model saved to %s", path)


@handle_errors(logger, "loading a model")
def load_model(path: Path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}. Run the train_model stage first.")
    return joblib.load(path)


def validate_columns(df: pd.DataFrame, required: list, name: str) -> None:
    """Raise a clear error if any expected column is missing."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"'{name}' is missing required columns: {missing}")
