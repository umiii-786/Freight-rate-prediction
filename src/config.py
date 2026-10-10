"""Central place for paths, constants and params.yaml loading."""
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PARAMS_PATH = PROJECT_ROOT / "params.yaml"

# ---- folders -------------------------------------------------------------
EXTERNAL_DIR = PROJECT_ROOT / "data" / "external"      # original files (dvc add)
RAW_DIR = PROJECT_ROOT / "data" / "raw"                # after ingestion
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"        # after cleaning
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"    # model-ready
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
PREDICTIONS_DIR = REPORTS_DIR / "predictions"
LOGS_DIR = PROJECT_ROOT / "logs"

# ---- standard file names used between stages -----------------------------
TRAIN_FILE = "train_test.csv"
VALIDATION_FILE = "validation.csv"
TEMPLATE_FILE = "validation_predictions_template.csv"
DECEMBER_FILE = "december_inputs.csv"

# ---- columns with special handling ---------------------------------------
DATE_COL = "date"
MARKET_INDEX_COL = "market_index"
MONTH_COL = "month"
ID_COL = "load_id"
PREDICTION_COL = "predicted_rate"


def load_params(path: Path = PARAMS_PATH) -> dict:
    """Read params.yaml and return it as a dict."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
