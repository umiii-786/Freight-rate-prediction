"""Stage 6 - predict_model
Uses models/model.pkl to create:
  reports/predictions/validation_predictions.csv                    (load_id, predicted_rate)
  reports/predictions/december_chart_input_with_predicted_rate.csv
"""
import numpy as np
import pandas as pd

from src.config import (DATE_COL, DECEMBER_FILE, ID_COL, MODELS_DIR, PREDICTION_COL,
                        PREDICTIONS_DIR, PROCESSED_DIR, RAW_DIR, TEMPLATE_FILE,
                        VALIDATION_FILE, load_params)
from src.exception import handle_errors
from src.features.preprocessing import get_feature_columns
from src.logging_config import get_logger
from src.utils import load_model, read_csv, save_csv

logger = get_logger(__name__)


@handle_errors(logger, "filling the validation prediction template")
def fill_template(template: pd.DataFrame, val_ids: pd.Series, preds: np.ndarray) -> pd.DataFrame:
    """Align predictions to the template by load_id (falls back to row order)."""
    pred_df = pd.DataFrame({ID_COL: val_ids.values, PREDICTION_COL: preds})
    out = template.drop(columns=[PREDICTION_COL], errors="ignore")

    if set(out[ID_COL]) == set(pred_df[ID_COL]):
        return out.merge(pred_df, on=ID_COL, how="left")

    if len(out) == len(pred_df):
        logger.warning("load_id values differ between template and validation file - "
                       "assigning predictions by row order")
        out = out.copy()
        out[PREDICTION_COL] = preds
        return out
    raise ValueError("Template and validation file do not match (different ids and lengths)")


@handle_errors(logger, "running the predict_model stage")
def main() -> None:
    logger.info("=== predict_model started ===")
    params = load_params()
    feats = params["features"]
    feature_cols = get_feature_columns(feats["numeric"], feats["categorical"])

    model = load_model(MODELS_DIR / "model.pkl")

    # ---- validation (Nov-Dec) -> fill the submission template
    val = read_csv(PROCESSED_DIR / VALIDATION_FILE, parse_dates=[DATE_COL])
    val_pred = model.predict(val[feature_cols])
    template = read_csv(RAW_DIR / TEMPLATE_FILE)
    save_csv(fill_template(template, val[ID_COL], val_pred),
             PREDICTIONS_DIR / "validation_predictions.csv")
    logger.info("Validation predictions: mean=%.2f min=%.2f max=%.2f",
                val_pred.mean(), val_pred.min(), val_pred.max())

    # ---- December chart inputs (same row order as the raw file)
    dec_feat = read_csv(PROCESSED_DIR / DECEMBER_FILE, parse_dates=[DATE_COL])
    dec_raw = read_csv(RAW_DIR / DECEMBER_FILE)
    dec_pred = model.predict(dec_feat[feature_cols])
    dec_raw[PREDICTION_COL] = dec_pred
    save_csv(dec_raw, PREDICTIONS_DIR / "december_chart_input_with_predicted_rate.csv")

    logger.info("=== predict_model completed ===")


if __name__ == "__main__":
    main()
