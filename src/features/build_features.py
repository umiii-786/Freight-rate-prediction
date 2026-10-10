"""Stage 3 - build_features
  * adds the `month` feature
  * builds the December chart inputs (lat/lon from city lookup, market_index
    from the daily average of train+validation - both are INPUT columns, so no
    target leakage)
  * keeps only the model columns (drops pickup, delivery, quote_signal -> the
    latter leaks the answer)
"""
import pandas as pd

from src.config import (DATE_COL, DECEMBER_FILE, ID_COL, INTERIM_DIR, MARKET_INDEX_COL,
                        MONTH_COL, PROCESSED_DIR, TRAIN_FILE, VALIDATION_FILE, load_params)
from src.exception import handle_errors
from src.features.preprocessing import get_feature_columns
from src.logging_config import get_logger
from src.utils import read_csv, save_csv

logger = get_logger(__name__)


@handle_errors(logger, "adding the month feature")
def add_month(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[MONTH_COL] = pd.to_datetime(df[DATE_COL]).dt.month
    return df


@handle_errors(logger, "building the city -> coordinates lookup")
def build_city_lookups(train: pd.DataFrame, val: pd.DataFrame) -> tuple:
    allrows = pd.concat([train, val], ignore_index=True)
    pickup = allrows.groupby("pickup")[["pickup_lat", "pickup_lon"]].first()
    delivery = allrows.groupby("delivery")[["delivery_lat", "delivery_lon"]].first()
    logger.info("Lookup built: %d pickup cities, %d delivery cities", len(pickup), len(delivery))
    return pickup, delivery


@handle_errors(logger, "building the daily market_index lookup")
def build_daily_market_index(train: pd.DataFrame, val: pd.DataFrame) -> pd.Series:
    both = pd.concat([train[[DATE_COL, MARKET_INDEX_COL]], val[[DATE_COL, MARKET_INDEX_COL]]])
    return both.groupby(DATE_COL)[MARKET_INDEX_COL].mean()


@handle_errors(logger, "building the December features")
def build_december_features(dec, pickup_lookup, delivery_lookup, daily_mi) -> pd.DataFrame:
    dec = dec.copy()

    unknown_p = set(dec["pickup"]) - set(pickup_lookup.index)
    unknown_d = set(dec["delivery"]) - set(delivery_lookup.index)
    if unknown_p or unknown_d:
        raise ValueError(f"December cities not found in training/validation data: "
                         f"pickup={unknown_p}, delivery={unknown_d}")

    for col in ["pickup_lat", "pickup_lon"]:
        dec[col] = dec["pickup"].map(pickup_lookup[col])
    for col in ["delivery_lat", "delivery_lon"]:
        dec[col] = dec["delivery"].map(delivery_lookup[col])

    dec[MARKET_INDEX_COL] = dec[DATE_COL].map(daily_mi)
    n_missing = int(dec[MARKET_INDEX_COL].isna().sum())
    if n_missing:
        logger.warning("%d December rows have no market_index (model will use the global average)",
                       n_missing)
    return dec


@handle_errors(logger, "running the build_features stage")
def main() -> None:
    logger.info("=== build_features started ===")
    params = load_params()
    target = params["base"]["target"]
    feature_cols = get_feature_columns(params["features"]["numeric"],
                                       params["features"]["categorical"])

    train = add_month(read_csv(INTERIM_DIR / TRAIN_FILE, parse_dates=[DATE_COL]))
    val = add_month(read_csv(INTERIM_DIR / VALIDATION_FILE, parse_dates=[DATE_COL]))
    dec = read_csv(INTERIM_DIR / DECEMBER_FILE, parse_dates=[DATE_COL])

    # December needs lookups built from the *full* train + validation rows
    pickup_lu, delivery_lu = build_city_lookups(train, val)
    daily_mi = build_daily_market_index(train, val)
    dec = add_month(build_december_features(dec, pickup_lu, delivery_lu, daily_mi))

    save_csv(train[feature_cols + [target]], PROCESSED_DIR / TRAIN_FILE)
    save_csv(val[[ID_COL] + feature_cols], PROCESSED_DIR / VALIDATION_FILE)
    save_csv(dec[feature_cols], PROCESSED_DIR / DECEMBER_FILE)

    logger.info("=== build_features completed ===")


if __name__ == "__main__":
    main()
