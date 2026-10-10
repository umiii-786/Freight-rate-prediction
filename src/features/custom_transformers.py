"""Custom sklearn transformers.

IMPORTANT: keep these classes in an importable module (not inside a script that
runs as __main__) - otherwise the pickled model cannot be loaded later.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class MarketIndexImputer(BaseEstimator, TransformerMixin):
    """Fill missing market_index with the average of that calendar date.

    Input columns (in this order): [date, market_index]
    Falls back to the global average for dates not seen during fit.
    """

    def _prepare(self, X) -> pd.DataFrame:
        X = pd.DataFrame(X).copy()
        X.columns = ["date", "market_index"]
        X["date"] = pd.to_datetime(X["date"])
        return X

    def fit(self, X, y=None):
        X = self._prepare(X)
        self.daily_avg_ = X.groupby("date")["market_index"].mean()
        self.global_avg_ = X["market_index"].mean()
        return self

    def transform(self, X):
        X = self._prepare(X)
        missing = X["market_index"].isna()
        X.loc[missing, "market_index"] = X.loc[missing, "date"].map(self.daily_avg_)
        X["market_index"] = X["market_index"].fillna(self.global_avg_)
        return X[["market_index"]].to_numpy()

    def get_feature_names_out(self, input_features=None):
        return np.array(["market_index"])
