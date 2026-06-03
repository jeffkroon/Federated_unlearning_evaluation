import pandas as pd


class StandardScalerDF:
    def __init__(self):
        self.means = {}
        self.stds = {}
        self.fitted = False

    def fit(self, df: pd.DataFrame, cols) -> "StandardScalerDF":
        self.means = {c: float(df[c].mean()) for c in cols}
        self.stds = {c: float(df[c].std() or 1.0) for c in cols}  #use 1 if std is 0
        self.fitted = True
        return self

    def transform(self, df: pd.DataFrame, cols) -> pd.DataFrame:
        if not self.fitted:
            raise RuntimeError(" Not fitted")
        out = df.copy()
        for c in cols:
            std = self.stds.get(c, 1.0) or 1.0
            out[c] = (out[c] - self.means.get(c, 0.0)) / std
        return out

    def fit_transform(self, df: pd.DataFrame, cols) -> pd.DataFrame:
        return self.fit(df, cols).transform(df, cols)
