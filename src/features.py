from __future__ import annotations

import pandas as pd

from config import CURRENT_YEAR


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data["car_age"] = CURRENT_YEAR - data["year"]
    data["km_per_year"] = data["km_driven"] / data["car_age"].clip(lower=1)
    data["description"] = data["description"].fillna("").str.lower()
    data["variant"] = data["variant"].fillna("unknown")
    data["sell_within_30_days"] = (data["days_listed"] <= 30).astype(int)
    return data
