from __future__ import annotations

import numpy as np
import pandas as pd

from config import CURRENT_YEAR


PREMIUM_BRANDS = {"Toyota", "Honda"}
PREMIUM_MODELS = {"City", "Innova"}


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data["car_age"] = CURRENT_YEAR - data["year"]
    data["km_per_year"] = data["km_driven"] / data["car_age"].clip(lower=1)
    data["log_km_driven"] = np.log1p(data["km_driven"].clip(lower=1))
    data["age_km_interaction"] = data["car_age"] * data["km_per_year"]
    data["is_premium"] = (
        data["brand"].fillna("").isin(PREMIUM_BRANDS)
        | data["model"].fillna("").isin(PREMIUM_MODELS)
    ).astype(int)
    data["description"] = data["description"].fillna("").str.lower()
    data["variant"] = data["variant"].fillna("unknown")
    data["sell_within_30_days"] = (data["days_listed"] <= 30).astype(int)
    return data
