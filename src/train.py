from __future__ import annotations

import json
import sys

import joblib
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import (
    MARKET_REFERENCE_PATH,
    METRICS_PATH,
    MODELS_DIR,
    PRICE_MODEL_PATH,
    PROCESSED_DATA_PATH,
    SALE_MODEL_PATH,
    SALE_SPEED_MODEL_PATH,
    UI_REFERENCE_PATH,
)
from data_utils import load_dataset
from features import add_derived_features


class ProgressBar:
    def __init__(self, total_steps: int) -> None:
        self.total_steps = total_steps
        self.current_step = 0

    def update(self, message: str) -> None:
        self.current_step += 1
        self._render(message)

    def _render(self, message: str) -> None:
        width = 24
        completed = int(width * self.current_step / self.total_steps)
        bar = "#" * completed + "-" * (width - completed)
        percent = int(100 * self.current_step / self.total_steps)
        sys.stdout.write(f"\r[{bar}] {percent:>3}% | {message:<40}")
        sys.stdout.flush()
        if self.current_step == self.total_steps:
            sys.stdout.write("\n")


def build_preprocessor() -> ColumnTransformer:
    numeric_features = ["year", "km_driven", "car_age", "km_per_year"]
    categorical_features = ["brand", "model", "variant", "fuel_type", "transmission", "owner_type", "seller_type", "city"]
    text_features = "description"

    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
            ("txt", TfidfVectorizer(max_features=60, ngram_range=(1, 1)), text_features),
        ]
    )


def train() -> None:
    progress = ProgressBar(total_steps=9)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    progress.update("Loading dataset")
    df = add_derived_features(load_dataset())
    progress.update("Engineering derived features")
    df["sale_speed_bucket"] = df["days_listed"].apply(
        lambda value: "fast" if value <= 15 else "medium" if value <= 30 else "slow"
    )
    df.to_csv(PROCESSED_DATA_PATH, index=False)
    progress.update("Saving processed dataset")

    feature_cols = [
        "brand",
        "model",
        "variant",
        "year",
        "km_driven",
        "fuel_type",
        "transmission",
        "owner_type",
        "seller_type",
        "city",
        "description",
        "car_age",
        "km_per_year",
    ]
    X = df[feature_cols]
    y_price = df["price"]
    y_sale = df["sell_within_30_days"]
    y_speed = df["sale_speed_bucket"]

    X_train, X_test, y_price_train, y_price_test, y_sale_train, y_sale_test, y_speed_train, y_speed_test = train_test_split(
        X, y_price, y_sale, y_speed, test_size=0.2, random_state=42
    )
    progress.update("Preparing train/test split")

    price_model = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", Ridge(alpha=1.5)),
        ]
    )
    sale_model = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", LogisticRegression(max_iter=600, class_weight="balanced")),
        ]
    )
    sale_speed_model = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", LogisticRegression(max_iter=600, class_weight="balanced")),
        ]
    )

    progress.update("Training price model")
    price_model.fit(X_train, y_price_train)
    progress.update("Training sale classifier")
    sale_model.fit(X_train, y_sale_train)
    progress.update("Training speed classifier")
    sale_speed_model.fit(X_train, y_speed_train)

    price_predictions = price_model.predict(X_test)
    sale_predictions = sale_model.predict(X_test)
    sale_probabilities = sale_model.predict_proba(X_test)[:, 1]
    speed_predictions = sale_speed_model.predict(X_test)

    market_reference = (
        df.groupby(["brand", "model", "city"])
        .agg(
            median_price=("price", "median"),
            median_days_listed=("days_listed", "median"),
            median_km_driven=("km_driven", "median"),
            sample_size=("price", "size"),
        )
        .reset_index()
        .to_dict(orient="records")
    )

    metrics = {
        "price_mae": float(mean_absolute_error(y_price_test, price_predictions)),
        "sale_accuracy": float(accuracy_score(y_sale_test, sale_predictions)),
        "sale_roc_auc": float(roc_auc_score(y_sale_test, sale_probabilities)),
        "sale_speed_macro_f1": float(f1_score(y_speed_test, speed_predictions, average="macro")),
        "rows_used": int(len(df)),
    }
    progress.update("Evaluating models")
    ui_reference = {
        "brands": sorted(df["brand"].dropna().unique().tolist())[:80],
        "top_models": sorted(df["model"].dropna().value_counts().head(150).index.tolist()),
        "cities": sorted(df["city"].dropna().value_counts().head(60).index.tolist()),
        "fuel_types": sorted(df["fuel_type"].dropna().unique().tolist()),
        "transmissions": sorted(df["transmission"].dropna().unique().tolist()),
        "owner_types": sorted(df["owner_type"].dropna().unique().tolist()),
        "seller_types": sorted(df["seller_type"].dropna().unique().tolist()),
        "year_min": int(df["year"].min()),
        "year_max": int(df["year"].max()),
        "km_default": int(df["km_driven"].median()),
    }

    joblib.dump(price_model, PRICE_MODEL_PATH)
    joblib.dump(sale_model, SALE_MODEL_PATH)
    joblib.dump(sale_speed_model, SALE_SPEED_MODEL_PATH)
    MARKET_REFERENCE_PATH.write_text(json.dumps(market_reference, indent=2))
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    UI_REFERENCE_PATH.write_text(json.dumps(ui_reference, indent=2))
    progress.update("Saving model artifacts")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    train()
