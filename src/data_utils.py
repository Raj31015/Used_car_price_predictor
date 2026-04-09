from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import ast
import random

import numpy as np
import pandas as pd

from config import CARDEKHO_DATA_PATH, CURRENT_YEAR, RAW_DATA_PATH


BRANDS = {
    "Maruti": ["Swift", "Baleno", "WagonR", "Dzire"],
    "Hyundai": ["i20", "Creta", "Verna", "Grand i10"],
    "Honda": ["City", "Amaze", "Jazz", "WR-V"],
    "Toyota": ["Innova", "Glanza", "Etios", "Corolla"],
    "Mahindra": ["XUV300", "Bolero", "Scorpio", "Thar"],
}

FUEL_TYPES = ["Petrol", "Diesel", "CNG"]
TRANSMISSIONS = ["Manual", "Automatic"]
OWNER_TYPES = ["First", "Second", "Third"]
SELLER_TYPES = ["Dealer", "Individual", "Trustmark Dealer"]
CITIES = ["Delhi", "Mumbai", "Bangalore", "Chennai", "Pune", "Hyderabad"]
DESCRIPTION_SNIPPETS = [
    "single owner, well maintained, service history available",
    "minor scratches, insurance active, city driven",
    "new tires, recently serviced, smooth engine",
    "family car, excellent mileage, urgent sale",
    "showroom condition, all papers clear, highway driven",
]


@dataclass
class CarPriceRule:
    base_price: int
    depreciation_per_year: int
    km_penalty_per_10k: int


PRICE_RULES = {
    "Maruti": CarPriceRule(650000, 45000, 8000),
    "Hyundai": CarPriceRule(750000, 50000, 9000),
    "Honda": CarPriceRule(850000, 55000, 10000),
    "Toyota": CarPriceRule(1100000, 60000, 9000),
    "Mahindra": CarPriceRule(980000, 52000, 8500),
}


def generate_synthetic_dataset(n_samples: int = 900, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []

    for _ in range(n_samples):
        brand = random.choice(list(BRANDS))
        model = random.choice(BRANDS[brand])
        rule = PRICE_RULES[brand]
        year = int(rng.integers(2010, 2026))
        km_driven = int(abs(rng.normal(65000, 30000)))
        fuel_type = random.choice(FUEL_TYPES)
        transmission = random.choice(TRANSMISSIONS)
        owner_type = random.choices(OWNER_TYPES, weights=[0.65, 0.27, 0.08])[0]
        seller_type = random.choices(SELLER_TYPES, weights=[0.25, 0.6, 0.15])[0]
        city = random.choice(CITIES)
        description = random.choice(DESCRIPTION_SNIPPETS)

        age = CURRENT_YEAR - year
        price = (
            rule.base_price
            - age * rule.depreciation_per_year
            - (km_driven / 10000) * rule.km_penalty_per_10k
            + (20000 if transmission == "Automatic" else 0)
            + (12000 if seller_type == "Dealer" else -8000)
            + rng.normal(0, 35000)
        )
        price = max(120000, int(price))

        days_listed = (
            20
            + age * 1.8
            + (km_driven / 25000)
            + (12 if price > 800000 else 0)
            + (-6 if seller_type == "Dealer" else 2)
            + rng.normal(0, 5)
        )
        days_listed = max(3, int(days_listed))

        rows.append(
            {
                "brand": brand,
                "model": model,
                "variant": f"{model} {random.choice(['LX', 'VX', 'ZX', 'Sport'])}",
                "year": year,
                "km_driven": km_driven,
                "fuel_type": fuel_type,
                "transmission": transmission,
                "owner_type": owner_type,
                "seller_type": seller_type,
                "city": city,
                "description": description,
                "price": price,
                "days_listed": days_listed,
            }
        )

    df = pd.DataFrame(rows)
    noise_idx = df.sample(frac=0.08, random_state=seed).index
    df.loc[noise_idx, "description"] = df.loc[noise_idx, "description"].str.upper()
    missing_idx = df.sample(frac=0.04, random_state=seed + 1).index
    df.loc[missing_idx, "variant"] = np.nan
    return df


def _clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.lower() in {"nan", "none"}:
        return ""
    return text


def _title_case(value: object) -> str:
    text = _clean_text(value)
    return text.title() if text else ""


def _parse_feature_list(value: object) -> list[str]:
    text = _clean_text(value)
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return [text.lower()]
    if not isinstance(parsed, list):
        return [text.lower()]
    return [str(item).strip().lower() for item in parsed if str(item).strip()]


def _build_description(df: pd.DataFrame) -> pd.Series:
    feature_columns = [
        "top_features",
        "comfort_features",
        "interior_features",
        "exterior_features",
        "safety_features",
    ]

    def row_to_description(row: pd.Series) -> str:
        parts: list[str] = []
        headline = _clean_text(row.get("dvn"))
        color = _clean_text(row.get("Color"))
        if headline:
            parts.append(headline.lower())
        if color:
            parts.append(f"color {color.lower()}")
        for column in feature_columns:
            features = _parse_feature_list(row.get(column))
            if features:
                parts.extend(features[:6])
        unique_parts = list(dict.fromkeys(part for part in parts if part))
        return ", ".join(unique_parts)

    return df.apply(row_to_description, axis=1)


def _derive_days_listed(data: pd.DataFrame) -> pd.Series:
    peer_median = data.groupby(["brand", "model"])["price"].transform("median").replace(0, np.nan)
    market_ratio = (data["price"] / peer_median).fillna(1.0)
    feature_count = data["description"].str.count(",").fillna(0) + 1
    seller_bonus = data["seller_type"].map({"Dealer": -4, "Individual": 2}).fillna(0)
    owner_penalty = data["owner_type"].map(
        {"First": 0, "Second": 4, "Third": 8, "Fourth": 10, "Fifth": 12, "Unregistered Car": 3}
    ).fillna(6)

    score = (
        18
        + data["car_age"] * 1.2
        + (data["km_per_year"] / 7000).clip(0, 15)
        + ((market_ratio - 1.0) * 35).clip(-10, 25)
        + owner_penalty
        + seller_bonus
        - feature_count.clip(1, 12) * 0.9
    )
    return score.round().clip(5, 90).astype(int)


def _load_cardekho_dataset(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    renamed = pd.DataFrame(
        {
            "brand": raw["oem"].map(_title_case),
            "model": raw["model"].map(_title_case),
            "variant": raw["variant"].map(_title_case),
            "year": pd.to_numeric(raw["myear"], errors="coerce"),
            "km_driven": pd.to_numeric(raw["km"], errors="coerce"),
            "fuel_type": raw["fuel"].map(_title_case),
            "transmission": raw["transmission"].map(_title_case),
            "owner_type": raw["owner_type"].map(_title_case),
            "seller_type": raw["utype"].map(_title_case),
            "city": raw["City"].map(_title_case),
            "description": _build_description(raw),
            "price": pd.to_numeric(raw["listed_price"], errors="coerce"),
        }
    )
    renamed = renamed.dropna(subset=["brand", "model", "year", "km_driven", "fuel_type", "transmission", "city", "price"])
    renamed["variant"] = renamed["variant"].replace("", np.nan).fillna("Unknown")
    renamed["seller_type"] = renamed["seller_type"].replace("", "Unknown")
    renamed["owner_type"] = renamed["owner_type"].replace("", "Unknown")
    renamed = renamed[
        renamed["year"].between(1995, CURRENT_YEAR)
        & renamed["km_driven"].between(0, 500000)
        & renamed["price"].between(50000, 10000000)
    ].copy()
    renamed["car_age"] = CURRENT_YEAR - renamed["year"]
    renamed["km_per_year"] = renamed["km_driven"] / renamed["car_age"].clip(lower=1)
    renamed["days_listed"] = _derive_days_listed(renamed)
    return renamed[
        [
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
            "price",
            "days_listed",
        ]
    ].reset_index(drop=True)


def load_dataset(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    if CARDEKHO_DATA_PATH.exists():
        return _load_cardekho_dataset(CARDEKHO_DATA_PATH)
    return generate_synthetic_dataset()
