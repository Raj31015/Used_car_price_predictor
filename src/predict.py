from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd

from .config import CURRENT_YEAR, MARKET_REFERENCE_PATH, PRICE_MODEL_PATH, SALE_MODEL_PATH, SALE_SPEED_MODEL_PATH


POSITIVE_SIGNALS = {
    "single owner": "single-owner history",
    "service history": "documented service history",
    "well maintained": "well-maintained condition",
    "new tires": "recent tire replacement",
    "all papers clear": "clear ownership papers",
    "like new": "like-new condition",
    "showroom maintained": "showroom-maintained condition",
    "mint condition": "mint condition",
    "excellent condition": "excellent condition",
}

RISK_SIGNALS = {
    "urgent sale": "urgent sale wording",
    "accident": "accident history mention",
    "repainted": "repaint disclosure",
    "engine issue": "engine issue mention",
}


def _load_market_reference() -> list[dict]:
    if not MARKET_REFERENCE_PATH.exists():
        return []
    return json.loads(MARKET_REFERENCE_PATH.read_text())


def _market_snapshot(features: dict, predicted_price: float) -> dict:
    rows = _load_market_reference()
    for row in rows:
        if row["brand"] == features["brand"] and row["model"] == features["model"] and row["city"] == features["city"]:
            delta_pct = ((predicted_price - row["median_price"]) / row["median_price"]) * 100 if row["median_price"] else 0
            if delta_pct <= -8:
                band = "underpriced"
            elif delta_pct >= 8:
                band = "overpriced"
            else:
                band = "fair"
            return {
                "market_price_band": band,
                "peer_median_price": round(float(row["median_price"]), 2),
                "peer_median_days_listed": round(float(row["median_days_listed"]), 1),
                "market_delta_pct": round(delta_pct, 2),
                "peer_sample_size": int(row["sample_size"]),
            }
    return {
        "market_price_band": "unknown",
        "market_price_label": "Market Data Unavailable",
        "peer_median_price": None,
        "peer_median_days_listed": None,
        "market_delta_pct": None,
        "peer_sample_size": 0,
    }


def _refresh_market_delta(market: dict, fair_price: float) -> dict:
    updated = dict(market)
    peer_price = updated["peer_median_price"]
    if peer_price is None or not peer_price:
        return updated

    delta_pct = ((fair_price - peer_price) / peer_price) * 100
    if delta_pct <= -8:
        band = "underpriced"
        label = "Below Market"
    elif delta_pct >= 8:
        band = "overpriced"
        label = "Above Market"
    else:
        band = "fair"
        label = "In Line With Market"

    updated["market_delta_pct"] = round(delta_pct, 2)
    updated["market_price_band"] = band
    updated["market_price_label"] = label
    return updated


def _feature_price_boost(features: dict, signals: dict) -> float:
    multiplier = 1.0
    if features["km_driven"] < 10000:
        multiplier *= 1.10
    if features["owner_type"] == "First":
        multiplier *= 1.05
    if features["transmission"] == "Automatic":
        multiplier *= 1.07
    if features["year"] >= 2020:
        multiplier *= 1.08
    if any(signal in signals["positive_signals"] for signal in {"like-new condition", "showroom-maintained condition", "mint condition", "excellent condition"}):
        multiplier *= 1.04
    return multiplier


def _calibrate_price(model_price: float, market: dict) -> dict:
    peer_price = market["peer_median_price"]
    if peer_price is None:
        fair_price = model_price
        lower_bound = fair_price * 0.92
        upper_bound = fair_price * 1.08
        method = "model_only"
    else:
        blended_price = (0.7 * model_price) + (0.3 * peer_price)
        max_allowed = peer_price * 1.15
        fair_price = min(blended_price, max_allowed)
        lower_bound = min(fair_price * 0.94, peer_price * 0.96)
        upper_bound = max(fair_price * 1.06, min(max_allowed, peer_price * 1.08))
        method = "market_calibrated"

    return {
        "raw_model_price": round(model_price, 2),
        "fair_price": round(fair_price, 2),
        "fair_price_range_low": round(lower_bound, 2),
        "fair_price_range_high": round(upper_bound, 2),
        "pricing_method": method,
    }


def _extract_listing_signals(description: str) -> dict:
    text = description.lower()
    positives = [label for token, label in POSITIVE_SIGNALS.items() if token in text]
    risks = [label for token, label in RISK_SIGNALS.items() if token in text]
    return {"positive_signals": positives, "risk_signals": risks}


def _sale_speed_from_market(fair_price: float, market: dict) -> tuple[str, float]:
    peer_price = market["peer_median_price"]
    if peer_price is None or not peer_price:
        return "medium", 0.55

    ratio = fair_price / peer_price
    if ratio <= 0.95:
        return "fast", 0.78
    if ratio <= 1.15:
        return "medium", 0.56
    return "slow", 0.28


def _suspicion_flags(features: dict, market: dict) -> list[str]:
    flags = []
    if features["km_driven"] / max(1, CURRENT_YEAR - features["year"]) > 35000:
        flags.append("abnormally high yearly mileage")
    if market["market_delta_pct"] is not None and abs(market["market_delta_pct"]) >= 18:
        flags.append("price far from local peer median")
    if features["owner_type"] == "Third":
        flags.append("multiple prior owners")
    if features["seller_type"] == "Individual" and features["transmission"] == "Automatic" and features["year"] < 2013:
        flags.append("rare configuration worth manual verification")
    return flags


def _explanation(features: dict, market: dict, signals: dict) -> list[str]:
    reasons = []
    if features["year"] >= 2021:
        reasons.append("newer registration year supports stronger resale value")
    if features["km_driven"] < 50000:
        reasons.append("lower odometer reading improves demand")
    if features["transmission"] == "Automatic":
        reasons.append("automatic transmission lifts price in urban resale markets")
    if features["owner_type"] == "First":
        reasons.append("first-owner history supports stronger resale demand")
    if features["km_driven"] < 10000:
        reasons.append("very low mileage supports a premium over typical listings")
    if market["market_price_band"] == "overpriced":
        reasons.append("listing appears above comparable market median")
    if market["market_price_band"] == "underpriced":
        reasons.append("listing is below comparable market median and may sell faster")
    if signals["positive_signals"]:
        reasons.append(f"text mentions {signals['positive_signals'][0]}")
    return reasons[:4]


def predict_listing(features: dict) -> dict:
    df = pd.DataFrame([features])
    df["car_age"] = CURRENT_YEAR - df["year"]
    df["km_per_year"] = df["km_driven"] / df["car_age"].clip(lower=1)
    df["log_km_driven"] = np.log1p(df["km_driven"].clip(lower=1))
    df["age_km_interaction"] = df["car_age"] * df["km_per_year"]
    df["is_premium"] = (
        df["brand"].fillna("").isin({"Toyota", "Honda"})
        | df["model"].fillna("").isin({"City", "Innova"})
    ).astype(int)

    price_model = joblib.load(PRICE_MODEL_PATH)
    sale_model = joblib.load(SALE_MODEL_PATH)
    sale_speed_model = joblib.load(SALE_SPEED_MODEL_PATH)

    raw_model_price = float(np.exp(price_model.predict(df)[0]))
    signals = _extract_listing_signals(features["description"])
    boosted_model_price = raw_model_price * _feature_price_boost(features, signals)
    market = _market_snapshot(features, raw_model_price)
    calibrated = _calibrate_price(boosted_model_price, market)
    market = _refresh_market_delta(market, calibrated["fair_price"])
    sale_speed_bucket, derived_sale_probability = _sale_speed_from_market(calibrated["fair_price"], market)
    # Keep model calls in place for compatibility with the existing architecture.
    _ = sale_model.predict_proba(df)[0, 1]
    _ = sale_speed_model.predict(df)[0]
    suspicion_flags = _suspicion_flags(features, market)
    return {
        "predicted_price": calibrated["fair_price"],
        "raw_model_price": round(raw_model_price, 2),
        "boosted_model_price": round(boosted_model_price, 2),
        "fair_price_range_low": calibrated["fair_price_range_low"],
        "fair_price_range_high": calibrated["fair_price_range_high"],
        "pricing_method": calibrated["pricing_method"],
        "sell_within_30_days_probability": round(derived_sale_probability, 4),
        "sale_speed_bucket": sale_speed_bucket,
        "market_snapshot": market,
        "listing_signals": signals,
        "suspicion_flags": suspicion_flags,
        "manual_review_recommended": bool(suspicion_flags),
        "decision_explanation": _explanation(features, market, signals),
    }
