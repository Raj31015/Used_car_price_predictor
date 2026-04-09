import json
from pathlib import Path
import sys

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.config import MARKET_REFERENCE_PATH, UI_REFERENCE_PATH
from src.predict import predict_listing


def load_ui_reference() -> dict:
    if UI_REFERENCE_PATH.exists():
        return json.loads(UI_REFERENCE_PATH.read_text())
    return {
        "brands": ["Maruti", "Hyundai", "Honda", "Toyota", "Mahindra"],
        "top_models": ["Swift", "Baleno", "City", "Creta", "Innova"],
        "cities": ["Delhi", "Mumbai", "Bangalore", "Chennai", "Pune", "Hyderabad"],
        "fuel_types": ["Petrol", "Diesel", "CNG"],
        "transmissions": ["Manual", "Automatic"],
        "owner_types": ["First", "Second", "Third"],
        "seller_types": ["Dealer", "Individual"],
        "year_min": 2005,
        "year_max": 2026,
        "km_default": 45000,
    }


def load_market_reference() -> pd.DataFrame:
    if MARKET_REFERENCE_PATH.exists():
        return pd.DataFrame(json.loads(MARKET_REFERENCE_PATH.read_text()))
    return pd.DataFrame()


def speed_badge(speed: str) -> str:
    labels = {
        "fast": "[FAST]",
        "medium": "[MEDIUM]",
        "slow": "[SLOW]",
    }
    return labels.get(speed.lower(), speed.title())


def recommendation_message(result: dict) -> dict:
    market = result["market_snapshot"]
    fair_price = result["predicted_price"]
    peer_price = market["peer_median_price"]
    price_band = market["market_price_band"]

    if peer_price is None:
        return {
            "headline": f"Suggested list price: INR {fair_price:,.0f}",
            "body": "Comparable market listings were not available, so this recommendation is based mainly on the calibrated model output.",
        }

    if price_band == "overpriced":
        target_price = min(fair_price, peer_price * 0.98)
        target_price = max(target_price, fair_price * 0.8)
        reduction_pct = max(0.0, ((fair_price - target_price) / max(fair_price, 1)) * 100)
        return {
            "headline": f"Suggested price: INR {target_price:,.0f}",
            "body": f"Reduce the listing by about {reduction_pct:.1f}% to move closer to local market levels and improve sell speed.",
        }

    if price_band == "underpriced":
        target_price = max(fair_price, peer_price * 0.97)
        uplift_pct = max(0.0, ((target_price - fair_price) / max(fair_price, 1)) * 100)
        return {
            "headline": f"Suggested price: INR {target_price:,.0f}",
            "body": f"You may be leaving value on the table. A price increase of about {uplift_pct:.1f}% should still stay close to market.",
        }

    return {
        "headline": f"Suggested price: INR {fair_price:,.0f}",
        "body": "Your listing is already close to the local market median. The displayed fair price is calibrated to local comparables rather than relying on a single model output.",
    }


def format_signal(label: str) -> str:
    return label.replace("-", " ").capitalize()


st.set_page_config(page_title="Used Car Pricing", layout="wide")
st.title("Used Car Pricing and Time-to-Sell Predictor")
st.write("Estimate a fair listing price, likely sale speed, and practical pricing guidance from real used-car listing details.")

ui = load_ui_reference()
market_reference = load_market_reference()

with st.form("car_form"):
    st.subheader("Vehicle Info")
    col1, col2, col3 = st.columns(3)
    with col1:
        brand = st.selectbox("Brand", ui["brands"], index=0)
        model = st.selectbox("Model", ui["top_models"], index=0)
    with col2:
        variant = st.text_input("Variant", "VXI")
        year = st.number_input(
            "Year",
            min_value=ui["year_min"],
            max_value=ui["year_max"],
            value=max(ui["year_min"], ui["year_max"] - 5),
        )
    with col3:
        km_driven = st.number_input("KM Driven", min_value=1000, max_value=300000, value=ui["km_default"])
        fuel_type = st.selectbox("Fuel Type", ui["fuel_types"], index=0)

    st.subheader("Listing Info")
    col4, col5 = st.columns(2)
    with col4:
        city = st.selectbox("City", ui["cities"], index=0)
        seller_type = st.selectbox("Seller Type", ui["seller_types"], index=0)
        owner_type = st.selectbox("Owner Type", ui["owner_types"], index=0)
    with col5:
        transmission = st.selectbox("Transmission", ui["transmissions"], index=0)
        description = st.text_area(
            "Listing Description",
            "well maintained, single owner, service history available",
            height=120,
        )

    with st.expander("Advanced (optional)"):
        st.caption("Use this section only if you want to simulate unusual or edge-case listings.")
        st.write("The current demo uses the listed fields to drive the pricing and sale-speed recommendation.")

    submitted = st.form_submit_button("Predict")

if submitted:
    result = predict_listing(
        {
            "brand": brand,
            "model": model,
            "variant": variant,
            "year": year,
            "km_driven": km_driven,
            "fuel_type": fuel_type,
            "transmission": transmission,
            "owner_type": owner_type,
            "seller_type": seller_type,
            "city": city,
            "description": description,
        }
    )
    market = result["market_snapshot"]
    recommendation = recommendation_message(result)

    st.markdown("## Suggested Listing Price")
    hero_col, side_col = st.columns([1.7, 1])
    with hero_col:
        st.metric("Fair Price", f"INR {result['predicted_price']:,.0f}")
        st.caption(
            f"Fair market range: INR {result['fair_price_range_low']:,.0f} to INR {result['fair_price_range_high']:,.0f}"
        )
        st.caption(recommendation["body"])
    with side_col:
        st.markdown(f"### {speed_badge(result['sale_speed_bucket'])}")
        speed_progress = {
            "fast": 0.8,
            "medium": 0.5,
            "slow": 0.2,
        }.get(result["sale_speed_bucket"].lower(), 0.5)
        st.progress(speed_progress)
        st.caption("Sale speed is shown as a relative market bucket based on listing characteristics and comparable behavior.")

    st.info(recommendation["headline"])

    st.subheader("Market Snapshot")
    insight_col1, insight_col2 = st.columns(2)
    insight_col1.metric("Market Position", market.get("market_price_label", market["market_price_band"].title()))
    insight_col2.metric(
        "Peer Median Price",
        f"INR {market['peer_median_price']:,.0f}" if market["peer_median_price"] is not None else "Not available",
    )
    if market["market_delta_pct"] is not None:
        st.caption(
            f"This listing is {market['market_delta_pct']}% away from the local peer median across {market['peer_sample_size']} comparable listings."
        )
    if market["peer_median_days_listed"] is not None:
        st.caption(f"Comparable listings take about {market['peer_median_days_listed']:.0f} days to sell in this market.")

    if not market_reference.empty:
        filtered = market_reference[
            (market_reference["brand"] == brand)
            & (market_reference["model"] == model)
            & (market_reference["city"] == city)
        ].copy()
        if not filtered.empty and market["peer_median_price"] is not None:
            price_chart_df = pd.DataFrame(
                {
                    "Label": ["Fair Price", "Peer Median Price"],
                    "Amount": [result["predicted_price"], market["peer_median_price"]],
                }
            )
            days_chart_df = pd.DataFrame(
                {
                    "Label": ["Predicted Sale Window", "Peer Median Days"],
                    "Days": [
                        15 if result["sale_speed_bucket"] == "fast" else 25 if result["sale_speed_bucket"] == "medium" else 40,
                        market["peer_median_days_listed"],
                    ],
                }
            )
            chart_col1, chart_col2 = st.columns(2)
            with chart_col1:
                st.subheader("Price vs Market")
                st.bar_chart(price_chart_df.set_index("Label"))
            with chart_col2:
                st.subheader("Expected Time to Sell")
                st.bar_chart(days_chart_df.set_index("Label"))

    st.subheader("Why The Model Thinks So")
    for reason in result["decision_explanation"]:
        st.write(f"- {reason.capitalize()}")

    positive_signals = result["listing_signals"]["positive_signals"]
    risk_signals = result["listing_signals"]["risk_signals"]
    signal_col1, signal_col2 = st.columns(2)
    with signal_col1:
        st.subheader("Positive Listing Signals")
        if positive_signals:
            for signal in positive_signals:
                st.write(f"- {format_signal(signal)}")
        else:
            st.write("- No strong positive text signals detected")
    with signal_col2:
        st.subheader("Risk Signals")
        if risk_signals:
            for signal in risk_signals:
                st.write(f"- {format_signal(signal)}")
        else:
            st.write("- No obvious risk phrases detected")

    st.subheader("Actionable Review")
    st.write(f"Manual review recommended: **{'Yes' if result['manual_review_recommended'] else 'No'}**")
    if result["suspicion_flags"]:
        for flag in result["suspicion_flags"]:
            st.write(f"- {flag.capitalize()}")
    else:
        st.write("- No strong risk flags detected")
