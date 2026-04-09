import json
from pathlib import Path
import sys

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.config import UI_REFERENCE_PATH
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


st.set_page_config(page_title="Used Car Pricing", layout="wide")
st.title("Used Car Pricing and Time-to-Sell Predictor")
st.write("Estimate fair price and market sell-speed from real used-car listing details.")

ui = load_ui_reference()

with st.form("car_form"):
    col1, col2 = st.columns(2)
    with col1:
        brand = st.selectbox("Brand", ui["brands"], index=0)
        model = st.selectbox("Model", ui["top_models"], index=0)
        variant = st.text_input("Variant", "VXI")
        year = st.number_input("Year", min_value=ui["year_min"], max_value=ui["year_max"], value=max(ui["year_min"], ui["year_max"] - 5))
        km_driven = st.number_input("KM Driven", min_value=1000, max_value=300000, value=ui["km_default"])
        fuel_type = st.selectbox("Fuel Type", ui["fuel_types"], index=0)
    with col2:
        transmission = st.selectbox("Transmission", ui["transmissions"], index=0)
        owner_type = st.selectbox("Owner Type", ui["owner_types"], index=0)
        seller_type = st.selectbox("Seller Type", ui["seller_types"], index=0)
        city = st.selectbox("City", ui["cities"], index=0)
        description = st.text_area("Listing Description", "well maintained, single owner, service history available")

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
    col1, col2, col3 = st.columns(3)
    col1.metric("Predicted Price", f"INR {result['predicted_price']:,.0f}")
    col2.metric("Sell Within 30 Days", f"{result['sell_within_30_days_probability'] * 100:.1f}%")
    col3.metric("Sale Speed Bucket", result["sale_speed_bucket"].title())

    market = result["market_snapshot"]
    st.subheader("Market Intelligence")
    st.write(f"Price band: **{market['market_price_band'].title()}**")
    if market["peer_median_price"] is not None:
        st.write(f"Peer median price in market: INR {market['peer_median_price']:,.0f}")
        st.write(f"Peer median days listed: {market['peer_median_days_listed']} days")
        st.write(f"Delta vs peers: {market['market_delta_pct']}% across {market['peer_sample_size']} comparable listings")

    st.subheader("Model Explanation")
    st.write(result["decision_explanation"])

    st.subheader("Listing Signals")
    st.json(result["listing_signals"])

    st.subheader("Risk Review")
    st.write(f"Manual review recommended: **{'Yes' if result['manual_review_recommended'] else 'No'}**")
    st.write(result["suspicion_flags"] or ["No strong risk flags detected"])
