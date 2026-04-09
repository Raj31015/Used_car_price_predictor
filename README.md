# Used Car Market Intelligence and Pricing Recommendation System

A machine learning project for estimating fair used-car listing prices, comparing listings against the local market, and giving basic pricing guidance.

## Overview

Used-car listings are messy. Sellers write inconsistent descriptions, similar cars are priced very differently, and listed prices do not always reflect fair market value. The goal of this project was to build a practical pricing assistant that uses both ML and market-comparable logic instead of relying only on a single raw model prediction.

The system takes common listing fields such as brand, model, year, km driven, transmission, owner type, city, and description text, then produces:

- a fair price estimate
- a fair market range
- a market position label
- a sale-speed bucket
- pricing guidance
- listing risk signals

## Dataset

The project was built on `37,729` used-car listings.

Main fields used:
- `brand`
- `model`
- `variant`
- `year`
- `km_driven`
- `fuel_type`
- `transmission`
- `owner_type`
- `seller_type`
- `city`
- `description`
- `price`
- `days_listed`

## Problem

A plain regression model on listing price is usually not enough for this kind of data. Asking prices are noisy, sellers often overprice cars, and premium or low-km cars behave differently from the rest of the market.

To make the output more usable, I combined ML predictions with market-comparable calibration and rule-based pricing logic.

## Approach

### Data preparation
- removed extreme price outliers using IQR filtering
- cleaned text and categorical fields
- created derived features such as:
  - `car_age`
  - `km_per_year`
  - `log_km_driven`
  - `age_km_interaction`
  - `is_premium`

### Models
- `XGBoostRegressor` for price estimation
- calibrated `LogisticRegression` for sale classification
- `LogisticRegression` for sale-speed classification

### Training changes
- used log transform on price target to reduce skew
- added sample weighting so newer, low-km, and premium cars have more influence during training
- evaluated model quality across important market segments instead of looking only at overall MAE

### Final pricing logic
The final displayed price is not just the raw model output. It is adjusted using peer-market median prices so the result behaves more like a pricing recommendation system than a black-box predictor.

## Results

Current metrics:
- Price MAE: `~129.7K`
- 30-day sale classification ROC-AUC: `~0.946`
- Sale-speed Macro-F1: `~0.735`

I also tracked segment-wise MAE for:
- low-km cars
- high-km cars
- newer cars
- older cars

## App

The Streamlit app shows:
- fair price
- fair market range
- peer median comparison
- market position
- sale-speed bucket
- pricing recommendation
- positive listing signals
- risk signals

## Tech stack

- Python
- pandas
- NumPy
- scikit-learn
- XGBoost
- Streamlit
- joblib

## Run locally

```bash
pip install -r requirements.txt
python -m src.train
streamlit run app/streamlit_app.py
