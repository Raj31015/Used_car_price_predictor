# Used Car Pricing and Time-to-Sell Prediction

## Problem

Used car listings are messy. Sellers use inconsistent model names, descriptions contain useful hidden signals, and price expectations are often unrealistic. This project predicts:

- expected resale price
- probability that the car will sell within 30 days
- sale-speed bucket: fast, medium, slow
- market price band: underpriced, fair, overpriced
- suspicious listing flags for manual review

## Expected Dataset Schema

Place a CSV in `data/raw/used_cars.csv` with columns similar to:

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

If the file does not exist, the training script generates a realistic synthetic dataset so the pipeline stays runnable.

## Run

```powershell
python -m src.train
streamlit run app\streamlit_app.py
```

## Resume Bullet

Built an end-to-end ML system to estimate used-car resale price and 30-day sale probability from noisy listing data using feature engineering, text processing, regression, and classification.

## 2026-Ready Upgrades

- market intelligence layer using brand-model-city peer medians
- price banding instead of only raw price prediction
- sale-speed classification for decision support
- text signal extraction from messy listing descriptions
- suspicious listing flags for manual review workflows
