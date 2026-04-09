from datetime import datetime
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "used_cars.csv"
CARDEKHO_DATA_PATH = DATA_DIR / "raw" / "cars_data_clean.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed" / "used_cars_processed.csv"
MODELS_DIR = BASE_DIR / "models"
PRICE_MODEL_PATH = MODELS_DIR / "price_model.joblib"
SALE_MODEL_PATH = MODELS_DIR / "sale_classifier.joblib"
SALE_SPEED_MODEL_PATH = MODELS_DIR / "sale_speed_classifier.joblib"
MARKET_REFERENCE_PATH = MODELS_DIR / "market_reference.json"
METRICS_PATH = MODELS_DIR / "metrics.json"
UI_REFERENCE_PATH = MODELS_DIR / "ui_reference.json"
CURRENT_YEAR = datetime.now().year
