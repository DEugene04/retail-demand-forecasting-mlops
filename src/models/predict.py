import joblib
import pandas as pd
from src.data.load_data import load_dataset
from pipelines.build_features import build_inference_features
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]

model_path = ROOT / "models" / "lightgbm_model.pkl"
feature_path = ROOT / "models" / "feature_columns.pkl"

def load_artifacts():
    model = joblib.load(model_path)
    feature_columns = joblib.load(feature_path)
    return model, feature_columns

def predict(input_df: pd.DataFrame):
    model, feature_columns = load_artifacts()
    holiday, oil, train, test, stores, transactions = load_dataset()

    X = build_inference_features(
        input_df=input_df,
        oil=oil,
        stores=stores,
        holiday=holiday,
        feature_columns=feature_columns
    )

    predictions = model.predict(X)
    return predictions