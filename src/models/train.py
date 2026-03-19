import lightgbm as lgb
from sklearn.metrics import mean_squared_log_error
import pandas as pd
import numpy as np
import joblib
import json
import mlflow
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]   # adjust if needed
MODELS_DIR = ROOT / "models"

MODELS_DIR.mkdir(exist_ok=True)

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("store-sales-local")



def run_baseline_model(train: pd.DataFrame):
    train = train.copy()

    train_data = train[train["date"] < "2017-01-01"].copy()
    val_data = train[train["date"] >= "2017-01-01"].copy()

    drop_cols = ["sales", "date", "id"]
    X_train = train_data.drop(columns=drop_cols, errors="ignore")
    X_val = val_data.drop(columns=drop_cols, errors="ignore")

    y_train_raw = train_data["sales"]
    y_val_raw = val_data["sales"]

    y_train_log = np.log1p(y_train_raw)
    y_val_log = np.log1p(y_val_raw)

    # numeric safety
    X_train, X_val = X_train.align(X_val, join="left", axis=1, fill_value=0)
    # X_train = X_train.astype(float)
    # X_val = X_val.astype(float)

    with mlflow.start_run():
        model = lgb.LGBMRegressor(
            objective="regression",
            n_estimators=1000,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42
        )
        model.fit(
            X_train,
            y_train_log,
            eval_set=[(X_val, y_val_log)],
            eval_metric="l2",
            callbacks=[
                lgb.log_evaluation(period=50),
                lgb.early_stopping(stopping_rounds=100)
            ]
        )

        y_pred_log = model.predict(X_val)
        y_pred = np.expm1(y_pred_log)
        y_pred = np.clip(y_pred, 0, None)

        rmsle = np.sqrt(mean_squared_log_error(y_val_raw, y_pred))
        print("RMSLE:", rmsle)

        # Save model
        joblib.dump(model, MODELS_DIR / "lightgbm_model.pkl")
        features = list(X_train.columns)

        # Save features column
        joblib.dump(features, MODELS_DIR / "feature_columns.pkl")

        #MLFlow for monitoring
        mlflow.log_param("learning_rate", 0.05)
        mlflow.log_param("n_estimators", 1000)
        mlflow.log_param('num_leaves', 31)
        mlflow.log_metric("rmsle", rmsle)
        mlflow.sklearn.log_model(model, artifact_path="model")
        mlflow.log_artifact( MODELS_DIR / "feature_columns.pkl", artifact_path="features")
    return model, rmsle