# Store Sales MLOps

This project is an end-to-end MLOps pipeline with the purpose of retail demand forecasting. This project covers the full ML Lifecycle such as data processing, feature engineering, experiment tracking, reproducible pipelines, and containerized deployment.

---

## Key Learnings

- End to end ML lifecyle design
- Separation of training and inference pipelines
- Handling feature consistency across different enviroments
- Tracking datasets using DVC instead of git
- Utilizing MLflow to keep track of machine learning experiments
- Containerization for reproducible deployment

---

## Overview

This project includes

- Feature engineering (Calendar, holidays, store, oil data)
- Experiment tracking with MLflow
- Data versioning with DVC
- FastAPI prediction service
- Dockerized deployment for reproducibility

---

## Architecture

Dataset -> Data Versioning (DVC) -> Feature Engineering -> Training Pipeline -> Experiment Tracking (MLflow) -> Model Artifact -> FastAPI Service -> Docker -> Monitoring

---

## Key Features Details

### 1. Feature Engineering Pipeline

- Calendar features (day, month, year)
- Holiday/ event signals
- Oil prices signals

---

### 2. Model training

- LightGBM Regression Model
- Log-transformed target ('log1p')
- RMSLE evaluation metric

---

### 3. Experiment tracking (MLflow)

- Tracks hyperparameters, evaluation metrics, trained models

---

### 4. Data Versioning (DVC)

- Tracks datasets in data/ folder

---

### 5. Inference API (FastAPI)

- Endpoint for predictions
- Input validations
- Example request
  {
  'date': "2017-01-01",
  'store_nbr': 1,
  'family': "BEAUTY",
  'onpromotion': 0
  }

---

## Setup

1. Clone repo
   - git clone 'https://github.com/DEugene04/retail-demand-forecasting-mlops.git'
   - cd store-sales-mlops
2. Install dependencies
   - pip install -r requirements.txt
3. Train model
   - python pipelines/ train_pipeline.py
4. Run MLflow UI
   - mlflow ui \
     --backend-store-uri sqlite:///mlflow.db \
     --default-artifact-root ./mlruns'
5. Open link to MLflow
   - http://127.0.0.1:5000
6. Run API
   - uvicorn app.main:app --reload
7. Open link to API
   - http://127.0.0.1:8000/docs
8. Run with Docker
   - docker build -t store-sales-api .
   - docker run -p 8000:8000 store-sales-api

---

## Limitations

- Lag/ rolling features are not included in the current pipeline which limits forecasting capability and accuracy

---

## Future Improvements

- Add multi-step forecasting strategy (recursive / direct)
- Implement input validation for categorical domains
- Deploy to cloud (AWS / Render)

## Author

Dave Eugene Wijaya
