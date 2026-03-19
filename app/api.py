from fastapi import FastAPI
import joblib
from pydantic import BaseModel
from src.models.predict import predict
import pandas as pd
from fastapi import HTTPException

app = FastAPI()

class PredictionRequest(BaseModel):
    date: str
    store_nbr: int
    family: str
    onpromotion: int

train_family = set(joblib.load("models/valid_families.pkl"))

@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/predict")
def make_prediction(request: PredictionRequest):
    try:
        input_df = pd.DataFrame([request.model_dump()])
        if request.family not in train_family:
            raise HTTPException(
            status_code=400,
            detail=f"Invalid family. Must be one of {list(train_family)[:5]}..."
        )
        prediction = predict(input_df)
        return {"prediction": float(prediction[0])}    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))