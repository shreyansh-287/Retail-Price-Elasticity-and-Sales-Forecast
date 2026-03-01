from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "Model", "retail_weekly_forecast_model.pkl")

model = pickle.load(open(MODEL_PATH, "rb"))

app = FastAPI(title="Retail Weekly Sales Forecast API")

# Define request schema
class SalesInput(BaseModel):
    Store: int
    Promo: int
    CompetitionDistance: float
    StoreType: str
    Assortment: str
    SchoolHoliday: int
    lag_1: float
    lag_4: float
    rolling_mean_4: float


@app.get("/")
def home():
    return {"message": "Retail Weekly Sales Forecast API is running"}


@app.post("/predict")
def predict_sales(input_data: SalesInput):

    # Convert input to DataFrame
    data = pd.DataFrame([input_data.dict()])

    # Predict log sales
    prediction_log = model.predict(data)

    # Convert back to actual sales
    prediction = np.expm1(prediction_log)

    return {
        "predicted_weekly_sales": round(float(prediction[0]), 2)
    }