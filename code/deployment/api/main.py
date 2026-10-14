"""
Stage 3: Model API (FastAPI)
Loads the trained model from model.pkl and returns predictions.
"""
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

MODEL_PATH = Path(__file__).parent / "model.pkl"

# The order of the columns must be the same as during training
FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]

app = FastAPI(title="Diabetes Prediction API")

# load the model once, when the API starts
model = joblib.load(MODEL_PATH)


class PatientData(BaseModel):
    """Input data of one patient (validated automatically by FastAPI)"""
    Pregnancies: float = Field(ge=0)
    Glucose: float = Field(gt=0)
    BloodPressure: float = Field(gt=0)
    SkinThickness: float = Field(gt=0)
    Insulin: float = Field(gt=0)
    BMI: float = Field(gt=0)
    DiabetesPedigreeFunction: float = Field(ge=0)
    Age: float = Field(gt=0)


@app.get("/health")
def health():
    """Used to check that the API is running"""
    return {"status": "ok"}


@app.post("/predict")
def predict(data: PatientData):
    # one row DataFrame with the same columns as in training
    row = pd.DataFrame([data.model_dump()])[FEATURES]
    prediction = int(model.predict(row)[0])
    probability = float(model.predict_proba(row)[0][1])
    return {"prediction": prediction, "probability": probability}
