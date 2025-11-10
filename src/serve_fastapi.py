# src/serve_fastapi.py
from fastapi import FastAPI
import joblib
import numpy as np
from pydantic import BaseModel

class HouseFeatures(BaseModel):
    OverallQual:int; GrLivArea:float; YearBuilt:int; TotalBsmtSF:float
    FullBath:int; GarageCars:int; LotArea:float; Neighborhood:str

app = FastAPI()
pipe = joblib.load("models/rf_model.joblib")  # dict with preproc & model

@app.post("/predict")
def predict(h: HouseFeatures):
    x = [[h.OverallQual,h.GrLivArea,h.YearBuilt,h.TotalBsmtSF,h.FullBath,h.GarageCars,h.LotArea,h.Neighborhood]]
    X_proc = pipe['preproc'].transform(x)
    pred = pipe['model'].predict(X_proc)[0]
    return {"predicted_price": float(pred)}
