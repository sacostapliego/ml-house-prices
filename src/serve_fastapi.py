from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict
import joblib
import uvicorn
import pandas as pd
from datetime import datetime, timedelta

# --- Load Model ---
model_path = "./models/random_forest_atlanta.joblib"
try:
    model = joblib.load(model_path)
    print("✓ Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# --- Define Input Schema ---
class HouseFeaturesInput(BaseModel):
    """Input features for Atlanta house price prediction."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "ZHVI_lag1": 285000.0,
                "ZHVI_lag3": 283000.0,
                "ZHVI_lag6": 280000.0,
                "ZHVI_roll3": 284000.0,
                "ZHVI_roll6": 282000.0,
                "Year": 2025,
                "Month": 11
            }
        }
    )
    
    ZHVI_lag1: float = Field(..., description="Home value 1 month ago")
    ZHVI_lag3: float = Field(..., description="Home value 3 months ago")
    ZHVI_lag6: float = Field(..., description="Home value 6 months ago")
    ZHVI_roll3: float = Field(..., description="3-month rolling average")
    ZHVI_roll6: float = Field(..., description="6-month rolling average")
    Year: int = Field(..., ge=2000, le=2030, description="Year")
    Month: int = Field(..., ge=1, le=12, description="Month (1-12)")

# --- Initialize FastAPI ---
app = FastAPI(
    title="Atlanta House Price Prediction API",
    description="Predicts home values for Atlanta, GA based on historical trends",
    version="1.0"
)

@app.get("/")
def home():
    """Root endpoint with API information."""
    return {
        "message": "Welcome to the Atlanta House Price Prediction API!",
        "endpoints": {
            "/predict": "POST - Predict house price",
            "/health": "GET - Check API health",
            "/docs": "GET - Interactive API documentation"
        }
    }

@app.get("/health")
def health_check():
    """Check if model is loaded and API is ready."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "model_loaded": True}

@app.post("/predict")
def predict_price(features: HouseFeaturesInput):
    """
    Predict Atlanta house price based on historical trends.
    
    Returns predicted ZHVI (Zillow Home Value Index).
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not available")
    
    try:
        # Convert input to DataFrame
        data = pd.DataFrame([features.model_dump()])
        
        # Make prediction
        prediction = model.predict(data)[0]
        
        return {
            "predicted_price": round(prediction, 2),
            "currency": "USD",
            "location": "Atlanta, GA",
            "note": "Prediction based on Zillow Home Value Index trends"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

# --- Run locally ---
if __name__ == "__main__":
    uvicorn.run("serve_fastapi:app", host="127.0.0.1", port=8000, reload=True)
    
class SimplePredictionInput(BaseModel):
    """Simplified input - just ZIP code."""
    zip_code: int = Field(..., description="Atlanta ZIP code", example=30305)
    months_ahead: int = Field(1, ge=1, le=12, description="Months to predict ahead")

@app.post("/predict/simple")
def predict_simple(input_data: SimplePredictionInput):
    """
    Predict home value for a ZIP code N months ahead.
    Automatically calculates features from historical data.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not available")
    
    try:
        # Load historical data
        df = pd.read_csv("./data/processed/atl_long.csv")
        df["Date"] = pd.to_datetime(df["Date"])
        
        # Filter for the specific ZIP code
        zip_data = df[df["RegionName"] == input_data.zip_code].sort_values("Date")
        
        if len(zip_data) == 0:
            raise HTTPException(status_code=404, detail=f"ZIP code {input_data.zip_code} not found")
        
        # Get most recent data point
        latest = zip_data.iloc[-1]
        
        # Calculate features automatically
        features = {
            "ZHVI_lag1": latest["ZHVI"],
            "ZHVI_lag3": zip_data.iloc[-3]["ZHVI"] if len(zip_data) >= 3 else latest["ZHVI"],
            "ZHVI_lag6": zip_data.iloc[-6]["ZHVI"] if len(zip_data) >= 6 else latest["ZHVI"],
            "ZHVI_roll3": zip_data["ZHVI"].tail(3).mean(),
            "ZHVI_roll6": zip_data["ZHVI"].tail(6).mean(),
            "Year": (latest["Date"] + timedelta(days=30*input_data.months_ahead)).year,
            "Month": (latest["Date"] + timedelta(days=30*input_data.months_ahead)).month
        }
        
        # Make prediction
        X = pd.DataFrame([features])
        prediction = model.predict(X)[0]
        
        return {
            "zip_code": input_data.zip_code,
            "current_value": round(latest["ZHVI"], 2),
            "predicted_value": round(prediction, 2),
            "months_ahead": input_data.months_ahead,
            "prediction_date": (latest["Date"] + timedelta(days=30*input_data.months_ahead)).strftime("%Y-%m-%d"),
            "change": round(prediction - latest["ZHVI"], 2),
            "percent_change": round(((prediction - latest["ZHVI"]) / latest["ZHVI"]) * 100, 2)
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")