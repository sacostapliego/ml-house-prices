import pandas as pd
import numpy as np

def create_time_features(df):
    """Create lag features, rolling averages, and time-based features."""
    df = df.copy()
    
    # Ensure Date is datetime
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    
    # Create lag features (1, 3, 6 months)
    for lag in [1, 3, 6]:
        df[f"ZHVI_lag{lag}"] = df.groupby("RegionName")["ZHVI"].shift(lag)
    
    # Rolling averages (3, 6 months)
    for window in [3, 6]:
        df[f"ZHVI_roll{window}"] = df.groupby("RegionName")["ZHVI"].transform(
            lambda x: x.rolling(window).mean()
        )
    
    # Time-based features
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    
    # Drop rows with missing values
    df.dropna(inplace=True)
    
    print(f"Features created. Final shape: {df.shape}")
    return df

def get_feature_columns():
    """Return list of feature column names."""
    return [
        "ZHVI_lag1", "ZHVI_lag3", "ZHVI_lag6",
        "ZHVI_roll3", "ZHVI_roll6", "Year", "Month"
    ]