import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error, r2_score
import os
from data_utils import load_and_preprocess_atlanta_data, save_processed_data
from features import create_time_features, get_feature_columns

def train_atlanta_model(
    raw_data_path="./data/raw/ZHVI_Atlanta_Zip.csv",
    processed_data_path="./data/processed/atl_long.csv",
    model_save_path="./models/random_forest_atlanta.joblib",
    test_size=0.2,
    n_estimators=200,
    random_state=42
):
    """Complete training pipeline for Atlanta house price prediction."""
    
    print("=" * 50)
    print("STEP 1: Loading and preprocessing data...")
    print("=" * 50)
    
    # Load and preprocess raw data
    df = load_and_preprocess_atlanta_data(raw_data_path)
    save_processed_data(df, processed_data_path)
    
    print("\n" + "=" * 50)
    print("STEP 2: Feature engineering...")
    print("=" * 50)
    
    # Create features
    df = create_time_features(df)
    
    print("\n" + "=" * 50)
    print("STEP 3: Preparing train/test split...")
    print("=" * 50)
    
    # Prepare features and target
    feature_cols = get_feature_columns()
    X = df[feature_cols]
    y = df["ZHVI"]
    
    # Train/test split (no shuffle to preserve time series order)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False
    )
    
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Testing samples: {X_test.shape[0]}")
    
    print("\n" + "=" * 50)
    print("STEP 4: Training Random Forest model...")
    print("=" * 50)
    
    # Train model
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    print("✓ Model trained successfully")
    
    print("\n" + "=" * 50)
    print("STEP 5: Evaluating model performance...")
    print("=" * 50)
    
    # Evaluate
    y_pred = model.predict(X_test)
    rmse = root_mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Root Mean Squared Error: ${rmse:,.2f}")
    print(f"R² Score: {r2:.4f}")
    
    print("\n" + "=" * 50)
    print("STEP 6: Saving model...")
    print("=" * 50)
    
    # Ensure models directory exists
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    
    # Save model
    joblib.dump(model, model_save_path)
    print(f"✓ Model saved to {model_save_path}")
    
    print("\n" + "=" * 50)
    print("TRAINING COMPLETE!")
    print("=" * 50)
    
    return model, rmse, r2

if __name__ == "__main__":
    train_atlanta_model()