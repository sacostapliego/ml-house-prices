import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
from features import basic_preprocess

def train_main(csv_path="data/raw/train.csv", save_model="models/rf_model.joblib"):
    df = pd.read_csv(csv_path)
    X, y, preproc = basic_preprocess(df)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    rmse = mean_squared_error(y_val, preds, squared=False)
    print("Validation RMSE:", rmse)
    # Save pipeline (wrap preprocessing + model)
    full = {"preproc": preproc, "model": model}
    joblib.dump(full, save_model)
    print("Saved model to", save_model)

if __name__ == "__main__":
    train_main()
