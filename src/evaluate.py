import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import root_mean_squared_error, r2_score, mean_absolute_error
from features import create_time_features, get_feature_columns

def evaluate_model(
    processed_data_path="./data/processed/atl_long.csv",
    model_path="./models/random_forest_atlanta.joblib",
    test_size=0.2,
    plot_samples=100
):
    """Evaluate trained model and visualize results."""
    
    print("Loading data and model...")
    df = pd.read_csv(processed_data_path)
    df = create_time_features(df)
    model = joblib.load(model_path)
    
    # Prepare features
    feature_cols = get_feature_columns()
    X = df[feature_cols]
    y = df["ZHVI"]
    
    # Use same split logic as training (no shuffle)
    split_idx = int(len(X) * (1 - test_size))
    X_test = X.iloc[split_idx:]
    y_test = y.iloc[split_idx:]
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    rmse = root_mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print("\n" + "=" * 50)
    print("MODEL EVALUATION RESULTS")
    print("=" * 50)
    print(f"Root Mean Squared Error: ${rmse:,.2f}")
    print(f"Mean Absolute Error: ${mae:,.2f}")
    print(f"R² Score: {r2:.4f}")
    print(f"Test samples: {len(y_test)}")
    
    # Plot predictions vs actuals
    plt.figure(figsize=(12, 6))
    plt.plot(y_test.values[:plot_samples], label="Actual", linewidth=2, marker='o')
    plt.plot(y_pred[:plot_samples], label="Predicted", linestyle="--", linewidth=2, marker='x')
    plt.title("Predicted vs Actual Home Values (Sample)", fontsize=14)
    plt.xlabel("Sample Index", fontsize=12)
    plt.ylabel("ZHVI (Home Value Index)", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("./models/evaluation_plot.png", dpi=150)
    print(f"\n✓ Plot saved to ./models/evaluation_plot.png")
    plt.show()
    
    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "predictions": y_pred,
        "actuals": y_test.values
    }

if __name__ == "__main__":
    evaluate_model()