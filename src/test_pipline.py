"""
Complete test script for the Atlanta house price prediction pipeline.
"""
import requests
import time

def test_training():
    """Test the training pipeline."""
    print("\n" + "="*60)
    print("TESTING TRAINING PIPELINE")
    print("="*60)
    
    from src.train import train_atlanta_model
    model, rmse, r2 = train_atlanta_model()
    
    print(f"\n✓ Training completed successfully!")
    print(f"  RMSE: ${rmse:,.2f}")
    print(f"  R²: {r2:.4f}")
    return True

def test_evaluation():
    """Test the evaluation pipeline."""
    print("\n" + "="*60)
    print("TESTING EVALUATION PIPELINE")
    print("="*60)
    
    from src.evaluate import evaluate_model
    results = evaluate_model()
    
    print(f"\n✓ Evaluation completed successfully!")
    return True

def test_api():
    """Test the FastAPI server."""
    print("\n" + "="*60)
    print("TESTING API (make sure server is running)")
    print("="*60)
    
    base_url = "http://127.0.0.1:8000"
    
    # Test root endpoint
    try:
        response = requests.get(f"{base_url}/")
        print(f"✓ Root endpoint: {response.status_code}")
        print(f"  {response.json()}")
    except requests.exceptions.ConnectionError:
        print("✗ Server not running. Start with: python src/serve_fastapi.py")
        return False
    
    # Test health check
    response = requests.get(f"{base_url}/health")
    print(f"✓ Health check: {response.status_code}")
    print(f"  {response.json()}")
    
    # Test prediction
    test_data = {
        "ZHVI_lag1": 285000.0,
        "ZHVI_lag3": 283000.0,
        "ZHVI_lag6": 280000.0,
        "ZHVI_roll3": 284000.0,
        "ZHVI_roll6": 282000.0,
        "Year": 2025,
        "Month": 11
    }
    
    response = requests.post(f"{base_url}/predict", json=test_data)
    print(f"✓ Prediction endpoint: {response.status_code}")
    print(f"  {response.json()}")
    
    return True

if __name__ == "__main__":
    print("\n" + "="*60)
    print("ATLANTA HOUSE PRICE PREDICTION - FULL PIPELINE TEST")
    print("="*60)
    
    # Test 1: Training
    test_training()
    
    # Test 2: Evaluation
    time.sleep(1)
    test_evaluation()
    
    # Test 3: API (optional - requires server to be running)
    print("\n" + "="*60)
    print("API TEST (Optional)")
    print("="*60)
    print("To test API:")
    print("1. Open a new terminal")
    print("2. Run: python src/serve_fastapi.py")
    print("3. Then run this test again or visit http://127.0.0.1:8000/docs")
    
    input("\nPress Enter to test API (or Ctrl+C to skip)...")
    test_api()
    
    print("\n" + "="*60)
    print("ALL TESTS COMPLETED!")
    print("="*60)