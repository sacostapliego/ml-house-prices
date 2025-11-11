import pandas as pd
import numpy as np

def load_and_preprocess_atlanta_data(raw_path="../data/raw/ZHVI_Atlanta_Zip.csv"):
    """Load raw Zillow data and convert to long format for Atlanta."""
    df = pd.read_csv(raw_path)
    
    # Detect date columns
    date_cols = [c for c in df.columns if c[:4].isdigit()]
    print(f"Found {len(date_cols)} date columns")
    
    # Filter for Atlanta, GA
    atl = df[
        (df["City"].str.lower() == "atlanta") & 
        (df["State"].str.upper() == "GA")
    ].copy()
    
    # Reshape to long format
    atl_long = atl.melt(
        id_vars=["RegionName", "City", "State", "CountyName"],
        value_vars=date_cols,
        var_name="Date",
        value_name="ZHVI"
    )
    
    # Convert date
    atl_long["Date"] = pd.to_datetime(atl_long["Date"], format="%Y-%m-%d", errors="coerce")
    atl_long.dropna(subset=["Date"], inplace=True)
    atl_long.sort_values(["RegionName", "Date"], inplace=True)
    atl_long.reset_index(drop=True, inplace=True)
    
    print(f"Processed shape: {atl_long.shape}")
    return atl_long

def save_processed_data(df, output_path="../data/processed/atl_long.csv"):
    """Save processed data to CSV."""
    df.to_csv(output_path, index=False)
    print(f"Saved processed data to {output_path}")