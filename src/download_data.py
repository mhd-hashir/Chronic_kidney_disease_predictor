import pandas as pd
import os

def download_data():
    url = "https://raw.githubusercontent.com/ArjunAnilPillai/Chronic-Kidney-Disease-dataset/master/kidney_disease.csv"
    output_path = "data/raw/kidney_disease.csv"
    
    print(f"Downloading dataset from {url}...")
    try:
        df = pd.read_csv(url)
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        df.to_csv(output_path, index=False)
        print(f"Dataset saved to {output_path}")
        print(f"Shape: {df.shape}")
        print(df.head())
    except Exception as e:
        print(f"Error downloading data: {e}")

if __name__ == "__main__":
    download_data()
