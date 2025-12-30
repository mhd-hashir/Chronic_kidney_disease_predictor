import pandas as pd
import joblib
import os
from src.preprocessing import CKDPreprocessor
from src.train import train_model

def build_artifacts():
    print("Building artifacts with correct module paths...")
    
    # 1. Preprocessing
    df = pd.read_csv("data/raw/kidney_disease.csv")
    processor = CKDPreprocessor()
    X_processed, y_processed = processor.fit_transform(df)
    
    # Save processed data
    processed_df = pd.DataFrame(X_processed)
    processed_df['target'] = y_processed
    
    if 'target' in processed_df.columns:
        processed_df = processed_df.dropna(subset=['target'])
        
    processed_df.to_csv("data/processed/train_data.csv", index=False)
    print("Preprocessing Done.")

    # 2. Training
    # Ensure train_model is called from this context if needed, 
    # but train_model usually just saves the RandomForest which is sklearn native (safe).
    # The issue was checking the preprocessor which has custom classes.
    train_model()
    print("Training Done.")

if __name__ == "__main__":
    build_artifacts()
