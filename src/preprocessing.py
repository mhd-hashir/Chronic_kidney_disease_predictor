import pandas as pd
import numpy as np
import os
import joblib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

class DataCleaner(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        
        # Drop ID if exists
        if 'id' in X.columns:
            X = X.drop('id', axis=1)
            
        # Clean text columns (remove tabs, whitespace)
        object_cols = X.select_dtypes(include=['object']).columns
        for col in object_cols:
            X[col] = X[col].str.replace('\t', '').str.replace(' ', '')
            X[col] = X[col].replace('?', np.nan)
            
        # Convert specific columns to numeric
        num_cols_err = ['pcv', 'wc', 'rc'] # These often come as objects in UCI dataset
        for col in num_cols_err:
            if col in X.columns:
                X[col] = pd.to_numeric(X[col], errors='coerce')
                
        return X

class CKDPreprocessor:
    def __init__(self, model_dir="models"):
        self.model_dir = model_dir
        self.pipeline = None
        self.target_encoder = {'ckd': 1, 'notckd': 0}
        
    def fit_transform(self, df):
        # Separate features and target
        # Prepare Target Variable
        target_col = 'class' if 'class' in df.columns else 'classification'
        
        X = df.drop(target_col, axis=1, errors='ignore')
        
        if target_col in df.columns:
            # Clean target column
            y_raw = df[target_col].astype(str).str.replace('\t', '').str.replace(' ', '')
            # Map to 0/1, unknowns become NaN
            y = y_raw.map(self.target_encoder)
        else:
            y = None

        # Identify columns
        # We need to run the cleaner first to infer correct types, BUT 
        # sklearn pipelines are static. So we run cleaner manually once to get columns
        cleaner = DataCleaner()
        X_temp = cleaner.transform(X)
        
        numeric_features = X_temp.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = X_temp.select_dtypes(include=['object']).columns.tolist()
        
        # Define Pipeline
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])
            
        self.pipeline = Pipeline(steps=[
            ('cleaner', DataCleaner()),
            ('preprocessor', preprocessor)
        ])
        
        X_processed = self.pipeline.fit_transform(X)
        
        # Save the pipeline
        os.makedirs(self.model_dir, exist_ok=True)
        joblib.dump(self.pipeline, os.path.join(self.model_dir, "preprocessor.pkl"))
        print(f"Preprocessor saved to {os.path.join(self.model_dir, 'preprocessor.pkl')}")

        return X_processed, y.values if y is not None else None

    def transform(self, df):
        if self.pipeline is None:
            # Try loading
            path = os.path.join(self.model_dir, "preprocessor.pkl")
            if os.path.exists(path):
                self.pipeline = joblib.load(path)
            else:
                raise ValueError("Pipeline not fitted yet!")
                
        target_col = 'class' if 'class' in df.columns else 'classification'
        X = df.drop(target_col, axis=1, errors='ignore')
        return self.pipeline.transform(X)

if __name__ == "__main__":
    # Test run
    df = pd.read_csv("data/raw/kidney_disease.csv")
    processor = CKDPreprocessor()
    X_processed, y_processed = processor.fit_transform(df)
    
    # Save processed data for verification
    processed_df = pd.DataFrame(X_processed)
    processed_df['target'] = y_processed
    
    # Drop rows where target is NaN (critical for training)
    if 'target' in processed_df.columns:
        initial_len = len(processed_df)
        processed_df = processed_df.dropna(subset=['target'])
        print(f"Dropped {initial_len - len(processed_df)} rows with missing target.")

    processed_df.to_csv("data/processed/train_data.csv", index=False)
    print("Preprocessing complete. Data shape:", X_processed.shape)
