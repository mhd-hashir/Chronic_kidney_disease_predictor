import pandas as pd
import numpy as np
import joblib
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score

def train_model(data_path="data/processed/train_data.csv", model_dir="models"):
    print("Loading data...")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"{data_path} not found. Run preprocessing first.")
        
    df = pd.read_csv(data_path)
    
    X = df.drop('target', axis=1)
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Training Random Forest on {X_train.shape[0]} samples...")
    # Using Random Forest as it works well with SHAP
    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"Model Accuracy: {acc:.4f}")
    print(f"Model F1 Score: {f1:.4f}")
    print(classification_report(y_test, y_pred))
    
    # Save Model
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "ckd_model.pkl")
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
    
    # Save Metrics
    metrics = {
        "accuracy": acc,
        "f1_score": f1,
        "samples": len(df),
        "timestamp": pd.Timestamp.now().isoformat()
    }
    with open(os.path.join(model_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)
        
    # Save Data Stats for Sensitivity Analysis (Min/Max/Mode)
    # We need this to simulate "What if" scenarios for missing values
    stats = {}
    
    # Numerics
    for col in X.select_dtypes(include=[np.number]).columns:
        stats[col] = {
            "min": float(df[col].min()),
            "max": float(df[col].max()),
            "mean": float(df[col].mean()),
            "median": float(df[col].median())
        }
        
    # Categoricals
    for col in X.select_dtypes(include=['object']).columns:
        # Get unique values and the most frequent one
        unique_vals = df[col].dropna().unique().tolist()
        mode_val = df[col].mode()[0] if not df[col].mode().empty else unique_vals[0]
        stats[col] = {
            "options": unique_vals,
            "mode": mode_val
        }
    
    with open(os.path.join(model_dir, "data_stats.json"), "w") as f:
        json.dump(stats, f, indent=4)
    print(f"Data stats saved to {os.path.join(model_dir, 'data_stats.json')}")
        
    return model, metrics

if __name__ == "__main__":
    train_model()
