from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import os
import shap
import json
from src.explainability import CKDExplainer

app = FastAPI(title="CKD Prediction & Explainability API")

# Load Model & Preprocessor
MODEL_PATH = "models/ckd_model.pkl"
PREPROCESSOR_PATH = "models/preprocessor.pkl"

from typing import Optional

class PatientData(BaseModel):
    age: Optional[float] = None
    bp: Optional[float] = None
    sg: Optional[float] = None
    al: Optional[float] = None
    su: Optional[float] = None
    rbc: Optional[str] = None
    pc: Optional[str] = None
    pcc: Optional[str] = None
    ba: Optional[str] = None
    bgr: Optional[float] = None
    bu: Optional[float] = None
    sc: Optional[float] = None
    sod: Optional[float] = None
    pot: Optional[float] = None
    hemo: Optional[float] = None
    pcv: Optional[float] = None
    wc: Optional[float] = None
    rc: Optional[float] = None
    htn: Optional[str] = None
    dm: Optional[str] = None
    cad: Optional[str] = None
    appet: Optional[str] = None
    pe: Optional[str] = None
    ane: Optional[str] = None

class FeedbackData(BaseModel):
    # Feedback requires full data usually, but for now allow sparse
    age: Optional[float] = None
    bp: Optional[float] = None 
    # ... simplifying for brevity, in real app duplicate PatientData fields or inherit
    correct_diagnosis: str

@app.on_event("startup")
def load_artifacts():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(PREPROCESSOR_PATH):
        print("Model or Preprocessor not found. Waiting for training pipeline.")
        return

    global model, preprocessor, data_stats
    
    # HACK: Fix for Pickle AttributeError when loading DataCleaner found in __main__
    import src.preprocessing
    import sys
    if not hasattr(sys.modules['__main__'], 'DataCleaner'):
        sys.modules['__main__'].DataCleaner = src.preprocessing.DataCleaner
        
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    
    # Load Stats
    STATS_PATH = "models/data_stats.json"
    if os.path.exists(STATS_PATH):
        with open(STATS_PATH, "r") as f:
            data_stats = json.load(f)
    else:
        data_stats = {}

def check_sensitivity(model, preprocessor, df, missing_cols):
    """
    Checks if missing columns are critical by testing Min/Max/Mode values.
    Returns a list of warnings.
    """
    warnings = []
    if not missing_cols or not data_stats:
        return warnings
        
    # Base prediction (using imputation handling in pipeline)
    base_pred = model.predict(preprocessor.transform(df))[0]
    
    for col in missing_cols:
        if col not in data_stats: continue
        
        stat = data_stats[col]
        test_values = []
        
        if "min" in stat: # Numeric
            test_values = [stat["min"], stat["max"]]
        elif "options" in stat: # Categorical
            test_values = stat["options"]
            
        # Test each value
        for val in test_values:
            df_test = df.copy()
            df_test[col] = val
            new_pred = model.predict(preprocessor.transform(df_test))[0]
            
            if new_pred != base_pred:
                warnings.append(
                    f"⚠️ Missing '{col}' is CRITICAL. Prediction flips from {base_pred} to {new_pred} if value is {val}."
                )
                break # Found a flip, no need to test other values for this col
                
    return warnings

@app.post("/predict")
def predict_risk(data: PatientData):
    if 'preprocessor' not in globals():
        load_artifacts()
        if 'preprocessor' not in globals():
             raise HTTPException(status_code=503, detail="Model not trained yet")

    input_dict = data.dict()
    # Identify missing values (None)
    missing_cols = [k for k, v in input_dict.items() if v is None]
    
    df = pd.DataFrame([input_dict])
    
    # Sensitivity Check
    warnings = check_sensitivity(model, preprocessor, df, missing_cols)
    
    # Preprocess
    try:
        X_processed = preprocessor.transform(df)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Preprocessing error: {str(e)}")
        
    # Predict
    prob = model.predict_proba(X_processed)[0][1] # Probability of CKD
    pred = model.predict(X_processed)[0] # 0 or 1
    
    prediction_label = "ckd" if pred == 1 else "notckd"
    
    return {
        "prediction": prediction_label,
        "probability": float(prob),
        "risk_level": "High" if prob > 0.7 else "Medium" if prob > 0.3 else "Low",
        "warnings": warnings,
        "missing_cols": missing_cols
    }

@app.post("/feedback")
def submit_feedback(data: FeedbackData):
    # Save feedback for retraining
    feedback_dir = "data/raw/feedback"
    os.makedirs(feedback_dir, exist_ok=True)
    
    df = pd.DataFrame([data.dict()])
    # Rename correct_diagnosis to classification for consistency
    df['classification'] = data.correct_diagnosis
    df.drop('correct_diagnosis', axis=1, inplace=True)
    
    filename = f"{feedback_dir}/feedback_{pd.Timestamp.now().strftime('%Y%m%d%H%M%S')}.csv"
    df.to_csv(filename, index=False)
    
    return {"status": "success", "message": "Feedback received and saved for continuous learning."}
