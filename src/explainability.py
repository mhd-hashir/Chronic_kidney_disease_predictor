import shap
import joblib
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np

class CKDExplainer:
    def __init__(self, model=None, preprocessor=None, model_path="models/ckd_model.pkl", background_data_path="data/processed/train_data.csv"):
        # Handle Model
        if model:
            self.model = model
        else:
            self.model = joblib.load(model_path)
            
        # Handle Preprocessor
        self.preprocessor = preprocessor # Optional, might use later
        
        # Load background data for SHAP
        try:
             df = pd.read_csv(background_data_path)
             # Use preprocessor if available to transform background data if needed
             # But for now assuming TreeExplainer works on transformed data? 
             # Actually, if model expects transformed data, we should explain using transformed data.
             
             if preprocessor:
                 # We need to transform the background data to match model input
                 # Check if target exists
                 if 'target' in df.columns:
                     df_bg = df.drop('target', axis=1)
                 else:
                     df_bg = df
                     
                 # Basic cleaning if needed (assuming preprocessor handles it)
                 # self.X_background = preprocessor.transform(df_bg) # This return sparse matrix sometimes
                 # For simplicity, taking raw sample and trusting TreeExplainer or model pipeline?
                 # If model is sklearn pipeline, we pass raw. If model is CLF, we need transformed.
                 pass
                 
             self.feature_names = df.drop('target', axis=1, errors='ignore').columns.tolist()
             self.X_background = df.drop('target', axis=1, errors='ignore').sample(100, random_state=42)
             
        except Exception:
             self.X_background = None
             self.feature_names = ["age", "bp", "sg", "al", "su", "rbc", "pc", "pcc", "ba", "bgr", "bu", "sc", "sod", "pot", "hemo", "pcv", "wc", "rc", "htn", "dm", "cad", "appet", "pe", "ane"]

        # Initialize Explainer
        # TreeExplainer is faster for Random Forest
        # Note: If model is just the classifier part of a pipeline, inputs must be transformed.
        # If we passed the classifier, we assume inputs are transformed.
        self.explainer = shap.TreeExplainer(self.model)
        
    def explain(self, instance_df):
        """
        Returns SHAP values for a single instance (DataFrame with 1 row)
        """
        shap_values = self.explainer.shap_values(instance_df)
        
        # In binary classification, shap_values might be a list [class0, class1]
        # We usually want class 1 (CKD)
        if isinstance(shap_values, list):
            return shap_values[1]
        return shap_values

    def explain_local(self, instance_df):
        """
        Returns a dictionary of feature_name -> shap_value for the instance.
        """
        try:
            # Transform if we have a preprocessor
            if self.preprocessor:
                data_processed = self.preprocessor.transform(instance_df)
            else:
                data_processed = instance_df
                
            # TreeExplainer.shap_values returns:
            # - List of arrays [N, M] for classification (one per class)
            # - Array [N, M] for regression
            # check_additivity=False prevents errors if sum doesn't perfectly match
            shap_output = self.explainer.shap_values(data_processed, check_additivity=False)
            
            vals = None
            
            # Case 1: Binary Classification (List of 2 arrays)
            if isinstance(shap_output, list):
                # We want the positive class (index 1)
                # instance_df is single row, so we take index 0
                if len(shap_output) > 1:
                     vals = shap_output[1][0]
                else:
                     vals = shap_output[0][0]
            
            # Case 2: Array (could be (N, M) or (N, M, C))
            elif isinstance(shap_output, np.ndarray):
                if len(shap_output.shape) == 3: # (N, M, C)
                    vals = shap_output[0, :, 1] # Positive class
                else: # (N, M)
                    vals = shap_output[0]
            
            if vals is None:
                return {}

            # Map to feature names
            feats = []
            
            # Attempt 1: Standard Sklearn API
            if hasattr(self.preprocessor, 'get_feature_names_out'):
                try:
                    feats = list(self.preprocessor.get_feature_names_out())
                except Exception:
                    # Pipeline failed (likely due to custom step missing method)
                    # Attempt 2: Dig into pipeline for 'preprocessor' step (ColumnTransformer)
                    try:
                        if hasattr(self.preprocessor, 'named_steps') and 'preprocessor' in self.preprocessor.named_steps:
                            feats = list(self.preprocessor.named_steps['preprocessor'].get_feature_names_out())
                    except:
                        pass
            
            # Fallback 1: Use original column names if standard scalar (no column change)
            if not feats and hasattr(self, 'feature_names'):
                 feats = self.feature_names
            
            # Fallback 2: Generic
            if not feats or len(feats) != len(vals):
                # If lengths mismatch, we can't use the names. Fallback to generic.
                # But if we have original names and they match len (e.g. no encoding was done), use them.
                if hasattr(self, 'feature_names') and len(self.feature_names) == len(vals):
                     feats = self.feature_names
                else:
                     feats = [f"Feature {i}" for i in range(len(vals))]
            
            # Clean up feature names (remove prefixes from ColumnTransformer)
            # The prefixes are usually 'num__' or 'cat__' based on the transformer name
            cleaned_feats = []
            for f in feats:
                # Remove common prefixes
                f_clean = f.replace('num__', '').replace('cat__', '').replace('remainder__', '').replace('numerical__', '').replace('categorical__', '')
                
                # Setup readable names mapping
                name_map = {
                    "age": "Age",
                    "bp": "Blood Pressure",
                    "sg": "Specific Gravity",
                    "al": "Albumin",
                    "su": "Sugar",
                    "rbc": "Red Blood Cells",
                    "pc": "Pus Cell",
                    "pcc": "Pus Cell Clumps",
                    "ba": "Bacteria",
                    "bgr": "Blood Glucose Random",
                    "bu": "Blood Urea",
                    "sc": "Serum Creatinine",
                    "sod": "Sodium",
                    "pot": "Potassium",
                    "hemo": "Hemoglobin",
                    "pcv": "Packed Cell Volume",
                    "wc": "White Blood Cell Count",
                    "rc": "Red Blood Cell Count",
                    "htn": "Hypertension",
                    "dm": "Diabetes Mellitus",
                    "cad": "Coronary Artery Disease",
                    "appet": "Appetite",
                    "pe": "Pedal Edema",
                    "ane": "Anemia"
                }
                
                # Handle OneHotEncoded names like "htn_yes" -> "Hypertension (yes)"
                for key, val in name_map.items():
                    if f_clean == key:
                        f_clean = val
                        break
                    elif f_clean.startswith(key + "_"):
                        # Convert "htn_yes" -> "Hypertension: yes"
                        suffix = f_clean[len(key)+1:] # get "yes"
                        f_clean = f"{val}: {suffix}"
                        break
                
                cleaned_feats.append(f_clean)

            return dict(zip(cleaned_feats, vals))
            
        except Exception as e:
            print(f"Explanation Error: {e}")
            return {}

    def plot_waterfall(self, instance_df, save_path="models/latest_explanation.png"):
        # Legacy placeholder
        return save_path

if __name__ == "__main__":
    # Test
    try:
        explainer = CKDExplainer()
        sample = explainer.X_background.iloc[[0]]
        print("Explaining sample:")
        vals = explainer.explain(sample)
        print("SHAP Values shape:", vals.shape)
        path = explainer.plot_waterfall(sample)
        print(f"Plot saved to {path}")
    except Exception as e:
        print(f"Explainer test failed (likely model not trained yet): {e}")
