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
            if hasattr(self.preprocessor, 'get_feature_names_out'):
                try:
                    feats = self.preprocessor.get_feature_names_out()
                except:
                     feats = [f"Feature {i}" for i in range(len(vals))]
            else:
                feats = self.feature_names if hasattr(self, 'feature_names') else [f"Feature {i}" for i in range(len(vals))]
                
            # Ensure lengths match
            if len(feats) != len(vals):
                 return {f"Feat_{i}": v for i, v in enumerate(vals)}
                 
            return dict(zip(feats, vals))
            
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
