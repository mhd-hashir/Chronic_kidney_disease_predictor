import shap
import joblib
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np

class CKDExplainer:
    def __init__(self, model_path="models/ckd_model.pkl", background_data_path="data/processed/train_data.csv"):
        self.model = joblib.load(model_path)
        
        # Load background data for SHAP (needed for KernelExplainer or TreeExplainer sometimes)
        # We'll take a sample of the training data as background
        df = pd.read_csv(background_data_path)
        self.X_background = df.drop('target', axis=1).sample(100, random_state=42)
        
        # Initialize Explainer
        # TreeExplainer is faster for Random Forest
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

    def plot_waterfall(self, instance_df, save_path="models/latest_explanation.png"):
        """
        Generates and saves a waterfall plot for the instance.
        """
        shap_values = self.explainer(instance_df)
        
        # For binary classification, slice it effectively
        # shap.plots.waterfall handles Explanation objects
        
        # If the model is binary, shap_values[0] is for the single instance
        # But we need to make sure we're looking at the positive class
        
        # Let's handle the Explanation object carefully
        expected_value = self.explainer.expected_value
        if isinstance(expected_value, list) or isinstance(expected_value, np.ndarray):
             expected_value = expected_value[1] # Positive class
             
        # Extract the SHAP values for the positive class for this instance
        # instance_shap = shap_values.values[:, 1] if len(shap_values.values.shape) == 2 else shap_values.values
        
        # Easier way with plotting:
        plt.figure()
        # We need to pick the index 1 for CKD class if it's a list based output in older SHAP versions
        # But with new API:
        s_val = shap_values[0, :, 1] if len(shap_values.shape) == 3 else shap_values[0]
        
        shap.plots.waterfall(s_val, show=False)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
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
