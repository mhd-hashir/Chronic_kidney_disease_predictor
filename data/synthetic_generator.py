import pandas as pd
import numpy as np
import random
import os
from datetime import datetime

def generate_synthetic_data(num_samples=10, output_dir="data/raw/incoming"):
    """
    Generates synthetic CKD patient data to simulate a continuous stream.
    """
    
    # Define value ranges based on typical medical data (simplified)
    data = {
        'id': range(1, num_samples + 1),
        'age': np.random.randint(20, 90, num_samples),
        'bp': np.random.choice([60, 70, 80, 90, 100], num_samples),
        'sg': np.random.choice([1.005, 1.010, 1.015, 1.020, 1.025], num_samples),
        'al': np.random.choice([0, 1, 2, 3, 4, 5], num_samples),
        'su': np.random.choice([0, 1, 2, 3, 4, 5], num_samples),
        'rbc': np.random.choice(['normal', 'abnormal'], num_samples),
        'pc': np.random.choice(['normal', 'abnormal'], num_samples),
        'pcc': np.random.choice(['present', 'notpresent'], num_samples),
        'ba': np.random.choice(['present', 'notpresent'], num_samples),
        'bgr': np.random.randint(70, 400, num_samples),
        'bu': np.random.randint(10, 200, num_samples),
        'sc': np.round(np.random.uniform(0.5, 15.0, num_samples), 1),
        'sod': np.random.randint(110, 150, num_samples),
        'pot': np.round(np.random.uniform(2.5, 7.0, num_samples), 1),
        'hemo': np.round(np.random.uniform(3.0, 17.0, num_samples), 1),
        'pcv': np.random.randint(20, 55, num_samples),
        'wc': np.random.randint(4000, 15000, num_samples),
        'rc': np.round(np.random.uniform(2.0, 6.5, num_samples), 1),
        'htn': np.random.choice(['yes', 'no'], num_samples),
        'dm': np.random.choice(['yes', 'no'], num_samples),
        'cad': np.random.choice(['yes', 'no'], num_samples),
        'appet': np.random.choice(['good', 'poor'], num_samples),
        'pe': np.random.choice(['yes', 'no'], num_samples),
        'ane': np.random.choice(['yes', 'no'], num_samples),
        'classification': np.random.choice(['ckd', 'notckd'], num_samples, p=[0.6, 0.4]) # Target
    }
    
    df = pd.DataFrame(data)
    
    # Introduce some NaNs to make it realistic (as per real dataset)
    mask = np.random.choice([True, False], size=df.shape, p=[0.05, 0.95])
    df = df.mask(mask)
    
    # Ensure ID and Class are not NaN
    df['id'] = range(1, num_samples + 1)
    df['classification'] = np.where(df['classification'].isna(), 'ckd', df['classification']) 

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{output_dir}/patients_{timestamp}.csv"
    
    df.to_csv(filename, index=False)
    print(f"Generated {num_samples} synthetic patient records at {filename}")
    return filename

if __name__ == "__main__":
    generate_synthetic_data()
