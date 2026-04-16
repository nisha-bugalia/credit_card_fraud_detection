import pandas as pd
import numpy as np
import os

def generate_data(num_samples=5000, fraud_ratio=0.02, output_path="data/synthetic_creditcard.csv"):
    """
    Generates a synthetic credit card fraud dataset mimicking the Kaggle dataset structure.
    Used for testing the pipeline when the massive Kaggle dataset is not downloaded yet.
    """
    np.random.seed(42)
    
    # Columns matching kaggle dataset (Time, V1..V28, Amount, Class)
    cols = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount", "Class"]
    
    # Generate Time (seconds elapsed, let's say over 2 days -> 172800 sec max)
    time_col = np.random.randint(0, 172800, num_samples)
    
    # Generate V1-V28 (PCA-like features, standard normal distributed)
    v_cols = np.random.normal(0, 1, size=(num_samples, 28))
    
    # Generate Amount (Right skewed, most amounts small)
    amount_col = np.abs(np.random.lognormal(mean=2.0, sigma=1.5, size=num_samples))
    
    # Generate Class labels heavily biased towards 0
    num_frauds = int(num_samples * fraud_ratio)
    labels = np.zeros(num_samples)
    
    # Randomly assign frauds
    fraud_indices = np.random.choice(num_samples, num_frauds, replace=False)
    labels[fraud_indices] = 1
    
    # Alter some V features for frauds to make them mathematically separable by models
    for idx in fraud_indices:
        # Shift means for fraud cases for V1, V2, V3, and Amount to make detection possible
        v_cols[idx][0] -= 5.0 # V1
        v_cols[idx][1] += 4.0 # V2
        v_cols[idx][2] -= 6.0 # V3
        amount_col[idx] += np.random.uniform(100, 1000) # Frauds might have larger irregular amounts occasionally
        
    # Build dataframe
    data = np.column_stack((time_col, v_cols, amount_col, labels))
    df = pd.DataFrame(data, columns=cols)
    
    # Ensure Class is integer
    df['Class'] = df['Class'].astype(int)
    
    # Make sure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save
    df.to_csv(output_path, index=False)
    print(f"✅ Generated synthetic dataset with {num_samples} samples at {output_path}")
    print(f"💰 Non-Frauds (Class 0): {num_samples - num_frauds}")
    print(f"🚨 Frauds (Class 1): {num_frauds}")

if __name__ == "__main__":
    generate_data(num_samples=10000, fraud_ratio=0.015)  # 10k rows, 1.5% fraud
