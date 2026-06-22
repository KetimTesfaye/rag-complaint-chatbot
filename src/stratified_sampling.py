import os
import pandas as pd
from sklearn.model_selection import train_test_split

def execute_stratified_sampling(input_path, output_path, target_sample_size=15000, random_seed=42):
    print("⏳ [LOAD] Reading cleaned dataset from disk...")
    if not os.path.exists(input_path):
        print(f"⚠️ [ERROR] Base asset missing at '{input_path}'. Please run Task 1 first.")
        return
        
    df = pd.read_csv(input_path, low_memory=False)
    print(f"📊 [AUDIT] Total available clean records: {df.shape[0]:,}")
    
    # 1. Verify distribution of our target grouping column
    group_col = 'Mapped_Product'
    if group_col not in df.columns:
        # Fallback to standard Product column if mapped version isn't present
        group_col = 'Product'
        
    print("\n📋 Baseline Population Proportions:")
    proportions = df[group_col].value_counts(normalize=True) * 100
    for prod, prop in proportions.items():
        print(f"  - {prod}: {prop:.2f}% ({df[group_col].value_counts()[prod]:,} rows)")
        
    # Calculate the fraction needed to get exactly our target sample size
    sampling_fraction = target_sample_size / len(df)
    
    print(f"\n🎯 [STRATIFY] Extracting a {sampling_fraction*100:.4f}% stratified sample ({target_sample_size:,} rows)...")
    
    # Perform Stratified Sampling using train_test_split
    sample_df, _ = train_test_split(
        df,
        train_size=target_sample_size,
        stratify=df[group_col],
        random_state=random_seed
    )
    
    print("\n✅ Stratified Sample Proportions (Verification):")
    sample_proportions = sample_df[group_col].value_counts(normalize=True) * 100
    for prod, prop in sample_proportions.items():
        print(f"  - {prod}: {prop:.2f}% ({sample_df[group_col].value_counts()[prod]:,} rows)")
        
    # Save the stratified subset
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sample_df.to_csv(output_path, index=False)
    print(f"\n💾 [EXPORT] Stratified dataset successfully written to: {output_path}")

if __name__ == "__main__":
    cleaned_data_base = "data/filtered_complaints.csv"
    sampled_data_output = "data/processed/stratified_complaints_sample.csv"
    
    execute_stratified_sampling(cleaned_data_base, sampled_data_output, target_sample_size=15000)