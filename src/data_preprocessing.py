import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_audit_dataset(file_path):
    print("⏳ [LOAD] Ingesting full raw CFPB complaint dataset...")
    if file_path.endswith('.parquet'):
        df = pd.read_parquet(file_path)
    else:
        df = pd.read_csv(file_path, low_memory=False)
        
    print(f"📊 [AUDIT] Total raw records ingested: {df.shape[0]:,}")
    
    narrative_col = 'Consumer complaint narrative'
    if narrative_col not in df.columns:
        df.rename(columns={col: narrative_col for col in df.columns if 'narrative' in col.lower()}, inplace=True)
        
    # 1. Count complaints with and without narratives
    null_narratives = df[narrative_col].isna().sum()
    valid_narratives = df[narrative_col].notna().sum()
    total = len(df)
    
    print("\n📋 === NARRATIVE SPARSITY AUDIT RESULTS ===")
    print(f"🔍 Complaints WITH narratives:    {valid_narratives:,} ({valid_narratives/total*100:.2f}%)")
    print(f"🔍 Complaints WITHOUT narratives: {null_narratives:,} ({null_narratives/total*100:.2f}%)")
    
    return df, narrative_col

def run_exploratory_data_analysis(df, narrative_col):
    print("\n📈 [EDA] Analyzing distribution of complaints across products...")
    product_counts = df['Product'].value_counts()
    print("\n📋 Top 10 Raw Product Categories in Dataset:")
    print(product_counts.head(10))
    
    print("\n📐 Calculating text length metrics...")
    active_narratives = df[df[narrative_col].notna()][narrative_col].astype(str)
    word_counts = active_narratives.apply(lambda x: len(x.split()))
    
    print("\n📐 Consumer Narrative Word Count Summary Statistics:")
    print(word_counts.describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9, 0.99]))
    
    # Render and save distribution chart
    plt.figure(figsize=(11, 5))
    sns.histplot(word_counts, bins=100, kde=True, color='#2563eb', alpha=0.8)
    plt.axvline(word_counts.median(), color='#ef4444', linestyle='--', label=f'Median: {int(word_counts.median())} words')
    plt.axvline(word_counts.quantile(0.99), color='#f59e0b', linestyle=':', label=f'99th Pct: {int(word_counts.quantile(0.99))} words')
    plt.title('Distribution of Consumer Complaint Narrative Word Counts', fontsize=12, fontweight='bold')
    plt.xlabel('Word Count (Logarithmic Scale Axis)')
    plt.ylabel('Complaint Frequency')
    plt.xscale('log')
    plt.xlim(left=1)
    plt.grid(True, alpha=0.2)
    plt.legend()
    
    os.makedirs('notebooks', exist_ok=True)
    chart_path = "notebooks/narrative_length_distribution.png"
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    print(f"💾 [EDA] Visual chart saved to: {chart_path}")
    plt.close()

def filter_and_clean_pipeline(df, narrative_col):
    print("\n🛡️ [FILTER] Retaining only specified CrediTrust products...")
    
    product_mapping = {
        'Credit card or prepaid card': 'Credit Card',
        'Credit card': 'Credit Card',
        'Prepaid card': 'Credit Card',
        'Student loan': 'Personal Loan',
        'Consumer Loan': 'Personal Loan',
        'Vehicle loan or lease': 'Personal Loan',
        'Checking or savings account': 'Savings Account',
        'Savings account': 'Savings Account',
        'Money transfer, virtual currency, or money service': 'Money Transfer',
        'Money transfers': 'Money Transfer'
    }
    
    df['Mapped_Product'] = df['Product'].map(product_mapping)
    
    # Filter step 1 & 2: Match products and ensure narrative is not empty
    filtered_df = df[df['Mapped_Product'].notna()].copy()
    filtered_df = filtered_df[filtered_df[narrative_col].notna()].copy()
    
    print(f"🧹 [FILTER] Subsetting complete. Domain rows: {filtered_df.shape[0]:,}")
    print(filtered_df['Mapped_Product'].value_counts())
    
    def clean_text(text):
        if not isinstance(text, str):
            return ""
        # 1. Lowercase text
        text = text.lower()
        # 2. Remove standard text masking characters (XXXX, XX/XX)
        text = re.sub(r'x{2,}', '', text)
        text = re.sub(r'\d{2,}', '', text)
        # 3. Remove common boilerplate introductory fragments
        boilerplate_patterns = [
            r"i am writing to file a complaint regarding",
            r"to whom it may concern",
            r"i am writing this complaint because",
            r"please look into this matter"
        ]
        for pattern in boilerplate_patterns:
            text = re.sub(pattern, "", text)
        # 4. Remove special characters and clean extra spacing whitespace
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    print("\n🧼 [CLEAN] Executing text normalization & cleaning pipeline...")
    filtered_df['cleaned_narrative'] = filtered_df[narrative_col].astype(str).apply(clean_text)
    filtered_df = filtered_df[filtered_df['cleaned_narrative'] != ""]
    
    return filtered_df

def main():
    raw_input = "data/raw/complaints.csv"
    output_path = "data/filtered_complaints.csv"
    
    if not os.path.exists(raw_input):
        print(f"⚠️ [NOTICE] Missing target file. Please ensure raw data is at: {raw_input}")
        return
        
    df, narrative_column = load_and_audit_dataset(raw_input)
    run_exploratory_data_analysis(df, narrative_column)
    processed_df = filter_and_clean_pipeline(df, narrative_column)
    
    print(f"\n💾 [EXPORT] Writing processed dataset to production disk: {output_path}...")
    os.makedirs('data', exist_ok=True)
    processed_df.to_csv(output_path, index=False)
    print("🏁 [TASK 1 COMPLETE] Process executed successfully.")

if __name__ == "__main__":
    main()