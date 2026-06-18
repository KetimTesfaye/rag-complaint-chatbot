import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

def execute_embedding_generation(input_csv_path, output_csv_path, output_npy_path):
    print("⏳ [LOAD] Loading chunked text payload...")
    if not os.path.exists(input_csv_path):
        print(f"⚠️ [ERROR] Source file missing at '{input_csv_path}'. Please run Task 2 chunking first.")
        return
        
    df = pd.read_csv(input_csv_path)
    print(f"📊 [AUDIT] Ingested {df.shape[0]:,} fragments ready for vector conversion.")
    
    # 1. Initialize the target transformer model
    print("🚀 [MODEL] Downloading/Loading sentence-transformers/all-MiniLM-L6-v2...")
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    # Extract the text list to pass to the embedding model
    chunks_text = df['chunked_text'].astype(str).tolist()
    
    # 2. Compute vectors with a progress monitor batch layout
    print("🛰️ [EMBED] Generating vector representations across chunks (This may take a minute)...")
    embeddings = model.encode(
        chunks_text, 
        batch_size=64, 
        show_progress_bar=True, 
        convert_to_numpy=True
    )
    
    print(f"✅ [COMPLETE] Embeddings computed successfully.")
    print(f"📐 Vector Matrix Dimensions: {embeddings.shape} (Total Items x 384 Latent Features)")
    
    # 3. Export vectors out to disk matrix storage
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    
    # Save the raw mathematical coordinates as a fast binary file
    np.save(output_npy_path, embeddings)
    print(f"💾 [EXPORT] Binary vector matrix written to: {output_npy_path}")
    
    # Save a reference index file keeping track of the data frame links
    df.to_csv(output_csv_path, index=False)
    print(f"💾 [EXPORT] Reference index metadata written to: {output_csv_path}")

if __name__ == "__main__":
    chunked_input = "data/processed/chunked_complaints.csv"
    indexed_output_csv = "data/processed/embedded_complaints_metadata.csv"
    vectors_output_npy = "data/processed/complaint_vectors.npy"
    
    execute_embedding_generation(chunked_input, indexed_output_csv, vectors_output_npy)