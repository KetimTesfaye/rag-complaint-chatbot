import os
import pickle
import pandas as pd
import numpy as np
import faiss

def build_vector_store(csv_metadata_path, npy_vectors_path, output_index_dir):
    print("⏳ [LOAD] Loading mathematical vectors and reference metadata...")
    if not os.path.exists(csv_metadata_path) or not os.path.exists(npy_vectors_path):
        print("⚠️ [ERROR] Base embedding assets missing. Run generate_embeddings.py first.")
        return
        
    df = pd.read_csv(csv_metadata_path)
    embeddings = np.load(npy_vectors_path).astype('float32')
    
    total_records, vector_dim = embeddings.shape
    print(f"📊 [AUDIT] Loaded {total_records:,} chunks with a vector dimensionality of {vector_dim}.")
    
    # 1. Initialize a flat L2 index structure
    print(f"🧱 [INDEX] Initializing FAISS FlatL2 Index index (Dimension: {vector_dim})...")
    index = faiss.IndexFlatL2(vector_dim)
    
    # 2. Feed the dense embedding coordinates into the FAISS index
    print("🛰️ [INDEX] Injecting embeddings into the similarity search engine...")
    index.add(embeddings)
    print(f"✅ FAISS Index populated successfully. Total indexed elements: {index.ntotal:,}")
    
    # 3. Create a clean metadata map linking each index position to its source attributes
    print("🏷️ [METADATA] Mapping vector positions to per-chunk payload metrics...")
    metadata_store = {}
    for idx, row in df.iterrows():
        metadata_store[idx] = {
            'feedback_id': int(row['feedback_id']),
            'product_category': str(row['Mapped_Product']),
            'chunk_sequence_id': int(row['chunk_sequence_id']),
            'text_payload': str(row['chunked_text'])
        }
        
    # 4. Save both the FAISS binary and metadata payload back out to local disk paths
    os.makedirs(output_index_dir, exist_ok=True)
    index_file_path = os.path.join(output_index_dir, "complaints_faiss.index")
    metadata_file_path = os.path.join(output_index_dir, "complaints_metadata.pkl")
    
    print(f"\n💾 [EXPORT] Writing FAISS binary index map out to: {index_file_path}...")
    faiss.write_index(index, index_file_path)
    
    print(f"💾 [EXPORT] Writing associated object dictionary payload to: {metadata_file_path}...")
    with open(metadata_file_path, 'wb') as f:
        pickle.dump(metadata_store, f)
        
    print("🏁 [TASK 2 COMPLETE] Local vector store successfully indexed and stored.")

if __name__ == "__main__":
    meta_input = "data/processed/embedded_complaints_metadata.csv"
    vectors_input = "data/processed/complaint_vectors.npy"
    store_directory = "vector_store"
    
    build_vector_store(meta_input, vectors_input, store_directory)