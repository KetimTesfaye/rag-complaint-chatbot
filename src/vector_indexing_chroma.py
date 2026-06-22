import os
import pandas as pd
import numpy as np
import chromadb

def build_chroma_vector_store(csv_metadata_path, npy_vectors_path, output_db_dir):
    print("⏳ [LOAD] Loading text fragments and mathematical vectors...")
    if not os.path.exists(csv_metadata_path) or not os.path.exists(npy_vectors_path):
        print("⚠️ [ERROR] Base embedding assets missing. Please run vector generation first.")
        return
        
    df = pd.read_csv(csv_metadata_path)
    embeddings = np.load(npy_vectors_path).tolist() # Chroma accepts vectors as list collections
    
    total_records = len(embeddings)
    print(f"📊 [AUDIT] Loaded {total_records:,} chunks ready for ChromaDB ingestion.")
    
    # 1. Initialize local persistent Chroma DB client
    print(f"🧱 [CHROMA] Initializing database storage directory at: {output_db_dir}...")
    client = chromadb.PersistentClient(path=output_db_dir)
    
    # 2. Create or fetch our complaints collection
    # We turn off internal embedding calculations since we already computed them with all-MiniLM-L6-v2
    collection = client.get_or_create_collection(name="financial_complaints")
    
    # 3. Restructure and pack payload into arrays for batch loading
    print("🏷️ [METADATA] Formatting indices, string content, and tracer metadata blocks...")
    
    # Document IDs must be individual strings
    ids = [str(i) for i in range(total_records)]
    
    # Text contents
    documents = df['chunked_text'].astype(str).tolist()
    
    # Dynamic Dictionary Metadata tracing back to source rows
    metadata_list = []
    for idx, row in df.iterrows():
        metadata_list.append({
            'feedback_id': int(row['feedback_id']),
            'product_category': str(row['Mapped_Product']),
            'chunk_sequence_id': int(row['chunk_sequence_id'])
        })
        
    # 4. Ingest datasets in efficient batch frames (Chroma safely batches inputs)
    print("🛰️ [CHROMA] Loading dataset matrix blocks into collection index space...")
    batch_size = 5000
    for i in range(0, total_records, batch_size):
        end_idx = min(i + batch_size, total_records)
        print(f"   📥 Ingesting batch slice: {i:,} to {end_idx:,}...")
        
        collection.add(
            ids=ids[i:end_idx],
            embeddings=embeddings[i:end_idx],
            documents=documents[i:end_idx],
            metadatas=metadata_list[i:end_idx]
        )
        
    print(f"✅ ChromaDB collection populated successfully. Total items tracked: {collection.count():,}")
    print("🏁 [TASK 2 CHROMA COMPLETE] Unified vector database securely compiled.")

if __name__ == "__main__":
    meta_input = "data/processed/embedded_complaints_metadata.csv"
    vectors_input = "data/processed/complaint_vectors.npy"
    store_directory = "chroma_db"
    
    build_chroma_vector_store(meta_input, vectors_input, store_directory)