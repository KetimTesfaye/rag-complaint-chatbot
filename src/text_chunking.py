import os
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter

def run_narrative_chunking_pipeline(input_path, output_path, chunk_size=500, chunk_overlap=50):
    print("⏳ [LOAD] Loading stratified complaints sample...")
    if not os.path.exists(input_path):
        print(f"⚠️ [ERROR] Sample file missing at '{input_path}'. Run the sampling script first.")
        return
        
    df = pd.read_csv(input_path)
    print(f"📊 [AUDIT] Ingested {df.shape[0]:,} records for text fragmentation.")
    
    # Initialize the splitter with dynamic paragraph/sentence boundaries
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunked_records = []
    
    print("\n✂️ [CHUNKING] Processing narratives into uniform segments...")
    for idx, row in df.iterrows():
        narrative = str(row['cleaned_narrative'])
        
        # Split the text into character blocks
        fragments = splitter.split_text(narrative)
        
        # Track each fragment as a standalone entry while keeping metadata intact
        for chunk_seq, fragment in enumerate(fragments):
            chunked_records.append({
                'feedback_id': row.get('feedback_id', idx),
                'Mapped_Product': row['Mapped_Product'],
                'chunk_sequence_id': chunk_seq,
                'chunked_text': fragment,
                'chunk_length_chars': len(fragment)
            })
            
    chunked_df = pd.DataFrame(chunked_records)
    print(f"✅ [COMPLETE] Splitting complete. Total generated chunks: {chunked_df.shape[0]:,}")
    print(f"📐 Average character length per chunk: {chunked_df['chunk_length_chars'].mean():.2f} characters.")
    
    # Export out to project processing environment
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    chunked_df.to_csv(output_path, index=False)
    print(f"💾 [EXPORT] Chunked vector-ready payload written to: {output_path}")

if __name__ == "__main__":
    sampled_input = "data/processed/stratified_complaints_sample.csv"
    chunked_output = "data/processed/chunked_complaints.csv"
    
    # Run pipeline with optimized parameters
    run_narrative_chunking_pipeline(sampled_input, chunked_output, chunk_size=500, chunk_overlap=50)