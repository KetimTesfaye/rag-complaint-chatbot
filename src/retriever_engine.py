import os
import chromadb
from sentence_transformers import SentenceTransformer

class CrediTrustChromaRetriever:
    def __init__(self, db_dir="chroma_db", collection_name="financial_complaints"):
        # 1. Load the exact same embedding model used to index the data
        print("🚀 [INIT] Loading all-MiniLM-L6-v2 embedding model...")
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        # 2. Connect to our persistent local database folder
        print(f"🧱 [LOAD] Connecting to persistent ChromaDB at: {db_dir}...")
        if not os.path.exists(db_dir):
            raise FileNotFoundError(f"ChromaDB directory '{db_dir}' not found. Please run indexing first.")
            
        self.client = chromadb.PersistentClient(path=db_dir)
        self.collection = self.client.get_collection(name=collection_name)
        print(f"✅ ChromaDB collection loaded. Ingested chunks available: {self.collection.count():,}")

    def search_relevant_complaints(self, user_query, k=5):
        print(f"\n🔍 [SEARCH] Scanning database for top-{k} matches to: '{user_query}'")
        
        # 3. Turn the human question into a list of numbers (embedding vector)
        query_vector = self.model.encode([user_query]).tolist()
        
        # 4. Query ChromaDB for the closest vector matches
        results = self.collection.query(
            query_embeddings=query_vector,
            n_results=k
        )
        
        # 5. Format the extracted matches beautifully
        extracted_chunks = []
        for rank in range(len(results['ids'][0])):
            extracted_chunks.append({
                'rank': rank + 1,
                'distance_score': float(results['distances'][0][rank]) if results['distances'] else None,
                'complaint_id': results['metadatas'][0][rank]['feedback_id'],
                'product': results['metadatas'][0][rank]['product_category'],
                'chunk_no': results['metadatas'][0][rank]['chunk_sequence_id'],
                'text': results['documents'][0][rank]
            })
            
        return extracted_chunks

if __name__ == "__main__":
    # Test our new retriever with a standard complaint scenario
    sample_problem = "My credit card was charged an annual fee even though I was promised it would be waived."
    
    try:
        # Initialize and run our search engine
        retriever = CrediTrustChromaRetriever()
        top_matches = retriever.search_relevant_complaints(sample_problem, k=5)
        
        print("\n🏆 === TOP RETRIEVED CONTEXT MATCHES ===")
        for match in top_matches:
            print(f"\n🥇 Rank {match['rank']} | Product: {match['product']} | Original ID: {match['complaint_id']} (Distance: {match['distance_score']:.4f})")
            print(f"📄 Fragment Content: {match['text']}...")
            print("-" * 50)
            
    except Exception as e:
        print(f"⚠️ Retrieval execution failed: {e}")