import os
import torch
from transformers import pipeline
from src.retriever_engine import CrediTrustChromaRetriever
from src.prompt_engineering import CrediTrustPromptBuilder

class CrediTrustRAGEngine:
    def __init__(self):
        print("🛰️ [INIT] Initializing unified CrediTrust RAG Engine...")
        
        # 1. Connect to our pre-built ChromaDB Retriever
        self.retriever = CrediTrustChromaRetriever()
        self.prompt_builder = CrediTrustPromptBuilder()
        
        # 2. Initialize our local Hugging Face LLM Text Generation pipeline
        # Using "TinyLlama/TinyLlama-1.1B-Chat-v1.0" - lightweight, fast, and optimized for local CPUs
        print("🚀 [LLM] Downloading/Loading local Text Generation model (TinyLlama-1.1B)...")
        self.generator = pipeline(
            "text-generation",
            model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            torch_dtype=torch.float32,
            device_map="auto"
        )
        print("✅ [ONLINE] RAG Engine is fully loaded and active.")

    def answer_customer_inquiry(self, user_question):
        # Step A: Retrieve top 5 matches from ChromaDB
        retrieved_chunks = self.retriever.search_relevant_complaints(user_question, k=5)
        
        # Step B: Assemble the rigid instruction prompt matrix
        compiled_prompt = self.prompt_builder.generate_rag_prompt(user_question, retrieved_chunks)
        
        # Step C: Feed the prompt into the local LLM generator
        print("🤖 [GENERATION] Executing model inference over context (Thinking)...")
        outputs = self.generator(
            compiled_prompt,
            max_new_tokens=200,
            temperature=0.1,  # Low temperature forces exact, deterministic factual reliance
            do_sample=False,  # Turns off creative random word swapping
            return_full_text=False # Returns only the new words typed under Factual Analyst Answer
        )
        
        # Extract and return the model's generated text response
        generated_answer = outputs[0]['generated_text'].strip()
        return generated_answer, retrieved_chunks

if __name__ == "__main__":
    # Test Query
    analytical_query = "What patterns are we seeing regarding customer experiences with promised credit card fee waivers?"
    
    try:
        # Initialize the complete system
        rag_system = CrediTrustRAGEngine()
        
        # Execute the full RAG loop
        answer, sources = rag_system.answer_customer_inquiry(analytical_query)
        
        print("\n==================================================")
        print("📝 SYSTEM GENRE OUTCOME (Factual Analyst Answer):")
        print("==================================================")
        print(answer)
        print("==================================================")
        
    except Exception as e:
        print(f"⚠️ RAG Pipeline Execution halted: {e}")