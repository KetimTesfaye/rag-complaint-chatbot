class CrediTrustPromptBuilder:
    @staticmethod
    def generate_rag_prompt(question, retrieved_chunks):
        """
        Combines the user's question and retrieved database context into a 
        highly structured, hallucination-resistant prompt for the LLM.
        """
        
        # 1. Format the retrieved chunks into a clean, readable text block for the LLM
        context_blocks = []
        for chunk in retrieved_chunks:
            context_blocks.append(
                f"--- COMPLAINT RECORD (Product: {chunk['product']} | ID: {chunk['complaint_id']}) ---\n"
                f"{chunk['text']}"
            )
        
        formatted_context = "\n\n".join(context_blocks)
        
        # 2. Define the strict system instructions and template boundaries
        prompt_template = f"""You are an elite, highly precise senior financial analyst assistant for CrediTrust. Your core mandate is to review historical consumer complaints and answer user inquiries with absolute factual accuracy.

CRITICAL INSTRUCTIONS FOR RESPONSIBLE GENERATION:
1. Ground your answer STRICTLY and EXCLUSIVELY within the provided "Retrieved Complaint Context" below.
2. If the retrieved context does not contain clear facts or explicit evidence to address the question, reply with exactly: "I am sorry, but the provided historical database context does not contain sufficient information to answer your question."
3. Do NOT utilize external knowledge, guess, or extrapolate beyond the text explicitly provided. 
4. Maintain a neutral, professional corporate tone. Avoid speculating about bank intentions.

Retrieved Complaint Context:
{formatted_context}

User's Analytical Question:
{question}

Factual Analyst Answer:"""

        return prompt_template

if __name__ == "__main__":
    # Simulate a run using the exact data structure we retrieved from ChromaDB in the previous step
    mock_retrieved_chunks = [
        {
            'rank': 1,
            'complaint_id': 3251,
            'product': 'Credit Card',
            'text': "i called regarding the annual fee that i was never told about... the representative had been able to waive the annual fee tonight employee number refused to waive the annual fee"
        }
    ]
    
    test_query = "What patterns are we seeing regarding customer experiences with promised credit card fee waivers?"
    
    builder = CrediTrustPromptBuilder()
    final_prompt = builder.generate_rag_prompt(test_query, mock_retrieved_chunks)
    
    print("🏆 === COMPILED PRODUCTION PROMPT TEMPLATE ===")
    print(final_prompt)