import os
import sys
# Ensure local path references resolve smoothly
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from rag_pipeline import CrediTrustRAGEngine

def run_system_evaluation():
    # 1. Define a robust list of 8 representative domain questions covering all 4 products
    test_questions = [
        # Credit Card
        "What common complaints do clients have regarding annual fee waivers being rejected?",
        "Are customers reporting unauthorized charges or fraud on their credit cards?",
        # Savings Account
        "What patterns exist around banks freezing savings accounts without notifying holders?",
        "What difficulties do customers face when trying to withdraw money or close a savings account?",
        # Personal Loan
        "Are consumers reporting deceptive interest rates or hidden fees on personal loans?",
        "What issues do individuals experience with customer service when requesting loan modifications?",
        # Money Transfer
        "What are the main bottlenecks causing money transfers or wire transfers to get delayed?",
        "How do customers describe their experiences trying to recover funds sent to the wrong recipient?"
    ]
    
    print("⏳ [EVALUATION] Initializing live CrediTrust RAG Engine...")
    try:
        rag_system = CrediTrustRAGEngine()
    except Exception as e:
        print(f"⚠️ Failed to initialize RAG pipeline: {e}")
        return

    print(f"\n📊 [EVALUATION] Running inference across {len(test_questions)} strategic benchmarks...")
    
    # Open a local evaluation report markdown file to append to
    report_path = "reports/task_3_evaluation_table.md"
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    
    with open(report_path, "w", encoding="utf-8") as f:
        # Write Markdown Table Header
        f.write("# Task 3: RAG Pipeline Qualitative Evaluation Matrix\n\n")
        f.write("| Question | Generated Answer | Retrieved Sources (IDs) | Quality Score (1-5) | Comments / Technical Analysis |\n")
        f.write("| :--- | :--- | :--- | :---: | :--- |\n")
        
        for idx, question in enumerate(test_questions, 1):
            print(f"\n🏃‍♂️ Running Question {idx}/{len(test_questions)}: '{question}'")
            try:
                answer, sources = rag_system.answer_customer_inquiry(question)
                
                # Format answers to remove line breaks for clean Markdown rendering
                clean_answer = answer.replace("\n", " ").replace("|", "\\|")
                
                # Extract top 2 unique source information details
                source_ids = [f"ID {s['complaint_id']} ({s['product']})" for s in sources[:2]]
                formatted_sources = ", ".join(source_ids)
                
                # Write an empty evaluation scaffold slot for manual scoring review
                f.write(f"| {question} | {clean_answer} | {formatted_sources} | [ ] | *Awaiting manual analysis criteria loop verification.* |\n")
                
            except Exception as e:
                print(f"⚠️ Error running query {idx}: {e}")
                f.write(f"| {question} | ERROR: Execution failed during local inference loop. | N/A | 1 | Pipeline crashed: {str(e)} |\n")
                
    print(f"\n🏁 [EVALUATION COMPLETE] Draft Markdown assessment table generated at: {report_path}")

if __name__ == "__main__":
    run_system_evaluation()