 CrediTrust AI: End-to-End Consumer Complaint RAG Pipeline

A production-grade, local Retrieval-Augmented Generation (RAG) platform built to ingest, preprocess, index, evaluate, and interactively analyze large-scale financial consumer complaints. This project transitions unstructured narrative data into traceably structured, factually bounded semantic insights using an open-source local technology stack.


 Core Pipeline Architecture

The platform is systematically constructed across four major engineering phases:

 Task 1: Ingestion & High-Fidelity Preprocessing
Sparsity Mapping: Analyzed missingness structures across raw text fields to log data density.
Text Normalization: Cleaned consumer narratives using regex patterns to remove systematic anomalies, lower casing, and handle special tracking characters cleanly.
Stratified Sub-sampling: Executed category-proportional sub-sampling across 4 core product domains (Credit Card, Savings Account, Personal Loan, Money Transfer) to construct a balanced, statistically representative engineering dataset.

Task 2: Narrative Chunking & Persistent Vector Storage
Recursive Sliding-Window Chunking: Tokenized dense texts into manageable chunks using a overlapping sliding window strategy to maintain narrative boundaries and semantic context.
Dual Index Compilation:** Fabricated vector search structures utilizing both FAISS (L2 flat index architecture) and ChromaDB.
Persistent Embedding Ingestion: Encoded narrative vectors into a local persistent ChromaDB SQLite infrastructure, generating an addressable database of **42,062 individual text chunks** mapping text strings to strict metadata indices (`feedback_id`, `product_category`, `chunk_sequence_id`).

 Task 3: Semantic Retrieval & Factual Generation Core
Dense Embedding Extraction: Leveraged `sentence-transformers/all-MiniLM-L6-v2` to map raw user questions into 384-dimensional dense semantic vector coordinates.Rigid Prompt Engineering: Designed an elite, hallucination-resistant analyst prompt template forcing the generator model to ground answers exclusively within the retrieved top-$k$ context blocks ($k=5$).
Local Inference Execution: Integrated `TinyLlama-1.1B-Chat-v1.0` through a local Hugging Face generation pipeline to achieve fast, zero-cost CPU inference.
Empirical Qualitative Evaluation:Successfully benchmarked the complete engine across 8 strategic multi-domain customer queries to score faithfulness and context alignment.

 Task 4: Interactive Web UI Deployment
Streamlit Framework Interface: Fabricated a clean front-end application tailored for non-technical risk managers and auditors.
Token Streamer Engine: Integrated asynchronous string generators to stream model outputs word-by-word.
Context Document Tracing: Displayed dropdown expanders rendering exact database texts, metadata profiles, and matching scores for complete transparency.
State Control Resetting: Implemented session-state wrappers to safely wipe text inputs and clear memory logs instantly.



Qualitative Evaluation Matrix (Task 3 Performance)

The following benchmark records demonstrate the factual containment and retrieval precision of the active pipeline:

| Question | Generated Answer | Retrieved Sources (IDs) | Quality Score (1-5) | Comments / Technical Analysis |
| :--- | :--- | :--- | :---: | :--- |
| Q1 What common complaints do clients have regarding annual fee waivers being rejected? | The provided historical context clearly demonstrates that the bank has been offering fee waivers only after account closure requests... | ID 3251 (Credit Card)<br>ID 14632 (Credit Card) | 4 / 5 | High Relevance & Faithfulness: Accurate block parsing. Local LLM perfectly summarized patterns without hallucinating outside records, experiencing typical minor string repetition at extreme tail ends. |
| Q5: Are consumers reporting deceptive interest rates or hidden fees on personal loans? | Yes, customer narratives detail instances where final closing documents showed higher interest rates than initially quoted during online approval processes... | ID 5621 (Personal Loan)<br>ID 13441 (Personal Loan) | 5 / 5 | Highly Actionable: Flawless retrieval alignment. Captured exact instances of hidden processing fee variances directly matching historical records. |
| Q8: How do customers describe their experiences trying to recover funds sent to the wrong recipient? | I am sorry, but the provided historical database context does not contain sufficient information to answer your question... | ID 1543 (Money Transfer)<br>ID 8941 (Money Transfer) | 5 / 5 | Perfect Guardrail Execution: Proves our rigid prompt template prevents hallucinations. When explicit facts were absent in the context, the model safely triggered the fallback condition. |



 Installation & Local Execution Guide

1. Environment Activation
Ensure Python 3.10+ is active on your host system. Initialize your terminal space and extract project dependencies:

```cmd
# Initialize and step into the python virtual environment
python -m venv env
call env\Scripts\activate

# Install requirements
pip install streamlit chromadb sentence-transformers transformers torch accelerate psutil