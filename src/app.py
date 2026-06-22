import os
import sys
import time
import streamlit as st

# Ensure root directory imports resolve cleanly on execution path
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from src.rag_pipeline import CrediTrustRAGEngine

# 1. Page Configuration Setup
st.set_page_config(
    page_title="CrediTrust - RAG Complaint Analyst",
    page_icon="🛡️",
    layout="wide"
)

# Initialize Session State values for conversation persistence and reset safety
if "generated_response" not in st.session_state:
    st.session_state.generated_response = None
if "retrieved_sources" not in st.session_state:
    st.session_state.retrieved_sources = None

# 2. Cache the heavy RAG model engine initialization so it only loads ONCE on startup
@st.cache_resource
def initialize_engine():
    return CrediTrustRAGEngine()

# Application Title Headers
st.title("🛡️ CrediTrust AI - Customer Complaint Analyst Hub")
st.markdown("Interact directly with our vector-indexed consumer grievance databases using local LLM retrieval synthesis.")
st.write("---")

try:
    with st.spinner("⏳ Loading local AI transformer layers and connecting to ChromaDB (Please hold)..."):
        rag_system = initialize_engine()
    st.success("✅ RAG Retrieval Core Online and Addressable!")
except Exception as e:
    st.error(f"⚠️ Failed to boot underlying RAG engine: {e}")
    st.stop()

# 3. Create structural layout columns
col1, col2 = st.columns([3, 2])

with col1:
    st.subheader("💬 Ask Your Analytical Query")
    
    # Text input container box synced with session state memory
    user_query = st.text_input(
        "Enter what pattern or issue you want to investigate:",
        placeholder="e.g., Are consumers experiencing hidden fees on personal loans?",
        key="query_input"
    )
    
    # Layout action buttons side-by-side using micro-columns
    btn_col1, btn_col2 = st.columns([1, 1])
    with btn_col1:
        submit_clicked = st.button("🚀 Analyze Database Patterns", type="primary", use_container_width=True)
    with btn_col2:
        clear_clicked = st.button("🗑️ Clear Conversation", type="secondary", use_container_width=True)

with col2:
    st.subheader("📊 Architectural Settings")
    st.info("**Backend Store:** Local Persistent ChromaDB\n\n**Embedding Architecture:** all-MiniLM-L6-v2\n\n**Local Inference Generator:** TinyLlama-1.1B-Chat\n\n**UX Enhancements:** Real-Time Streaming & Context Tracing Enabled")

# Handle Conversation Reset Switch Trigger
if clear_clicked:
    st.session_state.generated_response = None
    st.session_state.retrieved_sources = None
    st.rerun()

# 4. Trigger Execution Core Loop upon Submission Click
if submit_clicked and user_query:
    st.write("---")
    
    with st.spinner("🤖 Searching vector indices and initiating streaming pipeline..."):
        try:
            # Query the master RAG execution pipeline to get raw text and sources
            raw_response, retrieved_sources = rag_system.answer_customer_inquiry(user_query)
            
            # Save elements to session memory space
            st.session_state.generated_response = raw_response
            st.session_state.retrieved_sources = retrieved_sources
            
            st.subheader("📝 Factual Analyst Response Summary")
            
            # Helper generator function to mimic real token-by-token streaming UI response
            def text_stream_generator():
                for word in raw_response.split(" "):
                    yield word + " "
                    time.sleep(0.04) # Adds a clean human-like typing pacing speed
            
            # Stream the answer live onto the page surface dynamically
            st.write_stream(text_stream_generator)
            
        except Exception as e:
            st.error(f"An processing fault interrupted the retrieval matrix loop: {e}")

# 5. Persistent Render Section for Sources (Keeps sources visible after streaming completes)
if st.session_state.generated_response and st.session_state.retrieved_sources:
    st.write("")
    st.subheader("🔍 Context Tracer Documentation (Top Retrieved Chunks)")
    st.caption("Review the exact physical database segments the analyst model used to formulate its answer above:")
    
    for index, source in enumerate(st.session_state.retrieved_sources):
        with st.expander(f"📄 Match Rank {index+1} | Product Group: {source['product']} (Original Feedback ID: {source['complaint_id']})"):
            st.write(f"**Chunk ID Sequence Reference:** {source['chunk_no']}")
            st.write(f"**Retrieved Exact Context Text:**")
            st.code(source['text'], language="text")
            
elif submit_clicked and not user_query:
    st.warning("⚠️ Please provide a text question pattern inside the input container before initiating lookup execution.")