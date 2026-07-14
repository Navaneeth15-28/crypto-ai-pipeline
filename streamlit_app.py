import streamlit as st
import google.generativeai as genai
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import re

# 1. Setup Page Config
st.set_page_config(page_title="Crypto AI Pipeline", page_icon="📈")
st.title("📈 Crypto-AI Live Search")

# Sidebar for API Key
st.sidebar.header("Configuration")
api_key = st.sidebar.text_input("Enter Gemini API Key", type="password")

@st.cache_resource
def load_models():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    qdrant = QdrantClient("http://localhost:6333")
    return model, qdrant

model, qdrant = load_models()

query = st.text_input("What would you like to ask?", "Did anyone buy Bitcoin at $62,000?")

if st.button("Search Memory"):
    if not api_key:
        st.error("👈 Please enter a free Gemini API key in the sidebar!")
    else:
        genai.configure(api_key=api_key)
        with st.spinner("Searching..."):
            try:
                # 1. RETRIEVAL
                question_vector = model.encode(query).tolist()
                search_results = qdrant.query_points(
                    collection_name="crypto_memory",
                    query=question_vector,
                    limit=3
                )
                market_context = "\n".join([f"- {hit.payload['text']}" for hit in search_results.points])
                
                # 2. GENERATION
                prompt = f"Answer the user question based on context.\nContext: {market_context}\nUser: {query}\nAnswer in one sentence:"
                
                # DEBUGGING: Try to list models to see what is actually available
                models_list = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
                
                # TRY A RELIABLE MODEL NAME
                llm = genai.GenerativeModel('models/gemini-3.5-flash')
                response = llm.generate_content(prompt)
                st.success(response.text)
                
            except Exception as e:
                st.error(f"System Error: {e}")
                st.write("---")
                st.write("Diagnostic Info: Available Models for your key:")
                st.write(models_list)
                st.write("Tip: Try installing the latest library: `pip install -U google-generativeai`")