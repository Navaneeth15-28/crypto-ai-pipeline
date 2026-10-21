import streamlit as st
import google.generativeai as genai
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer


# 1. Setup Page Config
st.set_page_config(
    page_title="Crypto AI Pipeline",
    page_icon="📈"
)

st.title("📈 Crypto-AI Live Search")


# 2. Sidebar for API Key
st.sidebar.header("Configuration")

api_key = st.sidebar.text_input(
    "Enter Gemini API Key",
    type="password"
)


# 3. Load AI model and Qdrant
@st.cache_resource
def load_models():

    model = SentenceTransformer("all-MiniLM-L6-v2")

    qdrant = QdrantClient(
        "http://localhost:6333"
    )

    return model, qdrant


model, qdrant = load_models()


# 4. User question
query = st.text_input(
    "What would you like to ask?",
    "Did anyone buy Bitcoin at $62,000?"
)


# 5. Search button
if st.button("Search Memory"):

    if not api_key:

        st.error(
            "👉 Please enter a Gemini API key in the sidebar!"
        )

    else:

        try:

            # Configure Gemini
            genai.configure(
                api_key=api_key
            )

            with st.spinner("Searching..."):

                # --------------------------------
                # STEP 1: Convert question to vector
                # --------------------------------

                question_vector = model.encode(
                    query
                ).tolist()


                # --------------------------------
                # STEP 2: Search Qdrant
                # --------------------------------

                search_results = qdrant.search(
                    collection_name="crypto_memory",
                    query_vector=question_vector,
                    limit=3
                )


                # --------------------------------
                # STEP 3: Build market context
                # --------------------------------

                if search_results:

                    market_context = "\n".join(
                        [
                            f"- {hit.payload['text']}"
                            for hit in search_results
                        ]
                    )

                else:

                    market_context = (
                        "No matching trade data was found."
                    )


                # --------------------------------
                # STEP 4: Create prompt
                # --------------------------------

                prompt = f"""
Answer the user's question based only on the
crypto trade context below.

Crypto Trade Context:
{market_context}

User Question:
{query}

Give a short and clear answer.
"""


                # --------------------------------
                # STEP 5: Gemini
                # --------------------------------

                llm = genai.GenerativeModel(
                    "gemini-pro"
                )

                response = llm.generate_content(
                    prompt
                )


                # --------------------------------
                # STEP 6: Display answer
                # --------------------------------

                st.success(
                    response.text
                )


        except Exception as e:

            st.error(
                f"System Error: {e}"
            )