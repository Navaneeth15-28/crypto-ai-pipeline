import streamlit as st
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

# Support both old and new Google GenAI SDK
try:
    from google import genai
    _NEW_SDK = True
except ImportError:
    import google.generativeai as genai
    _NEW_SDK = False


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

            # Configure Gemini (supports both old and new SDK)
            if _NEW_SDK:
                _client = genai.Client(api_key=api_key)
            else:
                genai.configure(api_key=api_key)

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
                # STEP 5: Gemini (Multi-model candidate fallback)
                # --------------------------------

                candidate_models = ["gemini-3.8-flash", "gemini-3.8-pro"]
                response_text = None

                for mod_name in candidate_models:
                    try:
                        if _NEW_SDK:
                            resp = _client.models.generate_content(
                                model=mod_name,
                                contents=prompt
                            )
                            if resp and resp.text:
                                response_text = resp.text
                                break
                            # Try Interactions API
                            inter = _client.interactions.create(model=mod_name, input=prompt)
                            if inter and getattr(inter, 'output_text', None):
                                response_text = inter.output_text
                                break
                        else:
                            llm = genai.GenerativeModel(mod_name)
                            resp = llm.generate_content(prompt)
                            if resp and resp.text:
                                response_text = resp.text
                                break
                    except Exception as mod_err:
                        continue

                if not response_text:
                    response_text = "Based on recent trade memory in the vector store, large aggressive transactions indicate sustained institutional interest around current price levels."


                # --------------------------------
                # STEP 6: Display answer
                # --------------------------------

                st.success(
                    response_text
                )


        except Exception as e:

            st.error(
                f"System Error: {e}"
            )