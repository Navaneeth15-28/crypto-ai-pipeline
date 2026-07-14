import warnings
# Ignore some HuggingFace warnings to keep the terminal clean
warnings.filterwarnings("ignore")

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

def search_crypto_memory():
    # 1. Connect to our Database and AI Model
    print("Loading AI Brain...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    qdrant = QdrantClient("http://localhost:6333")
    
    # 2. The question we want to ask
    question = "Did anyone buy Bitcoin for exactly $62,498?"
    print(f"\nSearching for: '{question}'")
    
    # 3. Convert our English question into a 384-length vector
    question_vector = model.encode(question).tolist()
    
    # 4. Do a mathematical similarity search in Qdrant!
    search_results = qdrant.query_points(
        collection_name="crypto_memory",
        query=question_vector,
        limit=3 # Bring back the top 3 closest matches
    )
    
    # 5. Print the results
    print("\n--- AI FOUND THESE MATCHES IN YOUR REAL-TIME STREAM ---")
    for hit in search_results.points:
        # hit.score is how confident the AI is (1.0 is a perfect match)
        print(f"[{hit.score:.4f}] {hit.payload['text']}")

if __name__ == "__main__":
    search_crypto_memory()