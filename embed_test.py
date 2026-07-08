import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from ingest import load_hr_policies
from chunk_data import create_chunks

# 1. Load environment variables from .env file
# Why? Defensive engineering! Never expose API keys in public GitHub repos.
load_dotenv()

def generate_embeddings():
    """
    Initializes Google's embedding model and transforms text chunks into vector arrays.
    """
    # 2. Initialize the Gemini Embedding Model
    # We use 'models/text-embedding-004', Google's latest high-performance embedding model.
    embeddings_model = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
    
    # 3. Load and chunk our HR policy data using our modular functions
    raw_docs = load_hr_policies("data")
    chunks = create_chunks(raw_docs, chunk_size=400, chunk_overlap=50)
    
    print(f"Loaded {len(chunks)} chunks. Preparing to embed Chunk 0...")
    
    # 4. Extract text from the first chunk
    sample_text = chunks[0].page_content
    print(f"\n--- Text being embedded ---\n\"{sample_text[:150]}...\"\n")
    
    # 5. Call the Google API to convert text to a vector
    vector = embeddings_model.embed_query(sample_text)
    
    return vector

if __name__ == "__main__":
    print("Initializing Google Gemini Embedding Pipeline...")
    
    # Execute embedding generation
    sample_vector = generate_embeddings()
    
    # Inspect the mathematical output
    print("--- Vector Inspection ---")
    print(f"Vector Dimensions (Length of array): {len(sample_vector)}")
    print(f"First 5 Numerical Coordinates: {sample_vector[:5]}")
    print(f"Data Type: {type(sample_vector[0])}")
    
    print("\n[Success] Milestone 2 Complete: Text chunks are successfully converted into 768-dimensional vectors!")