import os
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

# Modular imports: Reusing our hard work from yesterday!
from ingest import load_hr_policies
from chunk_data import create_chunks

# 1. Load API keys defensively
load_dotenv()

# Define where we will save our database on the hard drive
FAISS_INDEX_PATH = "faiss_index"

def build_vector_store():
    """
    Ingests data, chunks it, generates embeddings, and saves the vectors to a local FAISS database.
    """
    print("1. Initializing Embedding Model (Gemini)...")
    embeddings_model = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
    
    print("2. Loading and Chunking HR Policies...")
    raw_docs = load_hr_policies("data")
    chunks = create_chunks(raw_docs, chunk_size=400, chunk_overlap=50)
    
    print(f"3. Building FAISS Vector Database with {len(chunks)} chunks...")
    # This is where the magic happens: FAISS maps text + metadata + vectors together!
    vector_store = FAISS.from_documents(chunks, embeddings_model)
    
    print("4. Saving FAISS index to local disk for permanent storage...")
    vector_store.save_local(FAISS_INDEX_PATH)
    
    return vector_store

# --- Execution & Verification ---
if __name__ == "__main__":
    print("--- Starting FAISS Vector Store Build ---")
    
    # Run the build function
    db = build_vector_store()
    
    print(f"\n[Success] Vector Database successfully built and saved to folder: '{FAISS_INDEX_PATH}'!")
    
    # --- The Ultimate Test: Semantic Search! ---
    print("\n--- Running a Semantic Search Test ---")
    
    # A user asks a question using normal words, not HR jargon
    test_query = "What happens if I get sick for 3 days?"
    print(f"Query: '{test_query}'")
    
    # FAISS performs mathematical similarity search and returns the top 2 closest chunks
    search_results = db.similarity_search(test_query, k=2)
    
    print("\nTop Match Found:")
    print(f"Content: {search_results[0].page_content}")
    print(f"Source Document: {search_results[0].metadata['source']}")