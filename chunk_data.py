from langchain_text_splitters import RecursiveCharacterTextSplitter
# Why are we importing from ingest? Because modular code is clean, enterprise code!
from ingest import load_hr_policies

def create_chunks(documents, chunk_size: int = 400, chunk_overlap: int = 50):
    """
    Slices LangChain Document objects into smaller, overlapping semantic chunks
    using a recursive character strategy.
    """
    # 1. Initialize the Recursive Splitter with our defensive parameters
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        # The separators hierarchy: Paragraphs -> Lines -> Spaces -> Characters
        separators=["\n\n", "\n", " ", ""]
    )
    
    # 2. Execute the splitting across all loaded documents
    chunks = text_splitter.split_documents(documents)
    
    return chunks

# --- Execution & Verification ---
if __name__ == "__main__":
    print("Step 1: Loading raw HR policies from disk...")
    raw_docs = load_hr_policies("data")
    
    print("Step 2: Slicing documents into semantic chunks...")
    # We pass our loaded documents into our chunking function
    policy_chunks = create_chunks(raw_docs, chunk_size=400, chunk_overlap=50)
    
    print(f"\n[Success] Sliced {len(raw_docs)} raw document into {len(policy_chunks)} distinct chunks!")
    
    # Let's inspect Chunk 0 and Chunk 1 to actually SEE the overlap in action
    print("\n--- Inspecting Chunk 0 ---")
    print(f"Content:\n{policy_chunks[0].page_content}")
    print(f"\nMetadata: {policy_chunks[0].metadata}")
    
    print("\n" + "="*50 + "\n")
    
    print("--- Inspecting Chunk 1 (Notice the overlapping text from Chunk 0!) ---")
    print(f"Content:\n{policy_chunks[1].page_content}")
    print(f"\nMetadata: {policy_chunks[1].metadata}")
    
    print("\nMilestone 2 (Part 1) Complete: Documents are successfully chunked and ready for embedding!")