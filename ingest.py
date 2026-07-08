import os
from langchain_community.document_loaders import TextLoader

def load_hr_policies(data_directory: str):
    """
    Loads unstructured markdown files from a local directory and converts them
    into LangChain Document objects with metadata.
    """
    file_path = os.path.join(data_directory, "company_hr_policy.md")
    
    # 1. Initialize the TextLoader
    # We specify encoding='utf-8' to prevent Windows/Mac character encoding crashes.
    loader = TextLoader(file_path, encoding='utf-8')
    
    # 2. Execute the load method
    # This reads the disk file and returns a list of Document objects.
    documents = loader.load()
    
    return documents

# --- Execution & Verification ---
if __name__ == "__main__":
    print("Initializing Data Ingestion Pipeline...")
    
    # Point to our data folder
    docs = load_hr_policies("data")
    
    # Let's verify what we actually loaded into memory
    print(f"\n[Success] Successfully loaded {len(docs)} document(s).")
    
    # Extract the first document to inspect its anatomy
    first_doc = docs[0]
    
    print("\n--- Inspecting Document Metadata ---")
    print(f"Source File: {first_doc.metadata['source']}")
    
    print("\n--- Inspecting Document Content (First 300 characters) ---")
    print(first_doc.page_content[:300] + "...\n")
    
    print("Milestone 1 Complete: Data is loaded into structured LangChain Document objects!")