import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader

try:
    from langchain.chains.retrieval import create_retrieval_chain
    from langchain.chains.combine_documents import create_stuff_documents_chain
except ImportError:
    from langchain_classic.chains.retrieval import create_retrieval_chain
    from langchain_classic.chains.combine_documents import create_stuff_documents_chain

from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

def setup_rag_pipeline():
    print("1. Initializing Embeddings...")
    embeddings_model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    index_path = "faiss_index"

    # Self-Healing Logic with Empty Check
    if os.path.exists(index_path):
        print("Loading existing FAISS index...")
        vector_store = FAISS.load_local(index_path, embeddings_model, allow_dangerous_deserialization=True)
    else:
        print("Index not found. Attempting to build from 'data/' folder...")
        if os.path.exists("data/") and os.listdir("data/"):
            loader = PyPDFDirectoryLoader("data/")
            docs = loader.load()
            if not docs:
                raise ValueError("The 'data/' folder exists but contains no readable PDF documents!")
            
            print(f"Building index from {len(docs)} document(s)...")
            vector_store = FAISS.from_documents(docs, embeddings_model)
            vector_store.save_local(index_path)
            print("New index built successfully.")
        else:
            raise FileNotFoundError("The 'data/' folder is empty or missing. Please upload your PDFs to a folder named 'data/' at the root of your project.")

    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)
    
    system_prompt = (
        "You are LexiHR Copilot. Your ONLY task is to answer employee queries strictly based on the provided HR Policy context.\n"
        "Context: {context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    rag_chain = create_retrieval_chain(retriever, create_stuff_documents_chain(llm, prompt))
    return rag_chain
```

### Critical Next Steps:
1.  **Check your folder structure in GitHub:**
    * Make sure the `data` folder is at the **exact same level** as `app.py`.
    * Make sure your HR PDF files are **directly inside** that `data` folder.
2.  **Push the update:**
    ```cmd
    git add rag_chain.py
    git commit -m "Add empty document safety check to rag_chain"
    git push -u origin main