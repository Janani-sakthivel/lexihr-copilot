import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
# Change: Using DirectoryLoader to support Markdown files
from langchain_community.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader

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

    # Self-Healing Logic
    if os.path.exists(index_path):
        print("Loading existing FAISS index...")
        vector_store = FAISS.load_local(index_path, embeddings_model, allow_dangerous_deserialization=True)
    else:
        print("Index not found. Attempting to build from 'data/' folder...")
        if os.path.exists("data/") and os.listdir("data/"):
            # Change: Load Markdown files instead of PDFs
            loader = DirectoryLoader("data/", glob="**/*.md", loader_cls=UnstructuredMarkdownLoader)
            docs = loader.load()
            
            if not docs:
                raise ValueError("The 'data/' folder contains files, but they could not be loaded as Markdown!")
            
            print(f"Building index from {len(docs)} document(s)...")
            vector_store = FAISS.from_documents(docs, embeddings_model)
            vector_store.save_local(index_path)
            print("New index built successfully.")
        else:
            raise FileNotFoundError("The 'data/' folder is empty or missing.")

    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)
    
    system_prompt = (
        "You are LexiHR Copilot. Answer employee queries strictly based on the HR Policy context.\n"
        "Context: {context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    rag_chain = create_retrieval_chain(retriever, create_stuff_documents_chain(llm, prompt))
    return rag_chain

if __name__ == "__main__":
    rag_bot = setup_rag_pipeline()
    query = "How many days of privilege leave do I get?"
    response = rag_bot.invoke({"input": query})
    print(f"\n🤖 Copilot: {response['answer']}")
```

### What to do now (Final Steps):

1.  **Paste** the above code into `rag_chain.py` and **Save**.
2.  **Update `requirements.txt`**: Ensure `unstructured` is added to your `requirements.txt` (it is required to parse Markdown). Your `requirements.txt` should look like this:
    ```text
    streamlit
    langchain
    langchain-community
    langchain-google-genai
    langchain-core
    faiss-cpu
    unstructured
    python-dotenv
    ```
3.  **Push the changes:**
    ```cmd
    git add .
    git commit -m "Switch to Markdown loader and add unstructured dependency"
    git push -u origin main
    ```
4.  **Reboot the app** on Streamlit.

This time, the logs will show it correctly identifying the `.md` file and building the index without the `IndexError`. You are going to see the app working in just a moment!