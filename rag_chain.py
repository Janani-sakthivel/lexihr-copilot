import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader
from langchain_core.prompts import ChatPromptTemplate

# Handle library version differences
try:
    from langchain.chains.retrieval import create_retrieval_chain
    from langchain.chains.combine_documents import create_stuff_documents_chain
except ImportError:
    from langchain_classic.chains.retrieval import create_retrieval_chain
    from langchain_classic.chains.combine_documents import create_stuff_documents_chain

load_dotenv()

def setup_rag_pipeline():
    embeddings_model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    index_path = "faiss_index"

    if os.path.exists(index_path):
        vector_store = FAISS.load_local(index_path, embeddings_model, allow_dangerous_deserialization=True)
    else:
        # Use DirectoryLoader to load your .md file from the data folder
        loader = DirectoryLoader("data/", glob="**/*.md", loader_cls=UnstructuredMarkdownLoader)
        docs = loader.load()
        
        if not docs:
            raise ValueError("No Markdown files found in data/ folder.")
            
        vector_store = FAISS.from_documents(docs, embeddings_model)
        vector_store.save_local(index_path)

    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)
    
    system_prompt = (
        "You are LexiHR Copilot. Answer queries strictly based on the provided HR Policy.\n"
        "Context: {context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    return create_retrieval_chain(retriever, create_stuff_documents_chain(llm, prompt))

if __name__ == "__main__":
    rag_bot = setup_rag_pipeline()
    response = rag_bot.invoke({"input": "What is the maternity leave policy?"})
    print(f"🤖 Copilot: {response['answer']}")