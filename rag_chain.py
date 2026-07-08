import os
from dotenv import load_dotenv

# Use the latest stable import patterns
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

# Unbreakable import block
try:
    from langchain.chains.retrieval import create_retrieval_chain
    from langchain.chains.combine_documents import create_stuff_documents_chain
except ImportError:
    from langchain_classic.chains.retrieval import create_retrieval_chain
    from langchain_classic.chains.combine_documents import create_stuff_documents_chain

from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

def setup_rag_pipeline():
    print("1. Loading FAISS Vector Database...")
    # UPDATED: Using the current production embedding model (gemini-embedding-2)
    embeddings_model = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2")
    
    vector_store = FAISS.load_local("faiss_index", embeddings_model, allow_dangerous_deserialization=True)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    
    print("2. Initializing Gemini LLM...")
    # UPDATED: Using the current production LLM (gemini-2.5-flash)
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)
    
    print("3. Building Guardrails...")
    system_prompt = (
        "You are LexiHR Copilot. Your ONLY task is to answer employee queries strictly based on the provided HR Policy context.\n"
        "STRICT RULES:\n"
        "1. If the answer is not in the context, say 'I cannot find this in the HR policy. Please contact HR.'\n"
        "2. Do NOT use outside knowledge.\n"
        "3. Keep it professional.\n\n"
        "Context: {context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    print("4. Chaining pieces...")
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    return rag_chain

if __name__ == "__main__":
    rag_bot = setup_rag_pipeline()
    query = "I am a new father. How many days of leave do I get?"
    response = rag_bot.invoke({"input": query})
    print(f"\n🤖 Copilot: {response['answer']}")