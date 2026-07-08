import streamlit as st
from rag_chain import setup_rag_pipeline

# 1. Page Configuration
st.set_page_config(page_title="LexiHR Copilot", page_icon="🏢", layout="centered")
st.title("🏢 LexiHR Copilot")
st.markdown("Ask me anything about Leave Policies, WFH Allowances, or the Code of Conduct!")

# 2. Initialize the AI Brain (Cached so it doesn't reload FAISS on every click)
@st.cache_resource
def load_brain():
    return setup_rag_pipeline()

rag_bot = load_brain()

# 3. Setup Session State for Chat History
# This tells Streamlit to remember the conversation so it doesn't disappear when you type a new question
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am your AI HR Assistant. How can I help you today?"}
    ]

# 4. Display previous chat messages on the screen
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 5. The Chat Input Box (Where the user types)
if prompt := st.chat_input("E.g., What is the moonlighting policy?"):
    
    # Show user message on screen and save to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Show the AI thinking and answering
    with st.chat_message("assistant"):
        with st.spinner("Searching HR Policies..."):
            try:
                # Send the question to our FAISS/Gemini engine!
                response = rag_bot.invoke({"input": prompt})
                answer = response["answer"]
                
                # Display the answer
                st.markdown(answer)
                
                # Show exactly which file the AI got the answer from (Citation)
                with st.expander("View Source Document"):
                    st.write(f"Source: `{response['context'][0].metadata['source']}`")
                
                # Save AI answer to history
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
            except Exception as e:
                st.error(f"An error occurred: {e}")