# SQLite Fix for Streamlit Cloud
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import os
import shutil
from pdf_processor import PDFProcessor
from rag_chain import RAGChain

# Page Configuration
st.set_page_config(
    page_title="PDF RAG Chat",
    page_icon="📚",
    layout="wide"
)

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []
if "processed_files" not in st.session_state:
    st.session_state.processed_files = []
if "pdf_processor" not in st.session_state:
    st.session_state.pdf_processor = PDFProcessor()

# Initialize RAG Chain (Lazy load or update on provider change)
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

def clear_database():
    """
    Clears the vector database and resets session state.
    """
    if os.path.exists("./chroma_db"):
        shutil.rmtree("./chroma_db")
    
    st.session_state.rag_chain = None # Force re-init
    st.session_state.messages = []
    st.session_state.processed_files = []
    st.success("Database cleared successfully!")

# Sidebar
with st.sidebar:
    st.title("⚙️ Configuration")
    
    # Provider Selection
    provider = st.selectbox(
        "Select Model Provider",
        ["Ollama (Local)", "OpenAI (Cloud)", "Gemini (Cloud)"]
    )
    
    api_key = None
    if provider == "OpenAI (Cloud)":
        api_key = st.text_input("Enter OpenAI API Key", type="password")
        provider_name = "OpenAI"
    elif provider == "Gemini (Cloud)":
        default_key = st.secrets.get("GEMINI_API_KEY", "")
        api_key = st.text_input("Enter Google API Key", value=default_key, type="password")
        provider_name = "Gemini"
    else:
        provider_name = "Ollama"
        
    # Initialize/Update Chain
    if st.button("Initialize / Update Agent"):
        try:
            st.session_state.rag_chain = RAGChain(provider=provider_name, api_key=api_key)
            st.success(f"Initialized with {provider_name}!")
        except Exception as e:
            st.error(f"Failed to initialize: {e}")

    st.divider()

    st.title("📄 Document Manager")
    
    uploaded_files = st.file_uploader(
        "Upload PDF Files", 
        type=["pdf"], 
        accept_multiple_files=True
    )
    
    if st.button("Process Documents"):
        if not st.session_state.rag_chain:
            st.error("Please initialize the agent first!")
        elif uploaded_files:
            with st.spinner("Processing documents..."):
                # Check for new files
                new_files = [f for f in uploaded_files if f.name not in st.session_state.processed_files]
                
                if new_files:
                    # Process files
                    chunks = st.session_state.pdf_processor.process_pdfs(new_files)
                    
                    # Update Vector Store
                    st.session_state.rag_chain.initialize_vectorstore(chunks)
                    
                    # Update processed files list
                    st.session_state.processed_files.extend([f.name for f in new_files])
                    
                    st.success(f"Indexed {len(new_files)} documents with {len(chunks)} chunks!")
                else:
                    st.info("No new files to process.")
        else:
            st.warning("Please upload at least one PDF.")

    st.divider()
    
    if st.button("Clear Database", type="primary"):
        clear_database()

# Main Chat Interface
st.title("💬 Chat with your PDFs")

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        if st.session_state.rag_chain:
            with st.spinner("Thinking..."):
                response = st.session_state.rag_chain.get_response(prompt)
                st.markdown(response)
                
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})
        else:
            st.error("Agent not initialized. Please verify configuration in sidebar.")
