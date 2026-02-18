# PDF RAG Chat Application

A local RAG (Retrieval-Augmented Generation) application for chatting with PDF documents using Ollama (Llama 3) and Streamlit.

## Features

- **Multi-Document Support**: Upload and query multiple PDFs simultaneously.
- **Smart Chunking**: Uses recursive character splitting with overlap for better context retention.
- **Local Inference**: Uses Ollama for embeddings (`nomic-embed-text`) and generation (`llama3`).
- **Source Citations**: Answers include filenames and page numbers.
- **Persistent Database**: ChromaDB vector store is saved locally.

- **Persistent Database**: ChromaDB vector store is saved locally.

> [!TIP]
> **Deployment Options**:
> - **Local**: Run with Ollama for privacy/offline use.
> - **Cloud**: Deploy to **Streamlit Cloud** using **Gemini** or **OpenAI**.
>   - *Note: Ollama will NOT work on Streamlit Cloud.*

## Short Description
A powerful PDF Q&A application built with Streamlit and LangChain. Chat with your documents using **local LLMs (Ollama)** or **cloud models (Gemini 1.5 Flash, OpenAI)**. Features smart chunking, source citations, and persistent memory.

## Prerequisites

1.  **Python 3.8+**
2.  **Ollama**: Installed and running.
    - Download from [ollama.com](https://ollama.com/)
    - Pull required models:
      ```bash
      ollama pull llama3
      ollama pull nomic-embed-text
      ```

## Installation

1.  Clone the repository or navigate to the project folder.
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```

2.  **Upload PDFs**:
    - Go to the sidebar.
    - Click "Browse files" and select PDF documents.
    - Click "Process Documents".

3.  **Chat**:
    - Type your question in the chat input box.
    - The AI will answer based *only* on the uploaded documents.

## Example Query

**Query**: "What is the main conclusion of the 'Annual_Report.pdf' regarding sustainability?"

**Response**: 
> The annual report concludes that sustainability efforts have reduced carbon footprint by 20% compared to the previous fiscal year.
> 
> [Source: Annual_Report.pdf, Page: 42]

## Project Structure

- `app.py`: Main Streamlit application.
- `pdf_processor.py`: Handles PDF loading and text chunking.
- `rag_chain.py`: Manages the RAG pipeline (Retrieval + Generation).
- `requirements.txt`: Python dependencies.
- `chroma_db/`: Directory where vector embeddings are stored (created automatically).

## Deploy to Streamlit Cloud

1.  Push this code to GitHub.
2.  Go to [share.streamlit.io](https://share.streamlit.io/).
3.  Deploy the app by selecting your repository and `app.py`.
4.  **Important**: In the Streamlit Cloud dashboard, go to **App Settings** -> **Secrets** and add:
    ```toml
    GEMINI_API_KEY = "your-api-key-here"
    ```
5.  Launch! Select "Gemini (Cloud)" in the app sidebar.
