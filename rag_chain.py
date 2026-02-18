import os
from typing import List
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Provider Imports
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

class RAGChain:
    def __init__(self, db_path: str = "./chroma_db", provider: str = "Ollama", api_key: str = None):
        self.db_path = db_path
        self.provider = provider
        self.api_key = api_key
        
        self.vectorstore = None
        self.retriever = None
        self.chain = None
        
        # Initialize Models based on Provider
        self._initialize_models()
        
        # Load existing DB if available
        self._load_existing_db()

    def _initialize_models(self):
        """
        Initializes Embedding and LLM models based on the selected provider.
        """
        if self.provider == "Ollama":
            self.embedding_model = OllamaEmbeddings(model="nomic-embed-text")
            self.llm = ChatOllama(model="llama3")
            
        elif self.provider == "OpenAI":
            if not self.api_key:
                raise ValueError("OpenAI API Key is required.")
            self.embedding_model = OpenAIEmbeddings(openai_api_key=self.api_key)
            self.llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=self.api_key)
            
        elif self.provider == "Gemini":
            if not self.api_key:
                raise ValueError("Google API Key is required.")
            self.embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=self.api_key)
            self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=self.api_key)
            
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def _load_existing_db(self):
        """
        Loads the existing vector store if the directory exists.
        """
        if os.path.exists(self.db_path) and os.path.isdir(self.db_path):
            try:
                self.vectorstore = Chroma(
                    persist_directory=self.db_path,
                    embedding_function=self.embedding_model
                )
                self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})
                self._create_chain()
                print(f"Loaded existing vector store for {self.provider}.")
            except Exception as e:
                print(f"Could not load existing vector store: {e}")

    def initialize_vectorstore(self, documents: List[Document]):
        """
        Initializes or updates the ChromaDB vector store with documents.
        """
        if self.vectorstore:
            self.vectorstore.add_documents(documents)
        else:
            self.vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embedding_model,
                persist_directory=self.db_path
            )
        
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})
        self._create_chain()

    def clear_vectorstore(self):
        """
        Clears the vector store.
        """
        self.vectorstore = None
        self.retriever = None
        self.chain = None

    def _create_chain(self):
        """
        Creates the RAG chain.
        """
        template = """Answer the question based ONLY on the following context:
{context}

Question: {question}

If you don't know the answer based on the context, say "I don't know based on the provided documents."
Always include the source filename and page number for your answer if possible.
"""
        prompt = ChatPromptTemplate.from_template(template)

        def format_docs(docs):
            return "\n\n".join(f"[Source: {doc.metadata['source']}, Page: {doc.metadata['page']}]\n{doc.page_content}" for doc in docs)

        self.chain = (
            {"context": self.retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

    def get_response(self, query: str):
        """
        Generates a response for a given query.
        """
        if not self.chain:
            return "Please upload and process documents first."
        
        try:
            return self.chain.invoke(query)
        except Exception as e:
            return f"Error: Could not generate response. Details: {str(e)}"
