import io
from typing import List
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

class PDFProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 100):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )

    def load_pdf(self, file, filename: str) -> List[Document]:
        """
        Extracts text from a PDF file stream and returns a list of Document objects.
        """
        pdf_reader = PdfReader(file)
        documents = []
        
        for i, page in enumerate(pdf_reader.pages):
            text = page.extract_text()
            if text:
                doc = Document(
                    page_content=text,
                    metadata={"source": filename, "page": i + 1}
                )
                documents.append(doc)
                
        return documents

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Splits a list of Documents into chunks.
        """
        return self.text_splitter.split_documents(documents)

    def process_pdfs(self, uploaded_files) -> List[Document]:
        """
        Orchestrates loading and splitting for multiple uploaded files.
        """
        all_chunks = []
        for uploaded_file in uploaded_files:
            # uploaded_file is a Streamlit UploadedFile object, which behaves like a file object
            filename = uploaded_file.name
            raw_docs = self.load_pdf(uploaded_file, filename)
            chunks = self.split_documents(raw_docs)
            all_chunks.extend(chunks)
            
        return all_chunks
