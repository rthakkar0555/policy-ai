"""Simple document processor - PDF only."""

import logging
from typing import List
from pathlib import Path

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader

logger = logging.getLogger(__name__)


class SimpleDocumentProcessor:
    """Simple PDF processor."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """Initialize the processor."""
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
            length_function=len,
        )
    
    def process_pdf(self, file_path: str) -> List[Document]:
        """Process a PDF file."""
        try:
            # Load PDF
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            
            if not documents:
                return []
            
            # Add metadata
            for i, doc in enumerate(documents):
                if not hasattr(doc, 'metadata') or doc.metadata is None:
                    doc.metadata = {}
                doc.metadata.update({
                    "source": file_path,
                    "type": "pdf",
                    "page_label": f"Page {i+1}"
                })
            
            # Split into chunks
            chunks = self.text_splitter.split_documents(documents)
            
            logger.info(f"Processed PDF: {len(documents)} pages -> {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            logger.error(f"Error processing PDF {file_path}: {str(e)}")
            raise
