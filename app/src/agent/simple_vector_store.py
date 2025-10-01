"""Simple vector store manager."""

import os
import logging
from typing import List

from langchain.schema import Document
from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings.openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http import models

logger = logging.getLogger(__name__)


class SimpleVectorStore:
    """Simple vector store for PDF documents."""
    
    def __init__(self, collection_name: str = "documents", qdrant_url: str = None, qdrant_api_key: str = None, openai_api_key: str = None):
        """Initialize the vector store."""
        self.collection_name = collection_name
        self.qdrant_url = qdrant_url or os.getenv("QDRANT_URL")
        self.qdrant_api_key = qdrant_api_key or os.getenv("QDRANT_API_KEY")
        
        # Initialize Qdrant client (cloud or local) with longer timeout
        if self.qdrant_api_key:
            # Qdrant Cloud
            self.client = QdrantClient(
                url=self.qdrant_url,
                api_key=self.qdrant_api_key,
                timeout=300  # 5 minutes timeout
            )
        else:
            # Local Qdrant
            self.client = QdrantClient(
                url=self.qdrant_url,
                timeout=300  # 5 minutes timeout
            )
        
        # Initialize OpenAI embeddings
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=openai_api_key,
            model="text-embedding-ada-002"
        )
        
        # Initialize vector store
        self.vector_store = None
        self._initialize_collection()
    
    def _initialize_collection(self):
        """Initialize or connect to the Qdrant collection."""
        try:
            # Check if collection exists
            collections = self.client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if self.collection_name not in collection_names:
                # Create collection
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=1536,  # OpenAI text-embedding-ada-002 embedding size
                        distance=models.Distance.COSINE
                    )
                )
                logger.info(f"Created collection: {self.collection_name}")
            else:
                logger.info(f"Using existing collection: {self.collection_name}")
            
            # Initialize vector store
            self.vector_store = Qdrant(
                client=self.client,
                collection_name=self.collection_name,
                embeddings=self.embeddings,
            )
            
        except Exception as e:
            logger.error(f"Error initializing collection: {str(e)}")
            raise
    
    def add_documents(self, documents: List[Document]):
        """Add documents to the vector store."""
        if not documents:
            return
        
        try:
            self.vector_store.add_documents(documents)
            logger.info(f"Added {len(documents)} documents to vector store")
        except Exception as e:
            logger.error(f"Error adding documents: {str(e)}")
            raise
    
    def similarity_search(self, query: str, k: int = 3) -> List[Document]:
        """Search for similar documents."""
        try:
            results = self.vector_store.similarity_search(query, k=k)
            logger.info(f"Found {len(results)} similar documents")
            return results
        except Exception as e:
            logger.error(f"Error in similarity search: {str(e)}")
            raise
