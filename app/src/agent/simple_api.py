"""Simple RAG API - PDF ingestion (simple) + Query (LangGraph)."""

import os
import logging
from typing import List
import tempfile

from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
import uvicorn

from .simple_processor import SimpleDocumentProcessor
from .simple_vector_store import SimpleVectorStore

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Simple RAG API", version="1.0.0")

# Global instances
document_processor = None
vector_store_manager = None


class QueryRequest(BaseModel):
    """Request model for queries."""
    query: str


class QueryResponse(BaseModel):
    """Response model for queries."""
    answer: str


def get_document_processor() -> SimpleDocumentProcessor:
    """Get document processor instance."""
    global document_processor
    if document_processor is None:
        document_processor = SimpleDocumentProcessor()
    return document_processor


def get_vector_store_manager() -> SimpleVectorStore:
    """Get vector store manager instance."""
    global vector_store_manager
    if vector_store_manager is None:
        vector_store_manager = SimpleVectorStore(
            collection_name=os.getenv("QDRANT_COLLECTION", "documents"),
            qdrant_url=os.getenv("QDRANT_URL"),
            qdrant_api_key=os.getenv("QDRANT_API_KEY"),
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
    return vector_store_manager


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    logger.info("Starting Simple RAG API...")
    get_document_processor()
    get_vector_store_manager()
    logger.info("Services initialized successfully")


@app.post("/ingest")
async def ingest_pdf(file: UploadFile = File(...)):
    """Ingest a PDF file with improved timeout handling."""
    temp_file_path = None
    try:
        # Check if it's a PDF
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        logger.info(f"Starting PDF processing for {file.filename}")
        
        # Process the PDF
        processor = get_document_processor()
        documents = processor.process_pdf(temp_file_path)
        
        if not documents:
            raise HTTPException(status_code=400, detail="No content found in the PDF")
        
        logger.info(f"PDF processed into {len(documents)} chunks, starting vector store upload")
        
        # Add to vector store with batch processing
        vector_store = get_vector_store_manager()
        vector_store.add_documents(documents)
        
        logger.info(f"Successfully ingested {file.filename}")
        
        return {
            "success": True,
            "message": f"Successfully ingested {file.filename}",
            "chunks": len(documents)
        }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error ingesting PDF: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")
    finally:
        # Clean up temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)


@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Query the documents using LangGraph."""
    try:
        # Use LangGraph for agentic query processing
        from .graph import graph, State
        
        # Create state for LangGraph (only query needed, everything else done in graph)
        state = State(query=request.query)
        
        # Run LangGraph pipeline (handles retrieval, context, and answer generation)
        result = await graph.ainvoke(state)
        
        # Debug: Print the result structure
        logger.info(f"LangGraph result type: {type(result)}")
        logger.info(f"LangGraph result: {result}")
        
        # Extract answer from result
        if hasattr(result, "answer"):
            answer = result.answer
        elif isinstance(result, dict):
            answer = result.get("answer", "No answer generated")
        else:
            answer = "No answer generated"
        
        # Check for errors
        if hasattr(result, "error") and result.error:
            raise HTTPException(status_code=400, detail=result.error)
        elif isinstance(result, dict) and result.get("error"):
            raise HTTPException(status_code=400, detail=result.get("error"))
        
        return QueryResponse(answer=answer)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error querying documents: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error querying documents: {str(e)}")


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Simple RAG API - Upload PDFs and ask questions"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
