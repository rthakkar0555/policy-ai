"""Run the OpenAI-only RAG API."""

import os
from dotenv import load_dotenv
import uvicorn
from src.agent.simple_api import app

if __name__ == "__main__":
    # Load environment variables from .env file
    load_dotenv()
    print("🚀 Starting OpenAI + Qdrant Cloud RAG API")
    print("=" * 50)
    print("SERVICES:")
    print("  ✅ OpenAI API (embeddings + LLM)")
    print("  ✅ Qdrant Cloud (vector storage)")
    print("=" * 50)
    print("ENDPOINTS:")
    print("  POST /ingest - Upload PDF files")
    print("  POST /query - Ask questions (LangGraph)")
    print("  GET / - Root endpoint")
    print("=" * 50)
    
    # Check for required API keys
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ ERROR: OPENAI_API_KEY not found!")
        print("Please set your OpenAI API key in .env file")
        exit(1)
    
    if not os.getenv("QDRANT_API_KEY"):
        print("❌ ERROR: QDRANT_API_KEY not found!")
        print("Please set your Qdrant Cloud API key in .env file")
        exit(1)
    
    print("✅ OpenAI API key found")
    print("✅ Qdrant Cloud API key found")
    print("✅ Starting server on http://localhost:8000")
    
    uvicorn.run("src.agent.simple_api:app", host="0.0.0.0", port=8000, reload=True)
