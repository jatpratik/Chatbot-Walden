import os
import uvicorn
import logging
from typing import Dict, List, Optional
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import uuid
import time
from data_science_rag import WaldenRAGChain

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    filename='walden_api.log')
logger = logging.getLogger('walden_api')

# Initialize the app
app = FastAPI(
    title="Walden University Chatbot API",
    description="API for the Walden University RAG-powered chatbot",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models
class ChatMessage(BaseModel):
    message: str
    user_id: Optional[str] = None
    conversation_id: Optional[str] = None


class ChatResponse(BaseModel):
    answer: str
    sources: List[Dict[str, str]]
    conversation_id: str
    timestamp: float


# Global variables
rag_chain = None


# Dependency to get the RAG chain
def get_rag_chain():
    global rag_chain
    if rag_chain is None:
        try:
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if not openai_api_key:
                raise ValueError("OpenAI API key not found")

            # Get Weaviate URL from environment or use default
            weaviate_url = "http://weaviate:8080"
            
            # Initialize RAG chain
            rag_chain = WaldenRAGChain(
                weaviate_url=weaviate_url,
                openai_api_key=openai_api_key
            )
            logger.info(f"RAG chain initialized successfully with Weaviate URL: {weaviate_url}")
        except Exception as e:
            logger.error(f"Failed to initialize RAG chain: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to initialize RAG chain: {str(e)}")

    return rag_chain


# Routes
@app.get("/")
async def root():
    return {"message": "Welcome to the Walden University Chatbot API"}


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": time.time()
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatMessage, rag=Depends(get_rag_chain)):
    try:
        # Generate or use provided IDs
        user_id = request.user_id or str(uuid.uuid4())
        conversation_id = request.conversation_id or str(uuid.uuid4())

        # Process the message
        response = rag.handle_chat(
            message=request.message,
            user_id=user_id,
            conversation_id=conversation_id
        )

        # Format the response
        return {
            "answer": response["formatted_answer"],
            "sources": response["sources"],
            "conversation_id": conversation_id,
            "timestamp": time.time()
        }

    except Exception as e:
        logger.error(f"Error processing chat request: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing chat request: {str(e)}")


@app.on_event("startup")
async def startup_event():
    try:
        # Initialize the RAG chain on startup
        get_rag_chain()
    except Exception as e:
        logger.error(f"Startup initialization failed: {e}")


def start():
    """Start the API server"""
    uvicorn.run("walden_api:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    start()