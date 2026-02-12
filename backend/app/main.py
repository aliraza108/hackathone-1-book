from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
import logging

# Load environment variables
load_dotenv()

# Import routers
from .routers import chat, embeddings

# Import services to initialize them
from .services.openai_gemini_client import OpenAIGeminiClient
from .services.rag import RAGPipeline
from .services.vectorstore import VectorStore
from .services.pdf_processor import PDFProcessor

# Import models to create tables
from .models.database import create_tables

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Initializing services...")
    create_tables()
    logger.info("Services initialized successfully")
    yield
    # Shutdown
    logger.info("Shutting down services...")

app = FastAPI(
    title="Physical AI & Humanoid Robotics - Backend API",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(chat.router, prefix="/api/chat", tags=["chat"])
app.include_router(embeddings.router, prefix="/api/embeddings", tags=["embeddings"])

@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "service": "Physical AI Textbook Backend"}

@app.get("/")
async def root():
    return {"message": "Physical AI & Humanoid Robotics Backend API"}

# Initialize services
gemini_client = OpenAIGeminiClient()
rag_pipeline = RAGPipeline()
vector_store = VectorStore()
pdf_processor = PDFProcessor()