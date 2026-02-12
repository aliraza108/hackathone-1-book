"""
Embeddings router for handling embedding-related endpoints
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
from ..services.pdf_processor import PDFProcessor
from ..models.schemas import GenerateEmbeddingsRequest, GenerateEmbeddingsResponse, StatusResponse
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize PDF processor
pdf_processor = PDFProcessor()

@router.post("/generate", response_model=GenerateEmbeddingsResponse)
async def generate_embeddings(request: GenerateEmbeddingsRequest):
    try:
        # Process the PDF and generate embeddings
        result = await pdf_processor.process_pdf(request.pdf_path)
        return GenerateEmbeddingsResponse(
            status="success",
            chunks_processed=result["chunks_processed"],
            message=f"Successfully processed {result['chunks_processed']} content chunks"
        )
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating embeddings: {str(e)}")

@router.get("/status", response_model=StatusResponse)
async def get_embeddings_status():
    try:
        status = await pdf_processor.get_processing_status()
        return StatusResponse(**status)
    except Exception as e:
        logger.error(f"Error getting status: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting status: {str(e)}")