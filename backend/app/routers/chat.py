"""
Chat router for handling chat-related endpoints
"""
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Optional
from ..services.rag import RAGPipeline
from ..models.schemas import ChatMessageCreate, ChatMessage, SelectionQueryRequest, SelectionQueryResponse
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize RAG pipeline
rag_pipeline = RAGPipeline()

@router.post("/message", response_model=dict)
async def send_message(request: ChatMessageCreate):
    try:
        result = await rag_pipeline.query(request.query)
        return {
            "response": result["answer"],
            "conversation_id": request.conversation_id or "default_conversation",
            "sources": result["sources"]
        }
    except Exception as e:
        logger.error(f"Error processing message: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing message: {str(e)}")

@router.post("/selection", response_model=SelectionQueryResponse)
async def query_selection(request: SelectionQueryRequest):
    try:
        query = f"Explain this section: {request.selected_text}"
        if request.context:
            query += f"\nContext: {request.context}"
        
        result = await rag_pipeline.query(query)
        return SelectionQueryResponse(
            explanation=result["answer"],
            conversation_id=request.conversation_id or "default_conversation",
            sources=result["sources"]
        )
    except Exception as e:
        logger.error(f"Error processing selection: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing selection: {str(e)}")

@router.get("/history")
async def get_history(conversation_id: str = "default_conversation"):
    # This would typically fetch from a database
    # For now, returning a placeholder response
    return {"conversation_id": conversation_id, "history": []}