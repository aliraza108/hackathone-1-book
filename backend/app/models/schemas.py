"""
Pydantic models for request/response validation
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime

# User models
class UserBase(BaseModel):
    email: str
    full_name: str


class UserCreate(UserBase):
    password: str


class User(UserBase):
    id: int
    created_at: datetime

    class Config:
        from_attributes = True


# Chat models
class ChatMessageBase(BaseModel):
    conversation_id: str
    query: str
    response: str
    sources: Optional[List[Dict]] = []


class ChatMessageCreate(ChatMessageBase):
    pass


class ChatMessage(ChatMessageBase):
    id: int
    user_id: int
    created_at: datetime

    class Config:
        from_attributes = True


# Embedding models
class EmbeddingRequest(BaseModel):
    text: str
    model: str = "text-embedding-3-small"


class EmbeddingResponse(BaseModel):
    embeddings: List[float]
    model: str


# RAG models
class RAGQuery(BaseModel):
    query: str
    top_k: int = 5
    conversation_id: Optional[str] = None


class RAGResponse(BaseModel):
    answer: str
    sources: List[Dict]
    conversation_id: str


# Selection query models
class SelectionQueryRequest(BaseModel):
    selected_text: str
    context: str
    conversation_id: Optional[str] = None


class SelectionQueryResponse(BaseModel):
    explanation: str
    sources: List[Dict]
    conversation_id: str


# PDF processing models
class GenerateEmbeddingsRequest(BaseModel):
    pdf_path: str


class GenerateEmbeddingsResponse(BaseModel):
    status: str
    chunks_processed: int
    message: str


class StatusResponse(BaseModel):
    status: str
    progress: int
    total_chunks: int
    processed_chunks: int


# Qwen test models
class QwenTestRequest(BaseModel):
    message: str


class QwenTestResponse(BaseModel):
    response: str
    success: bool


# Health check model
class HealthResponse(BaseModel):
    status: str
    service: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)