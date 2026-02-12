"""
Basic tests for the RAG pipeline
"""
import pytest
import asyncio
from backend.app.services.rag import RAGPipeline
from backend.app.services.qwen_client import QwenClient

@pytest.mark.asyncio
async def test_qwen_client():
    """Test the Qwen client"""
    client = QwenClient()
    
    # Test basic response
    messages = [
        {"role": "user", "content": "Say hello in one word."}
    ]
    
    response = await client.generate_response(messages, max_tokens=10)
    assert isinstance(response, str)
    assert len(response) > 0

@pytest.mark.asyncio
async def test_rag_pipeline():
    """Test the RAG pipeline with a simple query"""
    # Note: This test requires the vector store to be populated
    # For now, we'll just test the structure
    pipeline = RAGPipeline()
    
    # This would normally require embeddings to be present
    # For now, we'll skip the actual query but test initialization
    assert hasattr(pipeline, 'qwen')
    assert hasattr(pipeline, 'vectorstore')
    assert hasattr(pipeline, 'openai')