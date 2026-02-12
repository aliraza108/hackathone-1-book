"""
Qwen client service for interacting with the Qwen API
"""
import os
import httpx
from typing import List, Dict, Optional
from ..config import Config
import asyncio
import logging

logger = logging.getLogger(__name__)

class QwenClient:
    def __init__(self):
        self.api_key = Config.QWEN_API_KEY
        self.endpoint = Config.QWEN_API_ENDPOINT
        self.model = Config.QWEN_MODEL
        self.timeout = httpx.Timeout(30.0, connect=5.0)
    
    async def generate_response(
        self, 
        messages: List[Dict[str, str]], 
        temperature: float = 0.7,
        max_tokens: int = 2048
    ) -> str:
        """Generate response using Qwen API"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "input": {
                "messages": messages
            },
            "parameters": {
                "temperature": temperature,
                "max_tokens": max_tokens,
                "top_p": 0.9
            }
        }
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    self.endpoint,
                    headers=headers,
                    json=payload
                )
                
                response.raise_for_status()
                result = response.json()
                
                # Extract the text from the response
                # The exact structure depends on the API response format
                if "output" in result and "choices" in result["output"]:
                    return result["output"]["choices"][0]["message"]["content"]
                elif "output" in result and "text" in result["output"]:
                    return result["output"]["text"]
                else:
                    raise Exception(f"Unexpected response format: {result}")
                    
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error occurred while calling Qwen API: {e}")
            raise
        except Exception as e:
            logger.error(f"Error occurred while calling Qwen API: {e}")
            raise
    
    async def generate_rag_response(
        self, 
        query: str, 
        context: str
    ) -> str:
        """Generate RAG response with context"""
        system_msg = """You are an expert AI teaching assistant for Physical AI and Humanoid Robotics.
Use the provided textbook context to answer questions accurately and comprehensively.
Always cite specific chapters/sections when referencing content.
If information is not in the context, clearly state that."""
        
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
        ]
        
        return await self.generate_response(messages)