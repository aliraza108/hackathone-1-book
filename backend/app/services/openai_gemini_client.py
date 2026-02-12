"""
OpenAI-compatible Gemini service using OpenAI agent SDK
"""
import os
from typing import List, Dict, Optional
from openai import OpenAI, AsyncOpenAI
from ..config import Config
import logging

logger = logging.getLogger(__name__)

class OpenAIGeminiClient:
    def __init__(self):
        # Use VKEY from environment as the API key
        api_key = os.getenv("VKEY")
        if not api_key:
            raise ValueError("VKEY environment variable is required")
        
        # Use the Gemini base URL from environment or default
        base_url = os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
        
        # Initialize OpenAI client with Gemini base URL
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        
        self.async_client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url
        )
        
        # Default model - can be overridden
        self.default_model = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
        
        self.temperature = Config.TEMPERATURE
        self.max_tokens = Config.MAX_TOKENS

    def generate_response_sync(
        self,
        messages: List[Dict[str, str]],
        model: str = None,
        temperature: float = None,
        max_tokens: int = None
    ) -> str:
        """Generate response using OpenAI-compatible Gemini API (sync)"""
        if model is None:
            model = self.default_model
        if temperature is None:
            temperature = self.temperature
        if max_tokens is None:
            max_tokens = self.max_tokens

        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Error occurred while calling Gemini API: {e}")
            raise

    async def generate_response(
        self,
        messages: List[Dict[str, str]],
        model: str = None,
        temperature: float = None,
        max_tokens: int = None
    ) -> str:
        """Generate response using OpenAI-compatible Gemini API (async)"""
        if model is None:
            model = self.default_model
        if temperature is None:
            temperature = self.temperature
        if max_tokens is None:
            max_tokens = self.max_tokens

        try:
            response = await self.async_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Error occurred while calling Gemini API: {e}")
            raise

    async def generate_rag_response(
        self,
        query: str,
        context: str
    ) -> str:
        """Generate RAG response with context using OpenAI-compatible Gemini API"""
        system_msg = """You are an expert AI teaching assistant for Physical AI and Humanoid Robotics.
Use the provided textbook context to answer questions accurately and comprehensively.
Always cite specific chapters/sections when referencing content.
If information is not in the context, clearly state that."""

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
        ]

        return await self.generate_response(messages)