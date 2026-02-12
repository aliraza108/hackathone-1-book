"""
Gemini client service for interacting with Google's Gemini API using OpenAI-compatible interface
"""
import os
from typing import List, Dict, Optional
from openai import AsyncOpenAI
from google import genai
from ..config import Config
import logging
import asyncio

logger = logging.getLogger(__name__)

class GeminiClient:
    def __init__(self):
        # Initialize Google Generative AI
        self.google_api_key = os.getenv("GEMINI_API_KEY") or os.getenv("VKEY")
        if not self.google_api_key:
            raise ValueError("GEMINI_API_KEY or VKEY environment variable is required")
        
        genai.configure(api_key=self.google_api_key)
        
        # Initialize the model
        self.model_name = os.getenv("GEMINI_MODEL", "gemini-pro")
        self.model = genai.GenerativeModel(self.model_name)
        
        # Also initialize OpenAI client for compatibility
        self.openai_client = AsyncOpenAI(
            api_key=os.getenv("VKEY"),  # Using VKEY as the API key
            base_url=os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
        )
        
        self.temperature = Config.TEMPERATURE
        self.max_tokens = Config.MAX_TOKENS

    async def generate_response(
        self,
        messages: List[Dict[str, str]],
        temperature: float = None,
        max_tokens: int = None
    ) -> str:
        """Generate response using Gemini API"""
        if temperature is None:
            temperature = self.temperature
        if max_tokens is None:
            max_tokens = self.max_tokens

        try:
            # Convert messages to Gemini format
            gemini_messages = []
            for msg in messages:
                role = "model" if msg["role"] == "assistant" else "user"
                gemini_messages.append({
                    "role": role,
                    "parts": [{"text": msg["content"]}]
                })

            # Generate content using Gemini
            generation_config = {
                "temperature": temperature,
                "max_output_tokens": max_tokens,
            }

            response = await self.model.generate_content_async(
                contents=gemini_messages,
                generation_config=generation_config
            )

            return response.text

        except Exception as e:
            logger.error(f"Error occurred while calling Gemini API: {e}")
            raise

    async def generate_rag_response(
        self,
        query: str,
        context: str
    ) -> str:
        """Generate RAG response with context using Gemini"""
        system_msg = """You are an expert AI teaching assistant for Physical AI and Humanoid Robotics.
Use the provided textbook context to answer questions accurately and comprehensively.
Always cite specific chapters/sections when referencing content.
If information is not in the context, clearly state that."""

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
        ]

        return await self.generate_response(messages)

    async def generate_response_openai_compatible(
        self,
        messages: List[Dict[str, str]],
        temperature: float = None,
        max_tokens: int = None
    ) -> str:
        """Generate response using OpenAI-compatible interface for Gemini"""
        if temperature is None:
            temperature = self.temperature
        if max_tokens is None:
            max_tokens = self.max_tokens

        try:
            # Use the OpenAI client which is configured to work with Gemini's OpenAI-compatible endpoint
            response = await self.openai_client.chat.completions.create(
                model=self.model_name.replace("gemini-", "gemini-"),  # Adjust model name if needed
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Error occurred while calling Gemini API (OpenAI compatible): {e}")
            raise