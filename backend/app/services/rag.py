"""
RAG pipeline service for retrieving and generating responses
"""
from typing import List, Dict
from .openai_gemini_client import OpenAIGeminiClient
from .vectorstore import VectorStore
from openai import AsyncOpenAI
from ..config import Config
import asyncio
import logging

logger = logging.getLogger(__name__)

class RAGPipeline:
    def __init__(self):
        self.gemini = OpenAIGeminiClient()
        self.vectorstore = VectorStore()
        self._openai_client = None  # Lazy initialization for embeddings
        self.chunk_size = Config.CHUNK_SIZE
        self.top_k = Config.TOP_K_RESULTS

    @property
    def openai(self):
        if self._openai_client is None:
            from openai import AsyncOpenAI
            from ..config import Config
            self._openai_client = AsyncOpenAI(api_key=Config.OPENAI_API_KEY)
        return self._openai_client

    async def query(self, user_query: str) -> Dict:
        """Execute RAG query pipeline"""
        try:
            # Step 1: Embed user query
            query_embedding = await self._embed_text(user_query)

            # Step 2: Vector search for relevant chunks
            results = await self.vectorstore.search(
                query_embedding,
                top_k=self.top_k
            )

            # Step 3: Prepare context from retrieved chunks
            context = self._format_context(results)

            # Step 4: Generate response with Gemini via OpenAI SDK
            response = await self.gemini.generate_rag_response(
                user_query,
                context
            )

            # Step 5: Return response with sources
            return {
                "answer": response,
                "sources": [
                    {
                        "chapter": r["metadata"]["chapter"],
                        "section": r["metadata"]["section"],
                        "page": r["metadata"]["page"],
                        "score": r["score"]
                    }
                    for r in results
                ]
            }
        except Exception as e:
            logger.error(f"Error in RAG pipeline: {e}")
            raise

    async def _embed_text(self, text: str) -> List[float]:
        """Generate embedding using OpenAI"""
        try:
            response = await self.openai.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise

    def _format_context(self, results: List[Dict]) -> str:
        """Format retrieved chunks into context string"""
        context_parts = []
        for i, result in enumerate(results, 1):
            context_parts.append(
                f"[Source {i} - {result['metadata']['chapter']}]\n"
                f"{result['text']}\n"
            )
        return "\n".join(context_parts)