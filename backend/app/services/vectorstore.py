"""
Vector store service for managing embeddings
"""
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models
from typing import List, Dict
from ..config import Config
import asyncio
import logging

logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self):
        self.client = AsyncQdrantClient(
            url=Config.QDRANT_URL,
            api_key=Config.QDRANT_API_KEY
        )
        self.collection_name = Config.QDRANT_COLLECTION_NAME
        self._initialized = False
    
    async def initialize(self):
        """Initialize the vector store"""
        if not self._initialized:
            await self._ensure_collection_exists()
            self._initialized = True
    
    async def _ensure_collection_exists(self):
        """Ensure the collection exists in Qdrant"""
        try:
            await self.client.get_collection(self.collection_name)
        except:
            # Create collection if it doesn't exist
            await self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE),
            )
    
    async def search(self, query_vector: List[float], top_k: int = 5) -> List[Dict]:
        """Search for similar vectors in the collection"""
        if not self._initialized:
            await self.initialize()
            
        results = await self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k,
            with_payload=True
        )
        
        return [
            {
                "text": result.payload["text"],
                "metadata": result.payload["metadata"],
                "score": result.score
            }
            for result in results
        ]
    
    async def add_vectors(self, texts: List[str], metadatas: List[Dict]):
        """Add vectors to the collection"""
        if not self._initialized:
            await self.initialize()
            
        # Generate embeddings for the texts
        from openai import AsyncOpenAI
        from ..config import Config
        
        openai_client = AsyncOpenAI(api_key=Config.OPENAI_API_KEY)
        
        # Batch process embeddings
        embeddings = []
        for text in texts:
            response = await openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )
            embeddings.append(response.data[0].embedding)
        
        # Prepare points for insertion
        points = []
        for i, (text, embedding, metadata) in enumerate(zip(texts, embeddings, metadatas)):
            points.append(models.PointStruct(
                id=i,
                vector=embedding,
                payload={
                    "text": text,
                    "metadata": metadata
                }
            ))
        
        # Upload to Qdrant
        await self.client.upload_points(
            collection_name=self.collection_name,
            points=points
        )