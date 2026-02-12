"""
PDF processing service for extracting and chunking content
"""
import pdfplumber
from typing import List, Dict
import asyncio
from ..config import Config
from .vectorstore import VectorStore
import logging

logger = logging.getLogger(__name__)

class PDFProcessor:
    def __init__(self):
        self.chunk_size = Config.CHUNK_SIZE
        self.chunk_overlap = Config.CHUNK_OVERLAP
        self.vectorstore = VectorStore()
        self.processing_status = {
            "status": "idle",
            "progress": 0,
            "total_chunks": 0,
            "processed_chunks": 0
        }
    
    async def process_pdf(self, pdf_path: str) -> Dict:
        """Process PDF and generate embeddings"""
        await self.vectorstore.initialize()
        
        self.processing_status["status"] = "processing"
        self.processing_status["progress"] = 0
        
        # Extract text from PDF
        text_chunks = self._extract_and_chunk_pdf(pdf_path)
        self.processing_status["total_chunks"] = len(text_chunks)
        
        # Prepare metadata for each chunk
        metadatas = []
        for i, chunk in enumerate(text_chunks):
            metadatas.append({
                "source": pdf_path,
                "chunk_index": i,
                "page_range": chunk.get("page_range", ""),
                "chapter": chunk.get("chapter", "Unknown"),
                "section": chunk.get("section", "Unknown"),
                "module": chunk.get("module", "Unknown")
            })
        
        # Extract just the text content for vector storage
        texts = [chunk["text"] for chunk in text_chunks]
        
        # Add to vector store
        await self.vectorstore.add_vectors(texts, metadatas)
        
        self.processing_status["status"] = "completed"
        self.processing_status["progress"] = 100
        self.processing_status["processed_chunks"] = len(text_chunks)
        
        return {
            "chunks_processed": len(text_chunks),
            "status": "completed"
        }
    
    def _extract_and_chunk_pdf(self, pdf_path: str) -> List[Dict]:
        """Extract text from PDF and create semantic chunks"""
        chunks = []
        
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                text = page.extract_text()
                
                # Simple chunking by length (could be improved with semantic chunking)
                for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
                    chunk_text = text[i:i + self.chunk_size]
                    
                    # Try to identify chapter/section from the text
                    chapter_info = self._identify_chapter_section(chunk_text)
                    
                    chunk_data = {
                        "text": chunk_text,
                        "page_range": f"Page {page_num + 1}",
                        "chapter": chapter_info.get("chapter", "Unknown"),
                        "section": chapter_info.get("section", "Unknown"),
                        "module": chapter_info.get("module", "Unknown")
                    }
                    
                    chunks.append(chunk_data)
        
        return chunks
    
    def _identify_chapter_section(self, text: str) -> Dict:
        """Identify chapter and section from text content"""
        # This is a simplified implementation
        # In a real implementation, you'd use more sophisticated NLP techniques
        text_lower = text.lower()
        
        # Identify modules based on keywords
        module = "Unknown"
        if "module 1" in text_lower or "ros" in text_lower:
            module = "Module 1: The Robotic Nervous System (ROS 2)"
        elif "module 2" in text_lower or "gazebo" in text_lower or "unity" in text_lower:
            module = "Module 2: The Digital Twin (Gazebo & Unity)"
        elif "module 3" in text_lower or "nvidia isaac" in text_lower or "isaac" in text_lower:
            module = "Module 3: The AI-Robot Brain (NVIDIA Isaac)"
        elif "module 4" in text_lower or "vla" in text_lower or "humanoid" in text_lower:
            module = "Module 4: Vision-Language-Action (VLA)"
        
        # Identify chapters based on keywords
        chapter = "Unknown"
        if "week 1" in text_lower or "introduction" in text_lower:
            chapter = "Week 1: Introduction to Physical AI"
        elif "week 2" in text_lower or "embodied intelligence" in text_lower:
            chapter = "Week 2: Embodied Intelligence"
        elif "week 3" in text_lower or "ros 2" in text_lower:
            chapter = "Week 3: ROS 2 Fundamentals"
        elif "week 4" in text_lower:
            chapter = "Week 4: ROS 2 Nodes, Topics, Services"
        elif "week 5" in text_lower:
            chapter = "Week 5: ROS 2 Python Integration"
        elif "week 6" in text_lower:
            chapter = "Week 6: Robot Simulation with Gazebo"
        elif "week 7" in text_lower:
            chapter = "Week 7: Unity for High-Fidelity Rendering"
        elif "week 8" in text_lower:
            chapter = "Week 8: NVIDIA Isaac SDK Introduction"
        elif "week 9" in text_lower:
            chapter = "Week 9: AI-Powered Perception and Manipulation"
        elif "week 10" in text_lower:
            chapter = "Week 10: Reinforcement Learning and Sim-to-Real Transfer"
        elif "week 11" in text_lower:
            chapter = "Week 11: Humanoid Kinematics and Dynamics"
        elif "week 12" in text_lower:
            chapter = "Week 12: Bipedal Locomotion and Balance Control"
        elif "week 13" in text_lower:
            chapter = "Week 13: Conversational Robotics and Capstone Project"
        
        # Identify sections based on common headings
        section = "General Content"
        if "abstract" in text_lower[:200]:  # Check beginning of text
            section = "Abstract"
        elif "introduction" in text_lower[:200]:
            section = "Introduction"
        elif "background" in text_lower[:200]:
            section = "Background"
        elif "method" in text_lower[:200]:  # Covers methodology, methods
            section = "Methodology"
        elif "results" in text_lower[:200]:
            section = "Results"
        elif "conclusion" in text_lower[:200]:
            section = "Conclusion"
        elif "references" in text_lower[:200]:
            section = "References"
        
        return {
            "module": module,
            "chapter": chapter,
            "section": section
        }
    
    async def get_processing_status(self) -> Dict:
        """Get the current status of PDF processing"""
        return self.processing_status