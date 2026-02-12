import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Application configuration
class Config:
    # Qwen API Configuration
    QWEN_API_KEY = os.getenv("QWEN_API_KEY")
    QWEN_MODEL = os.getenv("QWEN_MODEL", "qwen-turbo")
    QWEN_API_ENDPOINT = os.getenv("QWEN_API_ENDPOINT", "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation")
    
    # OpenAI Configuration (for embeddings)
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    # Database Configuration
    DATABASE_URL = os.getenv("DATABASE_URL")
    
    # Qdrant Configuration
    QDRANT_URL = os.getenv("QDRANT_URL")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "physical-ai-textbook")
    
    # Application Configuration
    BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
    FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")
    ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
    
    # Ollama Configuration (optional)
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
    
    # RAG Configuration
    CHUNK_SIZE = 512
    CHUNK_OVERLAP = 50
    TOP_K_RESULTS = 5
    MAX_TOKENS = 2048
    TEMPERATURE = 0.7