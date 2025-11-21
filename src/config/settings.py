"""
Configuration Settings Module
------------------------------
Centralized configuration for the RAG Chatbot MVP.
Loads environment variables and provides config access.
"""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Settings:
    """
    Centralized settings for the RAG application.
    """
    
    # ==================== PROJECT PATHS ====================
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    CHUNKS_DIR = DATA_DIR / "DataGlossary_Chunks"
    USER_CHUNKS_DIR = DATA_DIR / "User_Upload_Chunks"
    VECTORSTORE_DIR = PROJECT_ROOT / "vectorstore"
    
    # ==================== API KEYS ====================
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    GROQ_API_KEY: Optional[str] = os.getenv("GROQ_API_KEY")
    
    # ==================== EMBEDDING CONFIGURATION ====================
    # Supported: "openai" only (Groq doesn't support embeddings)
    EMBEDDING_PROVIDER: str = os.getenv("EMBEDDING_PROVIDER", "openai")
    
    # OpenAI embedding models
    OPENAI_EMBEDDING_MODEL: str = os.getenv(
        "OPENAI_EMBEDDING_MODEL", 
        "text-embedding-3-small"  # Options: text-embedding-3-small, text-embedding-3-large
    )
    
    # Groq embedding models (if using Groq)
    GROQ_EMBEDDING_MODEL: str = os.getenv(
        "GROQ_EMBEDDING_MODEL",
        "text-embedding-3-small"  # Groq uses OpenAI-compatible format
    )
    
    # Embedding dimensions
    EMBEDDING_DIMENSION: int = int(os.getenv("EMBEDDING_DIMENSION", "1536"))
    # Note: text-embedding-3-small = 1536 dims, text-embedding-3-large = 3072 dims
    
    # ==================== QDRANT CONFIGURATION ====================
    # Qdrant mode: "local" or "cloud"
    QDRANT_MODE: str = os.getenv("QDRANT_MODE", "local")
    
    # Local Qdrant settings
    QDRANT_LOCAL_PATH: Path = VECTORSTORE_DIR / "qdrant_storage"
    
    # Cloud Qdrant settings
    QDRANT_URL: Optional[str] = os.getenv("QDRANT_URL")
    QDRANT_API_KEY: Optional[str] = os.getenv("QDRANT_API_KEY")
    
    # Collection name
    QDRANT_COLLECTION_NAME: str = os.getenv("QDRANT_COLLECTION_NAME", "rag_chatbot_chunks")
    
    # ==================== CHUNKING CONFIGURATION ====================
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "512"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))
    
    # ==================== RETRIEVAL CONFIGURATION ====================
    # Top-k results to retrieve
    RETRIEVAL_TOP_K: int = int(os.getenv("RETRIEVAL_TOP_K", "5"))
    
    # Minimum similarity score threshold (0.0 to 1.0)
    MIN_SIMILARITY_SCORE: float = float(os.getenv("MIN_SIMILARITY_SCORE", "0.5"))
    
    # ==================== LLM CONFIGURATION ====================
    # LLM provider: "openai" or "groq"
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "groq")
    
    # OpenAI LLM models
    OPENAI_LLM_MODEL: str = os.getenv("OPENAI_LLM_MODEL", "gpt-4o-mini")
    
    # Groq LLM models
    GROQ_LLM_MODEL: str = os.getenv(
        "GROQ_LLM_MODEL",
        "llama-3.3-70b-versatile"  # Options: mixtral-8x7b-32768, llama-3.3-70b-versatile
    )
    
    # LLM parameters
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.7"))
    LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "1024"))
    
    # ==================== VALIDATION ====================
    @classmethod
    def validate(cls) -> bool:
        """
        Validate that required settings are configured.
        
        Returns:
            bool: True if valid, False otherwise
        """
        errors = []
        
        # Check API keys based on provider
        if cls.EMBEDDING_PROVIDER == "openai" and not cls.OPENAI_API_KEY:
            errors.append("OPENAI_API_KEY is required for embeddings (Groq doesn't support embeddings)")
        
        if cls.EMBEDDING_PROVIDER == "groq":
            errors.append("EMBEDDING_PROVIDER cannot be 'groq' - Groq doesn't support embeddings. Use 'openai' instead.")
        
        if cls.LLM_PROVIDER == "openai" and not cls.OPENAI_API_KEY:
            errors.append("OPENAI_API_KEY is required when LLM_PROVIDER=openai")
        
        if cls.LLM_PROVIDER == "groq" and not cls.GROQ_API_KEY:
            errors.append("GROQ_API_KEY is required when LLM_PROVIDER=groq")
        
        # Check Qdrant cloud settings
        if cls.QDRANT_MODE == "cloud":
            if not cls.QDRANT_URL:
                errors.append("QDRANT_URL is required when QDRANT_MODE=cloud")
            if not cls.QDRANT_API_KEY:
                errors.append("QDRANT_API_KEY is required when QDRANT_MODE=cloud")
        
        if errors:
            print("❌ Configuration Errors:")
            for error in errors:
                print(f"   - {error}")
            return False
        
        return True
    
    # ==================== DISPLAY CONFIGURATION ====================
    @classmethod
    def display(cls):
        """Display current configuration (hides sensitive data)."""
        print("\n" + "="*70)
        print("⚙️  RAG CHATBOT CONFIGURATION")
        print("="*70)
        
        print(f"\n📁 Paths:")
        print(f"   Project Root: {cls.PROJECT_ROOT}")
        print(f"   Data Dir: {cls.DATA_DIR}")
        print(f"   Vectorstore Dir: {cls.VECTORSTORE_DIR}")
        
        print(f"\n🔑 API Keys:")
        print(f"   OpenAI: {'✅ Set' if cls.OPENAI_API_KEY else '❌ Not Set'}")
        print(f"   Groq: {'✅ Set' if cls.GROQ_API_KEY else '❌ Not Set'}")
        
        print(f"\n🧠 Embedding:")
        print(f"   Provider: {cls.EMBEDDING_PROVIDER}")
        if cls.EMBEDDING_PROVIDER == "openai":
            print(f"   Model: {cls.OPENAI_EMBEDDING_MODEL}")
        else:
            print(f"   Model: {cls.GROQ_EMBEDDING_MODEL}")
        print(f"   Dimension: {cls.EMBEDDING_DIMENSION}")
        
        print(f"\n💾 Qdrant:")
        print(f"   Mode: {cls.QDRANT_MODE}")
        if cls.QDRANT_MODE == "local":
            print(f"   Path: {cls.QDRANT_LOCAL_PATH}")
        else:
            print(f"   URL: {cls.QDRANT_URL}")
            print(f"   API Key: {'✅ Set' if cls.QDRANT_API_KEY else '❌ Not Set'}")
        print(f"   Collection: {cls.QDRANT_COLLECTION_NAME}")
        
        print(f"\n🔍 Retrieval:")
        print(f"   Top-K: {cls.RETRIEVAL_TOP_K}")
        print(f"   Min Score: {cls.MIN_SIMILARITY_SCORE}")
        
        print(f"\n🤖 LLM:")
        print(f"   Provider: {cls.LLM_PROVIDER}")
        if cls.LLM_PROVIDER == "openai":
            print(f"   Model: {cls.OPENAI_LLM_MODEL}")
        else:
            print(f"   Model: {cls.GROQ_LLM_MODEL}")
        print(f"   Temperature: {cls.LLM_TEMPERATURE}")
        print(f"   Max Tokens: {cls.LLM_MAX_TOKENS}")
        
        print("\n" + "="*70 + "\n")


# Create singleton instance
settings = Settings()


# Convenience function
def get_settings() -> Settings:
    """Get settings instance."""
    return settings


if __name__ == "__main__":
    # Display configuration and validate
    settings.display()
    
    if settings.validate():
        print("✅ Configuration is valid!\n")
    else:
        print("❌ Configuration has errors. Please check your .env file.\n")
