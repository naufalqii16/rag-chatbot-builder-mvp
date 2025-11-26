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
    # Supported: "openai" or "huggingface" (FREE!)
    EMBEDDING_PROVIDER: str = os.getenv("EMBEDDING_PROVIDER", "huggingface")
    
    # OpenAI embedding models
    OPENAI_EMBEDDING_MODEL: str = os.getenv(
        "OPENAI_EMBEDDING_MODEL", 
        "text-embedding-3-small"  # Options: text-embedding-3-small, text-embedding-3-large
    )
    
    # HuggingFace embedding models (FREE, local)
    HUGGINGFACE_MODEL: str = os.getenv(
        "HUGGINGFACE_MODEL",
        "sentence-transformers/all-mpnet-base-v2"  # 768 dims, excellent quality
    )
    # Popular alternatives:
    # - all-MiniLM-L6-v2: 384 dims, very fast
    # - multi-qa-mpnet-base-dot-v1: 768 dims, optimized for Q&A
    # - paraphrase-multilingual-mpnet-base-v2: 768 dims, multilingual
    
    # Embedding dimensions
    EMBEDDING_DIMENSION: int = int(os.getenv("EMBEDDING_DIMENSION", "768"))
    # Note: 
    # - text-embedding-3-small (OpenAI) = 1536 dims
    # - text-embedding-3-large (OpenAI) = 3072 dims
    # - all-mpnet-base-v2 (HuggingFace) = 768 dims
    # - all-MiniLM-L6-v2 (HuggingFace) = 384 dims
    
    # ==================== QDRANT CONFIGURATION ====================
    # Qdrant mode: "local" or "server" or "cloud"
    QDRANT_MODE: str = os.getenv("QDRANT_MODE", "local")
    
    # Local Qdrant settings
    QDRANT_LOCAL_PATH: Path = VECTORSTORE_DIR / "qdrant_storage"
    
    # Server/Cloud Qdrant settings
    QDRANT_URL: Optional[str] = os.getenv("QDRANT_URL")
    QDRANT_API_KEY: Optional[str] = os.getenv("QDRANT_API_KEY")
    
    # Collection names - separate collections for glossary and user uploads
    QDRANT_GLOSSARY_COLLECTION: str = os.getenv("QDRANT_GLOSSARY_COLLECTION", "glossary_collection")
    QDRANT_USER_UPLOAD_COLLECTION: str = os.getenv("QDRANT_USER_UPLOAD_COLLECTION", "user_upload_collection")
    
    # Backward compatibility
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
    
    # Groq LLM models (FREE!)
    GROQ_LLM_MODEL: str = os.getenv(
        "GROQ_LLM_MODEL",
        "llama-3.3-70b-versatile"  # Options: mixtral-8x7b-32768, llama-3.3-70b-versatile, gemma2-9b-it
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
            errors.append("OPENAI_API_KEY is required when EMBEDDING_PROVIDER=openai")
        
        if cls.EMBEDDING_PROVIDER == "huggingface":
            # HuggingFace is FREE and doesn't need API key
            pass
        
        if cls.EMBEDDING_PROVIDER not in ["openai", "huggingface"]:
            errors.append(f"EMBEDDING_PROVIDER must be 'openai' or 'huggingface', got '{cls.EMBEDDING_PROVIDER}'")
        
        if cls.LLM_PROVIDER == "openai" and not cls.OPENAI_API_KEY:
            errors.append("OPENAI_API_KEY is required when LLM_PROVIDER=openai")
        
        if cls.LLM_PROVIDER == "groq" and not cls.GROQ_API_KEY:
            errors.append("GROQ_API_KEY is required when LLM_PROVIDER=groq")
        
        if cls.LLM_PROVIDER not in ["openai", "groq"]:
            errors.append(f"LLM_PROVIDER must be 'openai' or 'groq', got '{cls.LLM_PROVIDER}'")
        
        # Check Qdrant cloud/server settings
        if cls.QDRANT_MODE in ["cloud", "server"]:
            if not cls.QDRANT_URL:
                errors.append(f"QDRANT_URL is required when QDRANT_MODE={cls.QDRANT_MODE}")
        
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
        elif cls.EMBEDDING_PROVIDER == "huggingface":
            print(f"   Model: {cls.HUGGINGFACE_MODEL}")
            print(f"   💰 FREE - No API costs!")
        print(f"   Dimension: {cls.EMBEDDING_DIMENSION}")
        
        print(f"\n💾 Qdrant:")
        print(f"   Mode: {cls.QDRANT_MODE}")
        if cls.QDRANT_MODE == "local":
            print(f"   Path: {cls.QDRANT_LOCAL_PATH}")
        else:
            print(f"   URL: {cls.QDRANT_URL}")
            print(f"   API Key: {'✅ Set' if cls.QDRANT_API_KEY else '❌ Not Set'}")
        print(f"   Glossary Collection: {cls.QDRANT_GLOSSARY_COLLECTION}")
        print(f"   User Upload Collection: {cls.QDRANT_USER_UPLOAD_COLLECTION}")
        
        print(f"\n🔍 Retrieval:")
        print(f"   Top-K: {cls.RETRIEVAL_TOP_K}")
        print(f"   Min Score: {cls.MIN_SIMILARITY_SCORE}")
        
        print(f"\n🤖 LLM:")
        print(f"   Provider: {cls.LLM_PROVIDER}")
        if cls.LLM_PROVIDER == "openai":
            print(f"   Model: {cls.OPENAI_LLM_MODEL}")
        else:
            print(f"   Model: {cls.GROQ_LLM_MODEL}")
            if cls.LLM_PROVIDER == "groq":
                print(f"   💰 FREE - No API costs!")
        print(f"   Temperature: {cls.LLM_TEMPERATURE}")
        print(f"   Max Tokens: {cls.LLM_MAX_TOKENS}")
        
        # Show if using 100% FREE setup
        if cls.EMBEDDING_PROVIDER == "huggingface" and cls.LLM_PROVIDER == "groq":
            print(f"\n🎉 100% FREE SETUP!")
            print(f"   Embeddings: HuggingFace (local)")
            print(f"   LLM: Groq")
        
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