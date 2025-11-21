"""
Index Builder Module
--------------------
Generates embeddings and indexes chunks into Qdrant vector store.

Features:
- Load chunks from JSON files
- Generate embeddings using OpenAI or Groq
- Batch processing with progress tracking
- Store vectors + metadata in Qdrant
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import sys
import time

sys.path.append(str(Path(__file__).parent.parent))
from config.settings import settings
from vectorstore.qdrant_store import get_qdrant_client

# Embedding providers
from openai import OpenAI


class EmbeddingGenerator:
    """
    Handles embedding generation using OpenAI or Groq APIs.
    """
    
    def __init__(self, provider: Optional[str] = None, model: Optional[str] = None):
        """
        Initialize embedding generator.
        
        Args:
            provider: 'openai', 'groq', or 'huggingface' (defaults to settings)
            model: Model name (defaults to settings)
        """
        self.provider = provider or settings.EMBEDDING_PROVIDER
        self.model = model or self._get_default_model()
        self.client = None
        self.hf_embedder = None
        
        # Initialize client based on provider
        if self.provider == "openai":
            if not settings.OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY not set in .env file")
            self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
            
        elif self.provider == "groq":
            if not settings.GROQ_API_KEY:
                raise ValueError("GROQ_API_KEY not set in .env file")
            # Groq uses OpenAI-compatible format
            self.client = OpenAI(
                api_key=settings.GROQ_API_KEY,
                base_url="https://api.groq.com/openai/v1"
            )
            
        elif self.provider == "huggingface":
            # Import HuggingFace embedder
            from vectorstore.embedding_huggingface import HuggingFaceEmbedding
            self.hf_embedder = HuggingFaceEmbedding(model_name=self.model)
            
        else:
            raise ValueError(f"Unsupported provider: {self.provider}. Use 'openai', 'groq', or 'huggingface'")
        
        print(f"✅ Embedding generator initialized: {self.provider} - {self.model}")
    
    def _get_default_model(self) -> str:
        """Get default model based on provider."""
        if self.provider == "openai":
            return settings.OPENAI_EMBEDDING_MODEL
        elif self.provider == "huggingface":
            return getattr(settings, 'HUGGINGFACE_MODEL', 'all-mpnet-base-v2')
        return "text-embedding-3-small"
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Input text
            
        Returns:
            List[float]: Embedding vector
        """
        try:
            if self.provider == "huggingface":
                return self.hf_embedder.generate_embedding(text)
            elif self.provider in ["openai", "groq"]:
                response = self.client.embeddings.create(
                    input=text,
                    model=self.model
                )
                return response.data[0].embedding
        except Exception as e:
            print(f"❌ Error generating embedding: {e}")
            return []
    
    def generate_embeddings_batch(
        self,
        texts: List[str],
        batch_size: int = 100,
        show_progress: bool = True
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in batches.
        
        Args:
            texts: List of texts
            batch_size: Number of texts per batch
            show_progress: Show progress bar
            
        Returns:
            List[List[float]]: List of embedding vectors
        """
        if self.provider == "huggingface":
            # HuggingFace can handle all at once efficiently
            return self.hf_embedder.generate_embeddings_batch(
                texts=texts,
                batch_size=batch_size,
                show_progress=show_progress
            )
        
        # OpenAI batch processing
        all_embeddings = []
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_num = i // batch_size + 1
            
            if show_progress:
                print(f"   Processing batch {batch_num}/{total_batches} ({len(batch)} texts)...")
            
            try:
                response = self.client.embeddings.create(
                    input=batch,
                    model=self.model
                )
                
                # Extract embeddings in order
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
                
                # Rate limiting (if needed)
                time.sleep(0.1)
                
            except Exception as e:
                print(f"❌ Error in batch {batch_num}: {e}")
                # Add empty embeddings for failed batch
                all_embeddings.extend([[] for _ in batch])
        
        return all_embeddings


class IndexBuilder:
    """
    Builds vector index from chunked documents.
    """
    
    def __init__(
        self,
        embedding_provider: Optional[str] = None,
        embedding_model: Optional[str] = None
    ):
        """
        Initialize index builder.
        
        Args:
            embedding_provider: 'openai' or 'groq'
            embedding_model: Model name
        """
        self.embedding_generator = EmbeddingGenerator(
            provider=embedding_provider,
            model=embedding_model
        )
        self.qdrant_client = None
    
    def load_chunks_from_json(self, json_path: str) -> List[Dict[str, Any]]:
        """
        Load chunks from JSON file.
        
        Args:
            json_path: Path to chunks JSON file
            
        Returns:
            List[Dict]: List of chunk dictionaries
        """
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            chunks = data.get('chunks', [])
            print(f"✅ Loaded {len(chunks)} chunks from {Path(json_path).name}")
            return chunks
            
        except Exception as e:
            print(f"❌ Error loading chunks: {e}")
            return []
    
    def build_index(
        self,
        chunks: List[Dict[str, Any]],
        batch_size: int = 100,
        create_new_collection: bool = False
    ) -> bool:
        """
        Build vector index from chunks.
        
        Args:
            chunks: List of chunk dictionaries (must have 'text' field)
            batch_size: Batch size for embedding generation
            create_new_collection: If True, recreate collection
            
        Returns:
            bool: True if successful
        """
        try:
            if not chunks:
                print("⚠️  No chunks to index")
                return False
            
            print("\n" + "="*70)
            print("🔨 BUILDING VECTOR INDEX")
            print("="*70)
            print(f"📊 Total chunks: {len(chunks)}")
            print(f"⚙️  Batch size: {batch_size}")
            print(f"🧠 Embedding model: {self.embedding_generator.model}")
            print("")
            
            # Initialize Qdrant client
            print("📦 Initializing Qdrant...")
            self.qdrant_client = get_qdrant_client(create_collection=False)
            
            # Create or recreate collection
            if create_new_collection:
                print("🔨 Recreating collection...")
                self.qdrant_client.create_collection(recreate=True)
            else:
                self.qdrant_client.create_collection(recreate=False)
            
            print("")
            
            # Extract texts
            print("📝 Extracting texts from chunks...")
            texts = [chunk.get('text', '') for chunk in chunks]
            
            # Filter empty texts
            valid_indices = [i for i, text in enumerate(texts) if text.strip()]
            valid_texts = [texts[i] for i in valid_indices]
            valid_chunks = [chunks[i] for i in valid_indices]
            
            if len(valid_texts) < len(texts):
                print(f"⚠️  Filtered out {len(texts) - len(valid_texts)} empty chunks")
            
            print(f"✅ Processing {len(valid_texts)} valid chunks\n")
            
            # Generate embeddings
            print("🧠 Generating embeddings...")
            embeddings = self.embedding_generator.generate_embeddings_batch(
                texts=valid_texts,
                batch_size=batch_size,
                show_progress=True
            )
            
            # Filter out failed embeddings
            valid_embeddings = []
            valid_metadatas = []
            valid_ids = []
            
            for i, embedding in enumerate(embeddings):
                if embedding:  # Check if embedding is not empty
                    valid_embeddings.append(embedding)
                    
                    # Prepare metadata (include all chunk fields)
                    metadata = valid_chunks[i].copy()
                    
                    # Ensure text is in metadata
                    if 'text' not in metadata:
                        metadata['text'] = valid_texts[i]
                    
                    valid_metadatas.append(metadata)
                    
                    # Use chunk_id as ID if available, otherwise generate
                    chunk_id = valid_chunks[i].get('chunk_id', f"chunk_{i}")
                    valid_ids.append(chunk_id)
            
            print(f"✅ Generated {len(valid_embeddings)} embeddings\n")
            
            # Insert into Qdrant
            print("💾 Inserting vectors into Qdrant...")
            success = self.qdrant_client.insert_vectors(
                vectors=valid_embeddings,
                metadatas=valid_metadatas,
                ids=valid_ids
            )
            
            if success:
                print("")
                print("="*70)
                print("✅ INDEX BUILT SUCCESSFULLY!")
                print("="*70)
                
                # Show collection info
                info = self.qdrant_client.get_collection_info()
                print(f"\n📊 Collection: {info.get('name')}")
                print(f"   Points: {info.get('points_count')}")
                print(f"   Dimension: {info.get('vector_size')}")
                print(f"   Distance: {info.get('distance')}")
                print("")
                
                return True
            else:
                print("❌ Failed to insert vectors")
                return False
            
        except Exception as e:
            print(f"❌ Error building index: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def build_index_from_file(
        self,
        json_path: str,
        batch_size: int = 100,
        create_new_collection: bool = False
    ) -> bool:
        """
        Build index directly from a JSON file.
        
        Args:
            json_path: Path to chunks JSON file
            batch_size: Batch size for embedding generation
            create_new_collection: If True, recreate collection
            
        Returns:
            bool: True if successful
        """
        chunks = self.load_chunks_from_json(json_path)
        if not chunks:
            return False
        
        return self.build_index(
            chunks=chunks,
            batch_size=batch_size,
            create_new_collection=create_new_collection
        )


# Convenience functions
def build_index_from_chunks(
    chunks: List[Dict[str, Any]],
    batch_size: int = 100,
    create_new_collection: bool = False
) -> bool:
    """
    Build index from chunk list.
    
    Args:
        chunks: List of chunk dictionaries
        batch_size: Batch size
        create_new_collection: Recreate collection
        
    Returns:
        bool: Success status
    """
    builder = IndexBuilder()
    return builder.build_index(chunks, batch_size, create_new_collection)


def build_index_from_file(
    json_path: str,
    batch_size: int = 100,
    create_new_collection: bool = False
) -> bool:
    """
    Build index from JSON file.
    
    Args:
        json_path: Path to chunks JSON
        batch_size: Batch size
        create_new_collection: Recreate collection
        
    Returns:
        bool: Success status
    """
    builder = IndexBuilder()
    return builder.build_index_from_file(json_path, batch_size, create_new_collection)


if __name__ == "__main__":
    # Example: Build index from DataGlossary chunks
    print("\n" + "="*70)
    print("🧪 Testing Index Builder")
    print("="*70 + "\n")
    
    # Path to chunks file
    chunks_file = settings.CHUNKS_DIR / "DataGlossary_chunks.json"
    
    if chunks_file.exists():
        print(f"📂 Using chunks file: {chunks_file}\n")
        
        # Build index
        success = build_index_from_file(
            json_path=str(chunks_file),
            batch_size=50,  # Smaller batch for testing
            create_new_collection=True  # Recreate collection
        )
        
        if success:
            print("✅ Index building test completed!\n")
        else:
            print("❌ Index building test failed!\n")
    else:
        print(f"❌ Chunks file not found: {chunks_file}")
        print("   Run chunking scripts first.\n")
