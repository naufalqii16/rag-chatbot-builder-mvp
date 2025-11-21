"""
HuggingFace Embedding Generator
--------------------------------
local embedding generation using Sentence Transformers.

Features:
- 100% FREE - no API keys needed
- Runs locally - no internet required after download
- High quality - comparable to OpenAI
- Fast - GPU support available
- Privacy - data never leaves your machine

Popular Models:
- all-MiniLM-L6-v2: 384 dims, very fast, good quality
- all-mpnet-base-v2: 768 dims, excellent quality (RECOMMENDED)
- multi-qa-mpnet-base-dot-v1: 768 dims, optimized for Q&A
"""

from typing import List, Optional
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("⚠️  sentence-transformers not installed. Run: pip install sentence-transformers")


class HuggingFaceEmbedding:
    """
    Free local embedding generation using HuggingFace Sentence Transformers.
    """
    
    # Popular model configurations
    MODELS = {
        'all-MiniLM-L6-v2': {
            'dimension': 384,
            'size': '80MB',
            'speed': 'Very Fast',
            'quality': 'Good',
            'description': 'Fast and lightweight, good for most use cases'
        },
        'all-mpnet-base-v2': {
            'dimension': 768,
            'size': '420MB',
            'speed': 'Fast',
            'quality': 'Excellent',
            'description': 'Best overall quality, RECOMMENDED for production'
        },
        'multi-qa-mpnet-base-dot-v1': {
            'dimension': 768,
            'size': '420MB',
            'speed': 'Fast',
            'quality': 'Excellent',
            'description': 'Optimized for question-answering tasks'
        },
        'paraphrase-multilingual-MiniLM-L12-v2': {
            'dimension': 384,
            'size': '420MB',
            'speed': 'Medium',
            'quality': 'Good',
            'description': 'Supports 50+ languages'
        }
    }
    
    def __init__(self, model_name: str = 'all-mpnet-base-v2', device: Optional[str] = None):
        """
        Initialize HuggingFace embedding generator.
        
        Args:
            model_name: Name of the sentence-transformers model
            device: 'cuda', 'cpu', or None (auto-detect)
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "sentence-transformers is required for HuggingFace embeddings. "
                "Install with: pip install sentence-transformers"
            )
        
        self.model_name = model_name
        
        # Load model (downloads on first use, then cached)
        print(f"📥 Loading HuggingFace model: {model_name}")
        if model_name in self.MODELS:
            print(f"   Size: {self.MODELS[model_name]['size']}")
            print(f"   Quality: {self.MODELS[model_name]['quality']}")
            print(f"   Speed: {self.MODELS[model_name]['speed']}")
        
        self.model = SentenceTransformer(model_name, device=device)
        
        # Get embedding dimension
        self.dimension = self.model.get_sentence_embedding_dimension()
        
        print(f"✅ Model loaded successfully!")
        print(f"   Dimension: {self.dimension}")
        print(f"   Device: {self.model.device}")
        print(f"   100% FREE - No API costs! 🎉\n")
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Input text
            
        Returns:
            List[float]: Embedding vector
        """
        try:
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding.tolist()
        except Exception as e:
            print(f"❌ Error generating embedding: {e}")
            return []
    
    def generate_embeddings_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = True
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in batches.
        
        Args:
            texts: List of input texts
            batch_size: Batch size for processing
            show_progress: Show progress bar
            
        Returns:
            List[List[float]]: List of embedding vectors
        """
        try:
            print(f"🧠 Generating {len(texts)} embeddings...")
            print(f"   Batch size: {batch_size}")
            print(f"   Model: {self.model_name}")
            
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True
            )
            
            print(f"✅ Generated {len(embeddings)} embeddings")
            
            return embeddings.tolist()
            
        except Exception as e:
            print(f"❌ Error in batch embedding: {e}")
            return []
    
    @classmethod
    def list_recommended_models(cls):
        """Print recommended models."""
        print("\n" + "="*70)
        print("🤗 RECOMMENDED HUGGINGFACE MODELS")
        print("="*70 + "\n")
        
        for model_name, info in cls.MODELS.items():
            print(f"📦 {model_name}")
            print(f"   Dimension: {info['dimension']}")
            print(f"   Size: {info['size']}")
            print(f"   Speed: {info['speed']}")
            print(f"   Quality: {info['quality']}")
            print(f"   Use case: {info['description']}")
            print("")
        
        print("💡 RECOMMENDATION: Use 'all-mpnet-base-v2' for best quality")
        print("="*70 + "\n")
    
    def get_model_info(self) -> dict:
        """Get information about the current model."""
        return {
            'model_name': self.model_name,
            'dimension': self.dimension,
            'device': str(self.model.device),
            'max_seq_length': self.model.max_seq_length,
            'free': True,
            'local': True
        }


def test_huggingface_embeddings():
    """Test HuggingFace embeddings."""
    print("\n" + "="*70)
    print("🧪 Testing HuggingFace Embeddings")
    print("="*70 + "\n")
    
    # Show recommended models
    HuggingFaceEmbedding.list_recommended_models()
    
    # Test with recommended model
    print("Testing with recommended model (all-mpnet-base-v2)...\n")
    
    embedder = HuggingFaceEmbedding('all-mpnet-base-v2')
    
    # Test single embedding
    test_text = "This is a test sentence for embedding generation."
    print(f"📝 Test text: {test_text}\n")
    
    embedding = embedder.generate_embedding(test_text)
    
    if embedding:
        print(f"✅ Embedding generated!")
        print(f"   Dimension: {len(embedding)}")
        print(f"   First 5 values: {embedding[:5]}")
        print("")
    
    # Test batch
    test_texts = [
        "What is machine learning?",
        "How does neural network work?",
        "Explain artificial intelligence"
    ]
    
    print(f"Testing batch embedding ({len(test_texts)} texts)...\n")
    embeddings = embedder.generate_embeddings_batch(test_texts, show_progress=True)
    
    if embeddings:
        print(f"\n✅ Batch embeddings generated!")
        print(f"   Total: {len(embeddings)} embeddings")
        print(f"   Dimension: {len(embeddings[0])}")
        print("")
    
    # Show model info
    info = embedder.get_model_info()
    print("="*70)
    print("📊 MODEL INFO")
    print("="*70)
    for key, value in info.items():
        print(f"   {key}: {value}")
    print("="*70 + "\n")
    
    print("✅ HuggingFace embeddings test completed!\n")


if __name__ == "__main__":
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        print("\n❌ sentence-transformers not installed!")
        print("\nInstall with:")
        print("   pip install sentence-transformers")
        print("\nThen run this script again.\n")
    else:
        test_huggingface_embeddings()
