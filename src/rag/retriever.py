"""
Retriever Module
----------------
Handle semantic search and retrieval from Qdrant vector store.

Features:
- Semantic search using embeddings
- Configurable top_k results
- Support for multiple embedding providers (OpenAI, HuggingFace)
- Automatic re-ranking (optional)
"""

from typing import List, Dict, Any, Optional
from config.settings import settings
from vectorstore.qdrant_store import get_qdrant_client
from vectorstore.embedding_huggingface import HuggingFaceEmbedding
class Retriever:
    """Retrieve relevant chunks from vector store."""
    
    def __init__(self, top_k: Optional[int] = None):
        """
        Initialize retriever.
        
        Args:
            top_k: Number of results to return (defaults to RETRIEVAL_TOP_K from settings)
        """
        self.top_k = top_k if top_k is not None else settings.RETRIEVAL_TOP_K
        self.collection_name = settings.QDRANT_COLLECTION_NAME
        self.min_score = settings.MIN_SIMILARITY_SCORE
        
        # Initialize embedding generator based on provider
        if settings.EMBEDDING_PROVIDER == "huggingface":
            # Try to get HUGGINGFACE_MODEL from settings, fallback to default
            model_name = getattr(settings, 'HUGGINGFACE_MODEL', 'sentence-transformers/all-mpnet-base-v2')
            self.embedder = HuggingFaceEmbedding(model_name=model_name)
            print(f"✅ Using HuggingFace embeddings: {model_name}")
        else:
            raise ValueError(f"Unsupported embedding provider: {settings.EMBEDDING_PROVIDER}")
        
        # Qdrant client will be created per request to avoid lock issues
        self.client = None
    
    def _get_query_embedding(self, query: str) -> List[float]:
        """
        Generate embedding for query using HuggingFace.
        
        Args:
            query: Query text
            
        Returns:
            Embedding vector
        """
        return self.embedder.generate_embedding(query)
    
    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Retrieve relevant chunks for query.
        
        Args:
            query: User query
            top_k: Number of results (overrides default)
            
        Returns:
            List of retrieved chunks with metadata and scores
        """
        k = top_k if top_k is not None else self.top_k
        
        try:
            # Get Qdrant client
            self.client = get_qdrant_client(create_collection=False)
            
            # Generate query embedding
            query_embedding = self._get_query_embedding(query)
            
            # Search in Qdrant using the correct method signature
            search_results = self.client.search(
                query_vector=query_embedding,
                top_k=k,
                score_threshold=self.min_score
            )
            
            # Format results - results already filtered by score_threshold
            results = []
            for result in search_results:
                results.append({
                    'chunk_id': result.get('id'),
                    'text': result.get('text', ''),
                    'metadata': result.get('metadata', {}),
                    'score': result.get('score', 0.0)
                })
            
            return results
            
        except Exception as e:
            print(f"❌ Retrieval error: {e}")
            import traceback
            traceback.print_exc()
            return []
        
        finally:
            # Always close client to avoid lock
            if self.client:
                self.client.close()
                self.client = None
    
    def retrieve_with_context(
        self, 
        query: str, 
        top_k: Optional[int] = None,
        include_neighbors: bool = False
    ) -> Dict[str, Any]:
        """
        Retrieve with additional context.
        
        Args:
            query: User query
            top_k: Number of results
            include_neighbors: Include neighboring chunks for context
            
        Returns:
            Dictionary with results and metadata
        """
        results = self.retrieve(query, top_k)
        
        return {
            'query': query,
            'num_results': len(results),
            'results': results,
            'avg_score': sum(r['score'] for r in results) / len(results) if results else 0.0,
            'top_score': results[0]['score'] if results else 0.0
        }
    
    def format_context(self, results: List[Dict[str, Any]]) -> str:
        """
        Format retrieved results into context string for LLM.
        
        Args:
            results: Retrieved chunks
            
        Returns:
            Formatted context string
        """
        if not results:
            return "No relevant information found."
        
        context_parts = []
        for i, result in enumerate(results, 1):
            text = result['text']
            score = result['score']
            metadata = result.get('metadata', {})
            
            # Build context entry - cleaner format without chunk labels
            entry = f"{text}"
            
            # Add metadata if available
            if metadata:
                source = metadata.get('source', '')
                if source:
                    entry += f"\n[Source: {source}]"
            
            context_parts.append(entry)
        
        return "\n\n---\n\n".join(context_parts)