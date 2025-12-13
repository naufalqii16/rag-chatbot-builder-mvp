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
    
    def __init__(self, top_k: Optional[int] = None, collection_name: Optional[str] = None):
        """
        Initialize retriever.
        
        Args:
            top_k: Number of results to return (defaults to RETRIEVAL_TOP_K from settings)
            collection_name: Name of Qdrant collection to use (defaults to QDRANT_COLLECTION_NAME)
        """
        self.top_k = top_k if top_k is not None else settings.RETRIEVAL_TOP_K
        self.collection_name = collection_name if collection_name is not None else settings.QDRANT_COLLECTION_NAME
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
            # Get Qdrant client with specified collection
            self.client = get_qdrant_client(
                collection_name=self.collection_name,
                create_collection=False
            )
            
            # DIAGNOSTIC: Check if collection exists and has data
            try:
                collection_info = self.client.get_collection_info()
                vector_count = collection_info.get('points_count', 0)
                print(f"🔍 DEBUG: Collection '{self.collection_name}' has {vector_count} vectors")
                
                if vector_count == 0:
                    print(f"⚠️  WARNING: Collection '{self.collection_name}' is empty! No documents indexed.")
                    return []
            except Exception as e:
                print(f"❌ ERROR: Cannot access collection '{self.collection_name}': {e}")
                print(f"   Make sure the collection exists and documents have been indexed.")
                return []
            
            # Generate query embedding
            print(f"🔍 DEBUG: Generating embedding for query: '{query[:50]}...'")
            query_embedding = self._get_query_embedding(query)
            
            if not query_embedding or len(query_embedding) == 0:
                print(f"❌ ERROR: Failed to generate query embedding")
                return []
            
            print(f"✅ DEBUG: Generated embedding with dimension {len(query_embedding)}")
            print(f"🔍 DEBUG: Searching with threshold: {self.min_score}, top_k: {k}")
            
            # First, try without threshold to see what scores we get
            search_results_all = self.client.search(
                query_vector=query_embedding,
                top_k=k * 2,  # Get more results to see actual scores
                score_threshold=None  # No threshold for diagnostic
            )
            
            print(f"🔍 DEBUG: Found {len(search_results_all)} results without threshold")
            
            if search_results_all:
                # Show score range
                scores = [r.get('score', 0.0) for r in search_results_all]
                print(f"🔍 DEBUG: Score range: min={min(scores):.4f}, max={max(scores):.4f}, avg={sum(scores)/len(scores):.4f}")
                print(f"🔍 DEBUG: Current threshold: {self.min_score}")
                
                # Filter by threshold
                filtered_results = [r for r in search_results_all if r.get('score', 0.0) >= self.min_score]
                print(f"🔍 DEBUG: After threshold filter: {len(filtered_results)} results")
                
                if len(filtered_results) == 0 and len(search_results_all) > 0:
                    print(f"⚠️  WARNING: All results filtered out by threshold {self.min_score}!")
                    print(f"   Highest score was {max(scores):.4f}. Consider lowering MIN_SIMILARITY_SCORE.")
                    # Return top results even if below threshold for debugging
                    filtered_results = search_results_all[:k]
                    print(f"   Returning top {len(filtered_results)} results anyway for debugging...")
            else:
                filtered_results = []
            
            # Now do the actual search with threshold
            search_results = self.client.search(
                query_vector=query_embedding,
                top_k=k,
                score_threshold=self.min_score
            )
            
            # Use diagnostic results if actual search returned nothing but we have results
            if not search_results and filtered_results:
                search_results = filtered_results[:k]
            
            print(f"✅ DEBUG: Returning {len(search_results)} results")
            
            # Format results - search_results is already formatted by qdrant_store
            # Each result has: id, score, text, metadata
            results = []
            for result in search_results:
                # Extract metadata, removing 'text' field since it's already at top level
                metadata = {k: v for k, v in result.get('metadata', {}).items() if k != 'text'}
                
                results.append({
                    'chunk_id': result.get('id'),
                    'text': result.get('text', ''),
                    'metadata': metadata,
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