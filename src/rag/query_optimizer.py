"""
Query Optimizer Module
----------------------
Enhance user queries for better retrieval performance.

Features:
- Query expansion with synonyms
- Multi-query generation
- Keyword extraction
- Fallback strategies
"""

from typing import List, Dict, Any
from groq import Groq
from config.settings import settings


class QueryOptimizer:
    """Optimize user queries for better retrieval."""
    
    def __init__(self):
        """Initialize query optimizer."""
        self.groq_client = Groq(api_key=settings.GROQ_API_KEY)
        self.model = settings.GROQ_LLM_MODEL
    
    def expand_query(self, original_query: str) -> List[str]:
        """
        Generate multiple query variations for better retrieval.
        
        Args:
            original_query: Original user query
            
        Returns:
            List of query variations
        """
        try:
            # Use LLM to generate variations
            prompt = f"""Given this database-related query, generate 3 alternative phrasings that have the same meaning but use different words. Focus on technical database terminology.

Original query: "{original_query}"

Generate 3 variations (one per line, no numbering):"""
            
            response = self.groq_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a database expert that helps rephrase queries."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=200
            )
            
            variations_text = response.choices[0].message.content.strip()
            variations = [v.strip() for v in variations_text.split('\n') if v.strip()]
            
            # Always include original
            all_queries = [original_query] + variations[:3]
            
            return all_queries
            
        except Exception as e:
            print(f"⚠️  Query expansion failed: {e}")
            # Fallback: simple keyword extraction
            return [original_query, self._extract_keywords(original_query)]
    
    def _extract_keywords(self, query: str) -> str:
        """
        Extract key terms from query.
        
        Args:
            query: User query
            
        Returns:
            Keywords string
        """
        # Simple keyword extraction (can be improved with NLP)
        keywords = []
        
        # Database-specific terms
        db_terms = [
            'schema', 'table', 'column', 'database', 'primary key', 'watermark',
            'extraction', 'incremental', 'full load', 'server', 'source'
        ]
        
        query_lower = query.lower()
        for term in db_terms:
            if term in query_lower:
                keywords.append(term)
        
        # Remove common words
        stop_words = ['what', 'which', 'show', 'me', 'all', 'the', 'are', 'is', 'in', 'for']
        words = query.lower().split()
        for word in words:
            if word not in stop_words and len(word) > 2:
                keywords.append(word)
        
        return ' '.join(set(keywords))
    
    def analyze_query_intent(self, query: str) -> Dict[str, Any]:
        """
        Analyze what the user is asking for.
        
        Args:
            query: User query
            
        Returns:
            Dictionary with intent analysis
        """
        query_lower = query.lower()
        
        # Detect query type
        intent = {
            'is_listing': any(word in query_lower for word in ['list', 'show', 'all', 'get']),
            'is_filtering': any(word in query_lower for word in ['exclude', 'except', 'not', 'without']),
            'is_counting': any(word in query_lower for word in ['how many', 'count', 'number of']),
            'is_definition': any(word in query_lower for word in ['what is', 'define', 'explain']),
            'is_comparison': any(word in query_lower for word in ['difference', 'compare', 'vs', 'versus']),
        }
        
        # Detect entities
        entities = {
            'schema': 'schema' in query_lower,
            'table': 'table' in query_lower,
            'column': 'column' in query_lower,
            'database': 'database' in query_lower,
            'extraction': 'extraction' in query_lower or 'load' in query_lower,
        }
        
        return {
            'intent': intent,
            'entities': entities,
            'original_query': query
        }


class HybridRetriever:
    """
    Hybrid retrieval combining:
    1. Semantic search (embeddings)
    2. Keyword search (BM25-like)
    3. Query expansion
    """
    
    def __init__(self, base_retriever, query_optimizer: QueryOptimizer = None):
        """
        Initialize hybrid retriever.
        
        Args:
            base_retriever: Base Retriever instance
            query_optimizer: QueryOptimizer instance
        """
        self.base_retriever = base_retriever
        self.optimizer = query_optimizer or QueryOptimizer()
    
    def retrieve_with_expansion(
        self, 
        query: str, 
        top_k: int = 5,
        use_expansion: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Retrieve using query expansion.
        
        Args:
            query: User query
            top_k: Number of results
            use_expansion: Whether to use query expansion
            
        Returns:
            List of retrieved results
        """
        all_results = []
        seen_ids = set()
        
        if use_expansion:
            # Generate query variations
            print(f"🔄 Expanding query...")
            queries = self.optimizer.expand_query(query)
            print(f"   Generated {len(queries)} query variations")
        else:
            queries = [query]
        
        # Retrieve for each query variation
        for i, q in enumerate(queries, 1):
            print(f"   {i}. Searching: {q[:60]}...")
            results = self.base_retriever.retrieve(q, top_k=top_k)
            
            # Deduplicate and add
            for result in results:
                result_id = result.get('chunk_id')
                if result_id not in seen_ids:
                    seen_ids.add(result_id)
                    # Add query variation info
                    result['query_variation'] = i
                    all_results.append(result)
        
        # Re-rank by score
        all_results.sort(key=lambda x: x['score'], reverse=True)
        
        # Return top_k
        return all_results[:top_k * 2]  # Return more results for better coverage
    
    def retrieve_with_fallback(
        self, 
        query: str, 
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Retrieve with fallback strategies.
        
        Args:
            query: User query
            top_k: Number of results
            
        Returns:
            Retrieved results
        """
        # Strategy 1: Normal semantic search
        results = self.base_retriever.retrieve(query, top_k=top_k)
        
        if results:
            print(f"✅ Found {len(results)} results with semantic search")
            return results
        
        # Strategy 2: Query expansion
        print("⚠️  No results, trying query expansion...")
        results = self.retrieve_with_expansion(query, top_k=top_k, use_expansion=True)
        
        if results:
            print(f"✅ Found {len(results)} results with expansion")
            return results
        
        # Strategy 3: Keyword-based (extract keywords only)
        print("⚠️  Still no results, trying keyword search...")
        keywords = self.optimizer._extract_keywords(query)
        results = self.base_retriever.retrieve(keywords, top_k=top_k)
        
        if results:
            print(f"✅ Found {len(results)} results with keywords: {keywords}")
            return results
        
        # Strategy 4: Lower threshold temporarily
        print("⚠️  Trying with lower threshold...")
        original_threshold = self.base_retriever.min_score
        self.base_retriever.min_score = 0.2  # Very low threshold
        results = self.base_retriever.retrieve(query, top_k=top_k * 2)
        self.base_retriever.min_score = original_threshold  # Restore
        
        if results:
            print(f"✅ Found {len(results)} results with lower threshold")
        else:
            print("❌ No results found with any strategy")
        
        return results


# Convenience function
def create_hybrid_retriever(base_retriever):
    """Create hybrid retriever with optimizer."""
    optimizer = QueryOptimizer()
    return HybridRetriever(base_retriever, optimizer)