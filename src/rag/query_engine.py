"""
Query Engine Module
-------------------
End-to-end RAG pipeline: Retrieve + Generate.

Features:
- Semantic retrieval
- Context formatting
- LLM generation using Groq (FREE)
- Streaming support (optional)
- Citation tracking
"""

from typing import List, Dict, Any, Optional
from config.settings import settings
from rag.retriever import Retriever
from groq import Groq


class QueryEngine:
    """RAG Query Engine: Retrieval + Generation using Groq."""
    
    def __init__(
        self, 
        top_k: Optional[int] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        collection_name: Optional[str] = None
    ):
        """
        Initialize query engine with Groq LLM.
        
        Args:
            top_k: Number of chunks to retrieve (defaults to RETRIEVAL_TOP_K)
            temperature: LLM temperature (defaults to LLM_TEMPERATURE)
            max_tokens: Max tokens for generation (defaults to LLM_MAX_TOKENS)
            collection_name: Qdrant collection to use (defaults to QDRANT_COLLECTION_NAME)
        """
        # Initialize retriever with specified collection
        self.retriever = Retriever(top_k=top_k, collection_name=collection_name)
        self.collection_name = collection_name
        
        # LLM settings from env or parameters
        self.temperature = temperature if temperature is not None else settings.LLM_TEMPERATURE
        self.max_tokens = max_tokens if max_tokens is not None else settings.LLM_MAX_TOKENS
        
        # Initialize Groq client
        self.llm_client = Groq(api_key=settings.GROQ_API_KEY)
        self.model = settings.GROQ_LLM_MODEL
        print(f"✅ Using Groq LLM: {self.model}")
    
    def _build_prompt(self, query: str, context: str) -> str:
        """
        Build RAG prompt with context.
        
        Args:
            query: User query
            context: Retrieved context
            
        Returns:
            Formatted prompt
        """
        prompt = f"""You are a helpful assistant that answers questions based on the provided context.

Context Information:
{context}

User Question: {query}

Instructions:
- Answer the question using ONLY the information from the context above
- If the question asks for a LIST (e.g., "which tables", "what are"), make sure to extract and list ALL relevant items from ALL chunks
- Be COMPREHENSIVE - don't just list examples, list EVERYTHING you find in the context
- Present your answer in a clean, organized format with bullet points or tables
- DO NOT mention "Chunk 1", "Chunk 2", etc. in your answer - just present the information directly
- Group related items together logically
- If the context doesn't contain enough information to answer, say so clearly

Answer:"""
        
        return prompt
    
    def query(self, question: str, top_k: Optional[int] = None) -> Dict[str, Any]:
        """
        Execute RAG query: Retrieve + Generate.
        
        Args:
            question: User question
            top_k: Number of chunks to retrieve (overrides default)
            
        Returns:
            Dictionary with answer, sources, and metadata
        """
        try:
            # Step 1: Retrieve relevant chunks
            print(f"🔍 Retrieving context for: '{question}'")
            retrieval_results = self.retriever.retrieve(question, top_k)
            
            # Debug: Print metadata structure
            if retrieval_results and len(retrieval_results) > 0:
                print(f"📋 DEBUG - First result metadata structure:")
                print(f"   Keys: {list(retrieval_results[0].keys())}")
                print(f"   Metadata: {retrieval_results[0].get('metadata', {})}")
            
            # Check if we got results
            if not retrieval_results:
                return {
                    'answer': "I couldn't find any relevant information in the database to answer your question. Try rephrasing or asking about different topics.",
                    'sources': [],
                    'num_sources': 0,
                    'avg_score': 0.0,  # Add default avg_score
                    'success': True,
                    'no_results': True,
                    'model': self.model,
                    'provider': 'groq'
                }
            
            print(f"✅ Retrieved {len(retrieval_results)} chunks")
            
            # Step 2: Format context
            context = self.retriever.format_context(retrieval_results)
            
            # Step 3: Build prompt
            prompt = self._build_prompt(question, context)
            
            # Step 4: Generate answer with Groq
            print(f"🤖 Generating answer with Groq ({self.model})...")
            
            response = self.llm_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            answer = response.choices[0].message.content
            
            # Step 5: Format response
            return {
                'answer': answer,
                'sources': retrieval_results,
                'num_sources': len(retrieval_results),
                'avg_score': sum(r['score'] for r in retrieval_results) / len(retrieval_results),
                'success': True,
                'model': self.model,
                'provider': 'groq'
            }
            
        except Exception as e:
            print(f"❌ Query error: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                'answer': f"Error processing query: {str(e)}",
                'sources': [],
                'num_sources': 0,
                'success': False,
                'error': str(e)
            }
    
    def query_with_chat_history(
        self, 
        question: str,
        chat_history: List[Dict[str, str]] = None,
        top_k: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Query with conversation history.
        
        Args:
            question: Current question
            chat_history: Previous messages [{"role": "user/assistant", "content": "..."}]
            top_k: Number of chunks to retrieve
            
        Returns:
            Response with answer and sources
        """
        # For now, we'll retrieve based on current question only
        # TODO: Implement conversation context handling
        
        result = self.query(question, top_k)
        result['chat_mode'] = True
        
        return result
    
    def format_response(self, result: Dict[str, Any]) -> str:
        """
        Format query result for display.
        
        Args:
            result: Query result dictionary
            
        Returns:
            Formatted string
        """
        if not result['success']:
            return f"❌ Error: {result.get('error', 'Unknown error')}"
        
        output = []
        output.append("=" * 70)
        output.append("ANSWER:")
        output.append("=" * 70)
        output.append(result['answer'])
        output.append("")
        
        if result['sources']:
            output.append("=" * 70)
            output.append(f"SOURCES ({result['num_sources']} chunks, avg score: {result['avg_score']:.3f}):")
            output.append("=" * 70)
            
            for i, source in enumerate(result['sources'], 1):
                output.append(f"\n[{i}] Score: {source['score']:.3f}")
                output.append(f"Text: {source['text'][:200]}...")
                
                metadata = source.get('metadata', {})
                if metadata:
                    if 'source' in metadata:
                        output.append(f"Source: {metadata['source']}")
        
        output.append("")
        output.append(f"Model: {result.get('model', 'Unknown')} ({result.get('provider', 'Unknown')})")
        
        return "\n".join(output)