"""
Embedding & Retrieval Testing Script
-------------------------------------
Untuk ngetes keseluruhan pipeline RAG

This script tests:
1. Configuration loading
2. Qdrant connection
3. Embedding generation (OpenAI/HuggingFace)
4. Vector indexing
5. Semantic search
6. Query engine
7. End-to-end RAG pipeline

Supports:
- OpenAI embeddings (paid)
- HuggingFace embeddings (FREE, local)
- Groq LLM (FREE)

Usage:
    python test_embedding_retrieval.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from config.settings import settings
from vectorstore.qdrant_store import get_qdrant_client
from vectorstore.index_builder import IndexBuilder, EmbeddingGenerator
from rag.retriever import Retriever
from rag.query_engine import QueryEngine


def print_section(title: str):
    """Print section header."""
    print("\n" + "="*70)
    print(f"{'='*5} {title} {'='*5}")
    print("="*70 + "\n")


def test_configuration():
    """Test 1: Configuration loading."""
    print_section("TEST 1: CONFIGURATION")
    
    settings.display()
    
    # Check embedding provider
    print(f"\n🔍 Embedding Provider: {settings.EMBEDDING_PROVIDER}")
    if settings.EMBEDDING_PROVIDER == "huggingface":
        print(f"   Model: {getattr(settings, 'HUGGINGFACE_MODEL', 'all-mpnet-base-v2')}")
        print(f"   Dimension: {settings.EMBEDDING_DIMENSION}")
        print("   ✅ Using FREE local embeddings!")
    elif settings.EMBEDDING_PROVIDER == "openai":
        print(f"   Model: {settings.OPENAI_EMBEDDING_MODEL}")
        print(f"   Dimension: {settings.EMBEDDING_DIMENSION}")
        if settings.OPENAI_API_KEY:
            print("   ✅ OpenAI API key found")
        else:
            print("   ❌ OpenAI API key missing!")
            return False
    
    # Check LLM provider
    print(f"\n🤖 LLM Provider: {settings.LLM_PROVIDER}")
    if settings.LLM_PROVIDER == "groq":
        print(f"   Model: {settings.GROQ_LLM_MODEL}")
        print("   ✅ Using FREE Groq LLM!")
    
    if settings.validate():
        print("\n✅ Configuration is valid\n")
        return True
    else:
        print("\n❌ Configuration validation failed\n")
        print("💡 TIP: Make sure .env file is configured correctly")
        print("   See .env.example for reference\n")
        return False


def test_qdrant_connection():
    """Test 2: Qdrant connection."""
    print_section("TEST 2: QDRANT CONNECTION")
    
    client = None
    try:
        client = get_qdrant_client(create_collection=False)
        
        # Try to get collection info
        info = client.get_collection_info()
        
        print("✅ Qdrant connection successful")
        print(f"   Collection: {info.get('name')}")
        print(f"   Points: {info.get('points_count')}")
        print(f"   Dimension: {info.get('vector_size')}")
        print("")
        
        result = True
        
    except Exception as e:
        print(f"❌ Qdrant connection failed: {e}\n")
        result = False
    
    finally:
        # IMPORTANT: Close the client to release the lock
        if client:
            client.close()
    
    return result, None  # Don't return client to avoid lock issues


def test_embedding_generation():
    """Test 3: Embedding generation."""
    print_section("TEST 3: EMBEDDING GENERATION")
    
    try:
        print(f"🔧 Initializing {settings.EMBEDDING_PROVIDER} embedder...\n")
        generator = EmbeddingGenerator()
        
        # Test single embedding
        test_text = "This is a test sentence for embedding generation."
        print(f"📝 Test text: {test_text}\n")
        
        print("⏳ Generating embedding...")
        embedding = generator.generate_embedding(test_text)
        
        if embedding:
            print(f"\n✅ Embedding generated successfully")
            print(f"   Provider: {settings.EMBEDDING_PROVIDER}")
            print(f"   Dimension: {len(embedding)}")
            print(f"   Expected: {settings.EMBEDDING_DIMENSION}")
            
            # Validate dimension
            if len(embedding) == settings.EMBEDDING_DIMENSION:
                print(f"   ✅ Dimension matches!")
            else:
                print(f"   ⚠️  Dimension mismatch! Update EMBEDDING_DIMENSION in .env")
            
            print(f"   First 5 values: {[f'{v:.4f}' for v in embedding[:5]]}")
            print("")
            return True, generator
        else:
            print("❌ Failed to generate embedding\n")
            return False, None
            
    except Exception as e:
        print(f"❌ Embedding generation failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False, None


def test_indexing():
    """Test 4: Vector indexing."""
    print_section("TEST 4: VECTOR INDEXING")
    
    # Check if chunks exist
    chunks_file = settings.CHUNKS_DIR / "DataGlossary_chunks.json"
    
    if not chunks_file.exists():
        print(f"⚠️  Chunks file not found: {chunks_file}")
        print("   Run chunking scripts first:")
        print("   > python src/ingestion/dataGlossary_chunking_preprocessing.py\n")
        return False
    
    try:
        print(f"📂 Chunks file: {chunks_file.name}\n")
        
        # Build index (limit for faster testing)
        # HuggingFace is slower than OpenAI, so use smaller batch
        if settings.EMBEDDING_PROVIDER == "huggingface":
            test_limit = 50  # Fewer chunks for local embeddings
            batch_size = 10
            print("   Using smaller batch for HuggingFace (slower but FREE)")
        else:
            test_limit = 100
            batch_size = 20
        
        builder = IndexBuilder()
        chunks = builder.load_chunks_from_json(str(chunks_file))
        
        # Limit for testing
        test_chunks = chunks[:test_limit] if len(chunks) > test_limit else chunks
        print(f"   Using {len(test_chunks)} chunks for testing")
        print(f"   Batch size: {batch_size}\n")
        
        success = builder.build_index(
            chunks=test_chunks,
            batch_size=batch_size,
            create_new_collection=True  # Recreate for clean test
        )
        
        if success:
            print("\n✅ Indexing test passed\n")
            return True
        else:
            print("\n❌ Indexing test failed\n")
            return False
            
    except Exception as e:
        print(f"❌ Indexing failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_retrieval():
    """Test 5: Semantic search."""
    print_section("TEST 5: SEMANTIC SEARCH")
   


def test_query_engine():
    """Test 6: Query engine (RAG)."""
    print_section("TEST 6: QUERY ENGINE (RAG)")
   


def run_all_tests():
    """Run all tests."""
    print("\n" + "🧪"*35)
    print("  RAG EMBEDDING & RETRIEVAL TEST SUITE")
    print("🧪"*35 + "\n")
    
    # Show configuration summary
    print("📋 Configuration Summary:")
    print(f"   Embeddings: {settings.EMBEDDING_PROVIDER}")
    if settings.EMBEDDING_PROVIDER == "huggingface":
        print(f"   Model: {getattr(settings, 'HUGGINGFACE_MODEL', 'all-mpnet-base-v2')}")
    print(f"   LLM: {settings.LLM_PROVIDER}")
    print(f"   Vector DB: Qdrant ({settings.QDRANT_MODE})")
    print("")
    
    results = {}
    
    # Test 1: Configuration
    results['configuration'] = test_configuration()
    if not results['configuration']:
        print("⛔ Stopping tests: Configuration invalid\n")
        return results
    
    # Test 2: Qdrant connection
    results['qdrant'], _ = test_qdrant_connection()
    if not results['qdrant']:
        print("⚠️  Qdrant connection failed, but continuing with other tests\n")
    
    # Test 3: Embedding generation
    results['embedding'], generator = test_embedding_generation()
    if not results['embedding']:
        print("⛔ Stopping tests: Embedding generation failed\n")
        return results
    
    # Test 4: Indexing
    results['indexing'] = test_indexing()
    if not results['indexing']:
        print("⚠️  Indexing failed, retrieval tests may not work\n")
    
    # Test 5: Retrieval
    if results['indexing']:
        results['retrieval'] = test_retrieval()
    else:
        results['retrieval'] = False
        print("⏭️  Skipping retrieval test (no index)\n")
    
    # Test 6: Query engine
    if results['retrieval']:
        results['query_engine'] = test_query_engine()
    else:
        results['query_engine'] = False
        print("⏭️  Skipping query engine test (retrieval failed)\n")
    
    # Summary
    print_section("TEST SUMMARY")
    
    print("Test Results:")
    print(f"   1. Configuration      : {'✅ PASS' if results['configuration'] else '❌ FAIL'}")
    print(f"   2. Qdrant Connection  : {'✅ PASS' if results['qdrant'] else '❌ FAIL'}")
    print(f"   3. Embedding Generation: {'✅ PASS' if results['embedding'] else '❌ FAIL'}")
    print(f"   4. Vector Indexing    : {'✅ PASS' if results['indexing'] else '❌ FAIL'}")
    print(f"   5. Semantic Search    : {'✅ PASS' if results['retrieval'] else '❌ FAIL'}")
    print(f"   6. Query Engine       : {'✅ PASS' if results['query_engine'] else '❌ FAIL'}")
    
    total = len(results)
    passed = sum(1 for v in results.values() if v)
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Your RAG pipeline is working correctly.")
        if settings.EMBEDDING_PROVIDER == "huggingface":
            print("   💰 100% FREE setup using HuggingFace + Groq!")
        print("")
    elif passed >= total - 1:
        print("\n⚠️  Most tests passed. Check failed tests above.\n")
    else:
        print("\n❌ Multiple tests failed. Review errors above.\n")
    
    return results


if __name__ == "__main__":
    results = run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if all(results.values()) else 1)
