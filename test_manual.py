"""
Quick Manual Test for RAG Query Engine
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from config.settings import settings
from rag.query_engine import QueryEngine

print("\n" + "="*70)
print("🧪 MANUAL RAG TEST")
print("="*70 + "\n")

# Show config
print(f"📊 Configuration:")
print(f"   Embedding: {settings.EMBEDDING_PROVIDER}")
print(f"   LLM: {settings.LLM_PROVIDER}")
print(f"   Vector DB: {settings.QDRANT_COLLECTION_NAME}")
print("")

# Initialize engine
print("🔧 Initializing QueryEngine...")
try:
    engine = QueryEngine()
    print("✅ QueryEngine initialized!\n")
except Exception as e:
    print(f"❌ Failed to initialize: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test query
test_question = "Which tables use incremental extraction with watermark datetime?"
print(f"❓ Question: {test_question}\n")

print("⏳ Processing query...\n")

try:
    result = engine.query(test_question)
    
    print("="*70)
    print("RESULT:")
    print("="*70)
    print(f"Success: {result.get('success', 'N/A')}")
    print(f"Num Sources: {result.get('num_sources', 'N/A')}")
    print(f"Avg Score: {result.get('avg_score', 'N/A')}")
    print(f"Model: {result.get('model', 'N/A')}")
    print(f"Provider: {result.get('provider', 'N/A')}")
    print("")
    
    if result.get('success'):
        print("="*70)
        print("ANSWER:")
        print("="*70)
        print(result['answer'])
        print("")
        
        print("="*70)
        print(f"SOURCES ({result['num_sources']}):")
        print("="*70)
        for i, source in enumerate(result['sources'], 1):
            print(f"\n[{i}] Score: {source['score']:.3f}")
            print(f"    Text: {source['text'][:150]}...")
        print("")
        
        print("✅ Test completed successfully!")
    else:
        print("="*70)
        print("ERROR:")
        print("="*70)
        print(f"Error message: {result.get('error', 'Unknown error')}")
        print(f"\nFull result: {result}")
        
except Exception as e:
    print("="*70)
    print("EXCEPTION OCCURRED:")
    print("="*70)
    print(f"Error: {str(e)}")
    print("\nFull traceback:")
    import traceback
    traceback.print_exc()

print("\n" + "="*70 + "\n")