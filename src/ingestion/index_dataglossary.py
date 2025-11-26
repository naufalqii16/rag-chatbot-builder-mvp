"""
Index DataGlossary to Vector Database
--------------------------------------
This script connects the DataGlossary chunking → embedding → Qdrant indexing pipeline.

Flow:
1. Load chunks from: data/DataGlossary_Chunks/DataGlossary_chunks.json
2. Generate embeddings using HuggingFace/OpenAI
3. Store in Qdrant vector database

Usage:
    python src/ingestion/index_dataglossary.py
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from config.settings import settings
from vectorstore.index_builder import IndexBuilder


def index_dataglossary(
    chunks_file: str = None,
    batch_size: int = None,
    recreate_collection: bool = False
):
    """
    Index DataGlossary chunks into Qdrant.
    
    Args:
        chunks_file: Path to chunks JSON file (default: auto-detect)
        batch_size: Batch size for embedding generation (default: based on provider)
        recreate_collection: Whether to recreate the collection (deletes existing data!)
    
    Returns:
        dict: Result with statistics
    """
    print("\n" + "="*70)
    print("🚀 INDEXING DATA GLOSSARY TO VECTOR DATABASE")
    print("="*70 + "\n")
    
    # Auto-detect chunks file if not provided
    if chunks_file is None:
        chunks_file = settings.CHUNKS_DIR / "DataGlossary_chunks.json"
    else:
        chunks_file = Path(chunks_file)
    
    if not chunks_file.exists():
        print(f"❌ Chunks file not found: {chunks_file}\n")
        print("💡 TIP: Run the chunking script first:")
        print("   python src/ingestion/dataGlossary_chunking_preprocessing.py\n")
        return {
            'status': 'error',
            'message': f'Chunks file not found: {chunks_file}'
        }
    
    # Set batch size based on provider if not specified
    if batch_size is None:
        if settings.EMBEDDING_PROVIDER == "huggingface":
            batch_size = 20  # HuggingFace can handle larger batches locally
        else:
            batch_size = 100  # OpenAI API is faster (tapi bayar)
    
    print(f"📂 Chunks file: {chunks_file.name}")
    print(f"🧠 Embedding provider: {settings.EMBEDDING_PROVIDER}")
    print(f"⚙️  Batch size: {batch_size}")
    print(f"🔄 Recreate collection: {recreate_collection}\n")
    
    # Initialize index builder with glossary collection
    try:
        builder = IndexBuilder(collection_name=settings.QDRANT_GLOSSARY_COLLECTION)
        
        # Load chunks
        print("="*70)
        print("STEP 1: LOADING CHUNKS")
        print("="*70 + "\n")
        
        chunks = builder.load_chunks_from_json(str(chunks_file))
        
        if not chunks:
            print("❌ No chunks loaded!\n")
            return {
                'status': 'error',
                'message': 'No chunks loaded from file'
            }
        
        print(f"✅ Loaded {len(chunks)} chunks\n")
        
        # Build index
        print("="*70)
        print("STEP 2: GENERATING EMBEDDINGS & INDEXING")
        print("="*70 + "\n")
        
        success = builder.build_index(
            chunks=chunks,
            batch_size=batch_size,
            create_new_collection=recreate_collection
        )
        
        if success:
            print("\n" + "="*70)
            print("✅ INDEXING COMPLETED SUCCESSFULLY!")
            print("="*70 + "\n")
            
            # Get statistics
            from vectorstore.qdrant_store import get_qdrant_client
            client = get_qdrant_client(
                collection_name=settings.QDRANT_GLOSSARY_COLLECTION,
                create_collection=False
            )
            info = client.get_collection_info()
            client.close()
            
            print("📊 Collection Statistics:")
            print(f"   Name: {info.get('name')}")
            print(f"   Total vectors: {info.get('points_count')}")
            print(f"   Dimension: {info.get('vector_size')}")
            print(f"   Distance metric: {info.get('distance')}")
            print("")
            
            print("💡 Next Steps:")
            print("   1. Test retrieval: python test_embedding_retrieval.py")
            print("   2. Or start the chatbot: streamlit run app.py")
            print("")
            
            return {
                'status': 'success',
                'message': 'DataGlossary indexed successfully',
                'statistics': {
                    'total_chunks': len(chunks),
                    'total_vectors': info.get('points_count'),
                    'dimension': info.get('vector_size'),
                    'provider': settings.EMBEDDING_PROVIDER
                }
            }
        else:
            print("\n❌ Indexing failed\n")
            return {
                'status': 'error',
                'message': 'Indexing failed'
            }
            
    except Exception as e:
        print(f"\n❌ Error: {e}\n")
        import traceback
        traceback.print_exc()
        return {
            'status': 'error',
            'message': str(e)
        }


def main():
    """Main entry point."""
    
    # Configuration
    RECREATE_COLLECTION = True  # Set to True to start fresh (deletes existing data!)
    
    # Run indexing
    result = index_dataglossary(
        recreate_collection=RECREATE_COLLECTION
    )
    
    if result['status'] == 'error':
        print(f"❌ ERROR: {result['message']}\n")
        sys.exit(1)
    else:
        print("🎉 Success!\n")
        sys.exit(0)


if __name__ == "__main__":
    main()
