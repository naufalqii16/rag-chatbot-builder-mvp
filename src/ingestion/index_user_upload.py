"""
Index User Upload to Vector Database
-------------------------------------
This script connects user upload chunking → embedding → Qdrant indexing pipeline.

Flow:
1. Process & chunk user file (cleaning + chunking)
2. Generate embeddings using HuggingFace/OpenAI
3. Store in Qdrant vector database

Usage:
    python src/ingestion/index_user_upload.py

    Or import and use programmatically:
    from index_user_upload import index_user_file
    result = index_user_file("SOP_Hotel.pdf")
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))

from user_upload_chunking import process_user_file
from config.settings import settings
from vectorstore.index_builder import IndexBuilder


def index_user_file(
    file_name: str,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    batch_size: int = None,
    append_to_existing: bool = True
):
    """
    Process and index a user-uploaded file.
    
    This is the main function to use from Streamlit or other interfaces
    
    Args:
        file_name: Name of file in data/ folder (e.g., "SOP_Hotel.pdf")
        chunk_size: Max characters per chunk (default: 512)
        chunk_overlap: Overlap characters (default: 50)
        batch_size: Batch size for embeddings (default: auto-detect)
        append_to_existing: Add to existing collection (True) or recreate (False)
    
    Returns:
        dict: {
            'status': 'success' or 'error',
            'message': str,
            'statistics': dict with counts
        }
    """
    print("\n" + "="*70)
    print("🚀 INDEXING USER UPLOADED FILES TO VECTOR DATABASE")
    print("="*70 + "\n")
    
    print(f"📂 File: {file_name}")
    print(f"🧠 Embedding provider: {settings.EMBEDDING_PROVIDER}")
    print(f"➕ Append to existing: {append_to_existing}\n")
    
    # STEP 1: Process file (cleaning + chunking)
    print("="*70)
    print("STEP 1: CLEANING & CHUNKING FILE")
    print("="*70 + "\n")
    
    try:
        result = process_user_file(
            file_name=file_name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        if result['status'] != 'success':
            return {
                'status': 'error',
                'message': f"Chunking failed: {result['message']}",
                'statistics': {}
            }
        
        chunks = result['chunks']
        
        if not chunks:
            return {
                'status': 'error',
                'message': 'No chunks generated from file',
                'statistics': {}
            }
        
        print(f"\n✅ Generated {len(chunks)} chunks\n")
        
    except Exception as e:
        return {
            'status': 'error',
            'message': f'Chunking error: {str(e)}',
            'statistics': {}
        }
    
    # STEP 2: Generate embeddings & index
    print("="*70)
    print("STEP 2: GENERATING EMBEDDINGS & INDEXING")
    print("="*70 + "\n")
    
    # Set batch size based on provider if not specified
    if batch_size is None:
        if settings.EMBEDDING_PROVIDER == "huggingface":
            batch_size = 20
        else:
            batch_size = 100
    
    print(f"⚙️  Batch size: {batch_size}")
    print(f"🔄 Recreate collection: {not append_to_existing}\n")
    
    try:
        builder = IndexBuilder()
        
        success = builder.build_index(
            chunks=chunks,
            batch_size=batch_size,
            create_new_collection=not append_to_existing
        )
        
        if success:
            print("\n" + "="*70)
            print("✅ INDEXING COMPLETED SUCCESSFULLY!")
            print("="*70 + "\n")
            
            # Get statistics
            from vectorstore.qdrant_store import get_qdrant_client
            client = get_qdrant_client(create_collection=False)
            info = client.get_collection_info()
            client.close()
            
            print("📊 Collection Statistics:")
            print(f"   Name: {info.get('name')}")
            print(f"   Total vectors: {info.get('points_count')}")
            print(f"   Dimension: {info.get('vector_size')}")
            print(f"   Distance metric: {info.get('distance')}")
            print("")
            
            print("💡 Next Steps:")
            print("   1. Upload another file (run this script again)")
            print("   2. Test the chatbot: streamlit run app.py")
            print("")
            
            return {
                'status': 'success',
                'message': f'Successfully indexed {file_name}',
                'statistics': {
                    'file_name': file_name,
                    'chunks_generated': len(chunks),
                    'total_vectors_in_db': info.get('points_count'),
                    'dimension': info.get('vector_size'),
                    'provider': settings.EMBEDDING_PROVIDER
                }
            }
        else:
            return {
                'status': 'error',
                'message': 'Indexing failed',
                'statistics': {}
            }
            
    except Exception as e:
        print(f"\n❌ Error: {e}\n")
        import traceback
        traceback.print_exc()
        return {
            'status': 'error',
            'message': f'Indexing error: {str(e)}',
            'statistics': {}
        }


def main():
    """Main entry point for command-line usage."""
    
    # =========================================================================
    # CONFIGURATION: Edit this to process different files
    # =========================================================================
    
    FILE_TO_INDEX = "SOP_Hotel.pdf"  # Change to your file name
    
    CHUNK_SIZE = 512        # Max characters per chunk
    CHUNK_OVERLAP = 50      # Overlap between chunks
    
    APPEND_TO_EXISTING = True  # True = add to existing, False = recreate collection
    
    # =========================================================================
    
    # Run indexing
    result = index_user_file(
        file_name=FILE_TO_INDEX,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        append_to_existing=APPEND_TO_EXISTING
    )
    
    if result['status'] == 'error':
        print(f"\n❌ ERROR: {result['message']}\n")
        print("💡 TIP: Make sure the file exists in the data/ folder\n")
        sys.exit(1)
    else:
        print("🎉 Success!\n")
        sys.exit(0)


if __name__ == "__main__":
    main()
