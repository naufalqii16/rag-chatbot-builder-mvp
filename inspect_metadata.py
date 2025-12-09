"""
Quick script to inspect Qdrant metadata structure
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / 'src'))

from config.settings import settings
from vectorstore.qdrant_store import get_qdrant_client

print("\n" + "="*70)
print("🔍 INSPECTING QDRANT METADATA")
print("="*70 + "\n")

# Get client
client = get_qdrant_client(
    collection_name=settings.QDRANT_USER_UPLOAD_COLLECTION,
    create_collection=False
)

# Get collection info
info = client.get_collection_info()
print(f"📊 Collection: {info.get('name')}")
print(f"   Points: {info.get('points_count')}")
print(f"   Status: {info.get('status')}")
print("")

# Get a sample point
if info.get('points_count', 0) > 0:
    print("📋 Fetching sample point...")
    
    try:
        # Get first point
        points = client.client.scroll(
            collection_name=settings.QDRANT_USER_UPLOAD_COLLECTION,
            limit=1,
            with_payload=True,
            with_vectors=False
        )[0]
        
        if points:
            sample = points[0]
            print(f"\n✅ Sample Point:")
            print(f"   ID: {sample.id}")
            print(f"   Payload keys: {list(sample.payload.keys())}")
            print(f"\n   Full Payload:")
            for key, value in sample.payload.items():
                if key == 'text':
                    print(f"      {key}: {str(value)[:100]}...")
                else:
                    print(f"      {key}: {value}")
            
            # Check if metadata.source exists
            if 'metadata' in sample.payload:
                print(f"\n   ✅ 'metadata' field found!")
                print(f"      Type: {type(sample.payload['metadata'])}")
                print(f"      Content: {sample.payload['metadata']}")
            else:
                print(f"\n   ⚠️ 'metadata' field NOT found in payload")
            
            if 'source' in sample.payload:
                print(f"\n   ✅ 'source' field found at top level: {sample.payload['source']}")
            else:
                print(f"\n   ⚠️ 'source' field NOT found at top level")
                
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
else:
    print("⚠️ No points in collection yet. Upload and index some files first.")

print("\n" + "="*70)
print("✅ Inspection complete")
print("="*70 + "\n")
