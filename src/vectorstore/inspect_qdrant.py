"""
Qdrant Database Inspector
--------------------------
Tool to inspect and explore your Qdrant vector database.

Features:
- View collection statistics
- Browse stored vectors and metadata
- Search and preview chunks
- Export data

Usage:
    python src/vectorstore/inspect_qdrant.py
"""

import sys
from pathlib import Path
import json

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from config.settings import settings
from vectorstore.qdrant_store import get_qdrant_client


def print_header(title):
    """Print section header."""
    print("\n" + "="*70)
    print(f" {title}")
    print("="*70 + "\n")


def show_collection_info(client):
    """Show collection information."""
    print_header("📊 COLLECTION INFORMATION")
    
    try:
        info = client.get_collection_info()
        
        if not info:
            print("❌ Collection does not exist or is empty\n")
            return False
        
        print(f"Collection Name: {info.get('name')}")
        print(f"Total Vectors: {info.get('points_count')}")
        print(f"Vector Dimension: {info.get('vector_size')}")
        print(f"Distance Metric: {info.get('distance')}")
        print(f"Status: {info.get('status')}")
        print("")
        
        return True
        
    except Exception as e:
        print(f"❌ Error getting collection info: {e}\n")
        return False


def browse_vectors(client, limit=10):
    """Browse stored vectors."""
    print_header(f"📚 BROWSING VECTORS (showing {limit} samples)")
    
    try:
        # Scroll through points
        points, _ = client.client.scroll(
            collection_name=settings.QDRANT_COLLECTION_NAME,
            limit=limit,
            with_payload=True,
            with_vectors=False  # Don't load full vectors (too large!)
        )
        
        if not points:
            print("❌ No vectors found in collection\n")
            return
        
        print(f"Found {len(points)} vectors\n")
        
        for i, point in enumerate(points, 1):
            print(f"{'─'*70}")
            print(f"Vector #{i}")
            print(f"{'─'*70}")
            print(f"ID: {point.id}")
            
            # Show metadata
            payload = point.payload
            
            # Show original ID if exists
            if 'original_id' in payload:
                print(f"Original ID: {payload['original_id']}")
            
            # Show text (truncated)
            text = payload.get('text', 'N/A')
            preview_length = 200
            if len(text) > preview_length:
                print(f"Text: {text[:preview_length]}...")
            else:
                print(f"Text: {text}")
            
            # Show other metadata
            print("\nMetadata:")
            for key, value in payload.items():
                if key not in ['text', 'original_id']:
                    print(f"  {key}: {value}")
            
            print("")
        
        print(f"{'─'*70}\n")
        
    except Exception as e:
        print(f"❌ Error browsing vectors: {e}\n")
        import traceback
        traceback.print_exc()


def search_by_keyword(client, keyword, limit=5):
    """Search vectors by keyword in metadata."""
    print_header(f"🔍 SEARCHING FOR: '{keyword}'")
    
    try:
        # Scroll through all points and filter by keyword
        points, _ = client.client.scroll(
            collection_name=settings.QDRANT_COLLECTION_NAME,
            limit=100,  # Search first 100
            with_payload=True,
            with_vectors=False
        )
        
        # Filter by keyword
        matches = []
        for point in points:
            text = point.payload.get('text', '').lower()
            if keyword.lower() in text:
                matches.append(point)
                if len(matches) >= limit:
                    break
        
        if not matches:
            print(f"❌ No matches found for '{keyword}'\n")
            return
        
        print(f"✅ Found {len(matches)} match(es)\n")
        
        for i, point in enumerate(matches, 1):
            print(f"{'─'*70}")
            print(f"Match #{i}")
            print(f"{'─'*70}")
            
            payload = point.payload
            text = payload.get('text', 'N/A')
            
            # Highlight keyword (simple version)
            if keyword.lower() in text.lower():
                # Find context around keyword
                idx = text.lower().find(keyword.lower())
                start = max(0, idx - 100)
                end = min(len(text), idx + len(keyword) + 100)
                context = text[start:end]
                
                print(f"Context: ...{context}...")
            else:
                print(f"Text: {text[:200]}...")
            
            print(f"\nSource: {payload.get('source_file', 'N/A')}")
            print("")
        
    except Exception as e:
        print(f"❌ Error searching: {e}\n")


def export_to_json(client, output_file, limit=None):
    """Export vectors to JSON file."""
    print_header(f"💾 EXPORTING TO JSON")
    
    try:
        print(f"Output file: {output_file}")
        
        # Scroll through all points
        all_points = []
        offset = None
        
        while True:
            points, offset = client.client.scroll(
                collection_name=settings.QDRANT_COLLECTION_NAME,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=True  # Include vectors for complete export
            )
            
            if not points:
                break
            
            all_points.extend(points)
            
            if offset is None or (limit and len(all_points) >= limit):
                break
        
        if limit:
            all_points = all_points[:limit]
        
        # Convert to JSON-serializable format
        export_data = []
        for point in all_points:
            export_data.append({
                'id': str(point.id),
                'vector': point.vector,
                'metadata': point.payload
            })
        
        # Write to file
        output_path = Path(output_file)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Exported {len(export_data)} vectors to {output_path}\n")
        
    except Exception as e:
        print(f"❌ Error exporting: {e}\n")


def get_vector_by_id(client, vector_id):
    """Get specific vector by ID."""
    print_header(f"🔍 RETRIEVING VECTOR: {vector_id}")
    
    try:
        # Try to retrieve by UUID
        from qdrant_client.models import PointIdsList
        
        result = client.client.retrieve(
            collection_name=settings.QDRANT_COLLECTION_NAME,
            ids=[vector_id],
            with_payload=True,
            with_vectors=True
        )
        
        if not result:
            print(f"❌ Vector not found: {vector_id}\n")
            return
        
        point = result[0]
        
        print(f"ID: {point.id}")
        print(f"\nVector (dimension {len(point.vector)}):")
        print(f"  First 10 values: {point.vector[:10]}")
        print(f"\nMetadata:")
        
        for key, value in point.payload.items():
            if key == 'text':
                print(f"  {key}: {value[:200]}..." if len(value) > 200 else f"  {key}: {value}")
            else:
                print(f"  {key}: {value}")
        
        print("")
        
    except Exception as e:
        print(f"❌ Error retrieving vector: {e}\n")


def interactive_menu():
    """Interactive menu for database inspection."""
    
    print("\n" + "🔍"*35)
    print("  QDRANT DATABASE INSPECTOR")
    print("🔍"*35 + "\n")
    
    print(f"📁 Storage: {settings.QDRANT_MODE} mode")
    print(f"📦 Collection: {settings.QDRANT_COLLECTION_NAME}\n")
    
    # Initialize client
    try:
        client = get_qdrant_client(create_collection=False)
    except Exception as e:
        print(f"❌ Failed to connect to Qdrant: {e}\n")
        return
    
    # Show collection info
    has_data = show_collection_info(client)
    
    if not has_data:
        print("💡 TIP: Index some data first:")
        print("   python src/ingestion/index_dataglossary.py")
        print("   python src/ingestion/index_user_upload.py\n")
        client.close()
        return
    
    # Menu loop
    while True:
        print("\n" + "─"*70)
        print("📋 MENU")
        print("─"*70)
        print("1. Browse vectors (10 samples)")
        print("2. Search by keyword")
        print("3. Get vector by ID")
        print("4. Export to JSON")
        print("5. Show collection info")
        print("6. Exit")
        print("")
        
        choice = input("Enter choice (1-6): ").strip()
        
        if choice == "1":
            browse_vectors(client, limit=10)
            
        elif choice == "2":
            keyword = input("\nEnter keyword to search: ").strip()
            if keyword:
                search_by_keyword(client, keyword, limit=5)
                
        elif choice == "3":
            vector_id = input("\nEnter vector UUID: ").strip()
            if vector_id:
                get_vector_by_id(client, vector_id)
                
        elif choice == "4":
            output_file = input("\nEnter output filename (e.g., export.json): ").strip()
            if not output_file:
                output_file = "qdrant_export.json"
            limit = input("Limit (press Enter for all): ").strip()
            limit = int(limit) if limit else None
            export_to_json(client, output_file, limit)
            
        elif choice == "5":
            show_collection_info(client)
            
        elif choice == "6":
            print("\n👋 Goodbye!\n")
            break
            
        else:
            print("\n❌ Invalid choice. Try again.\n")
    
    # Close client
    client.close()


def main():
    """Main entry point."""
    interactive_menu()


if __name__ == "__main__":
    main()
