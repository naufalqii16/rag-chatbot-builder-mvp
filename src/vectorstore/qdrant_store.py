"""
Qdrant Vector Store Client
---------------------------
Manages connection to Qdrant and provides vector store operations.

Features:
- Local or cloud Qdrant support
- Collection creation and management
- Vector insertion with metadata
- Semantic search (cosine similarity)
"""

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    SearchParams
)
from typing import List, Dict, Any, Optional
import uuid
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config.settings import settings


class QdrantVectorStore:
    """
    Wrapper class for Qdrant vector database operations.
    """
    
    def __init__(
        self,
        collection_name: Optional[str] = None,
        embedding_dimension: Optional[int] = None,
        mode: Optional[str] = None
    ):
        """
        Initialize Qdrant client.
        
        Args:
            collection_name: Name of the collection (defaults to settings)
            embedding_dimension: Vector dimension (defaults to settings)
            mode: 'local', 'server', or 'cloud' (defaults to settings)
        """
        self.collection_name = collection_name or settings.QDRANT_COLLECTION_NAME
        self.embedding_dimension = embedding_dimension or settings.EMBEDDING_DIMENSION
        self.mode = mode or settings.QDRANT_MODE
        
        # Initialize client
        self.client = self._initialize_client()
        
        print(f"✅ Qdrant client initialized ({self.mode} mode)")
    
    def _initialize_client(self) -> QdrantClient:
        """
        Initialize Qdrant client based on mode.
        
        Returns:
            QdrantClient: Initialized client
        """
        if self.mode == "local":
            # Local mode: store in file system
            storage_path = settings.QDRANT_LOCAL_PATH
            storage_path.mkdir(parents=True, exist_ok=True)
            
            client = QdrantClient(path=str(storage_path))
            print(f"📁 Local Qdrant storage: {storage_path}")
            
        elif self.mode == "server":
            # Server mode: connect to Qdrant server (e.g., Docker)
            if not settings.QDRANT_URL:
                raise ValueError(
                    "QDRANT_URL is required for server mode. "
                    "Please set it in your .env file (e.g., http://localhost:6333)"
                )
            
            client = QdrantClient(url=settings.QDRANT_URL)
            print(f"🌐 Connected to Qdrant Server: {settings.QDRANT_URL}")
            
        elif self.mode == "cloud":
            # Cloud mode: connect to Qdrant cloud
            if not settings.QDRANT_URL or not settings.QDRANT_API_KEY:
                raise ValueError(
                    "QDRANT_URL and QDRANT_API_KEY are required for cloud mode. "
                    "Please set them in your .env file."
                )
            
            client = QdrantClient(
                url=settings.QDRANT_URL,
                api_key=settings.QDRANT_API_KEY
            )
            print(f"☁️  Connected to Qdrant Cloud: {settings.QDRANT_URL}")
        
        else:
            raise ValueError(
                f"Invalid QDRANT_MODE: {self.mode}. "
                f"Use 'local', 'server', or 'cloud'."
            )
        
        return client
    
    def create_collection(self, recreate: bool = False) -> bool:
        """
        Create collection if it doesn't exist.
        
        Args:
            recreate: If True, delete existing collection and create new one
            
        Returns:
            bool: True if created/exists, False on error
        """
        try:
            # Check if collection exists
            collections = self.client.get_collections().collections
            collection_names = [col.name for col in collections]
            
            if self.collection_name in collection_names:
                if recreate:
                    print(f"🗑️  Deleting existing collection: {self.collection_name}")
                    self.client.delete_collection(self.collection_name)
                else:
                    print(f"✅ Collection already exists: {self.collection_name}")
                    return True
            
            # Create collection
            print(f"🔨 Creating collection: {self.collection_name}")
            print(f"   Dimension: {self.embedding_dimension}")
            print(f"   Distance: Cosine")
            
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_dimension,
                    distance=Distance.COSINE  # Cosine similarity
                )
            )
            
            print(f"✅ Collection created: {self.collection_name}")
            return True
            
        except Exception as e:
            print(f"❌ Error creating collection: {e}")
            return False
    
    def insert_vectors(
        self,
        vectors: List[List[float]],
        metadatas: List[Dict[str, Any]],
        ids: Optional[List[str]] = None
    ) -> bool:
        """
        Insert vectors with metadata into Qdrant.
        
        Args:
            vectors: List of embedding vectors
            metadatas: List of metadata dictionaries (must match vectors length)
            ids: Optional list of IDs (will be converted to UUIDs or generated)
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if len(vectors) != len(metadatas):
                raise ValueError(f"Vectors ({len(vectors)}) and metadatas ({len(metadatas)}) length mismatch")
            
            # Generate UUIDs - either from provided IDs or random
            if ids is None:
                point_ids = [str(uuid.uuid4()) for _ in range(len(vectors))]
            else:
                # Convert custom IDs to deterministic UUIDs
                point_ids = []
                for custom_id in ids:
                    # Use uuid5 to create deterministic UUID from custom ID
                    point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, str(custom_id)))
                    point_ids.append(point_id)
            
            # Create points with original IDs stored in metadata
            points = []
            for idx, (vector, metadata) in enumerate(zip(vectors, metadatas)):
                # Store original ID in metadata if provided
                if ids is not None:
                    metadata['original_id'] = ids[idx]
                
                point = PointStruct(
                    id=point_ids[idx],  # UUID format
                    vector=vector,
                    payload=metadata  # Store metadata as payload
                )
                points.append(point)
            
            # Batch upload
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            
            print(f"✅ Inserted {len(points)} vectors into {self.collection_name}")
            return True
            
        except Exception as e:
            print(f"❌ Error inserting vectors: {e}")
            return False
    
    def search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        score_threshold: Optional[float] = None,
        filter_conditions: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform semantic search using cosine similarity.
        
        Args:
            query_vector: Embedding vector of the query
            top_k: Number of results to return
            score_threshold: Minimum similarity score (0.0 to 1.0)
            filter_conditions: Optional metadata filters
            
        Returns:
            List of dictionaries containing:
                - id: Point ID
                - score: Similarity score
                - text: Chunk text
                - metadata: All metadata fields
        """
        try:
            # Try new API first (qdrant-client >= 1.8.0)
            try:
                search_results = self.client.query_points(
                    collection_name=self.collection_name,
                    query=query_vector,
                    limit=top_k,
                    score_threshold=score_threshold,
                ).points
            except (AttributeError, TypeError):
                # Fallback to old API (qdrant-client < 1.8.0)
                search_results = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=query_vector,
                    limit=top_k,
                    score_threshold=score_threshold,
                )
            
            # Format results
            results = []
            for hit in search_results:
                result = {
                    'id': hit.id,
                    'score': hit.score,
                    'text': hit.payload.get('text', ''),
                    'metadata': hit.payload
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            print(f"❌ Error during search: {e}")
            return []
    
    def count_vectors(self) -> int:
        """
        Count total vectors in collection.
        
        Returns:
            int: Number of vectors
        """
        try:
            collection_info = self.client.get_collection(self.collection_name)
            return collection_info.points_count
        except Exception as e:
            print(f"❌ Error counting vectors: {e}")
            return 0
    
    def delete_collection(self) -> bool:
        """
        Delete the collection.
        
        Returns:
            bool: True if successful
        """
        try:
            self.client.delete_collection(self.collection_name)
            print(f"🗑️  Deleted collection: {self.collection_name}")
            return True
        except Exception as e:
            print(f"❌ Error deleting collection: {e}")
            return False
    
    def get_collection_info(self) -> Dict[str, Any]:
        """
        Get collection information.
        
        Returns:
            dict: Collection info
        """
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                'name': self.collection_name,
                'points_count': info.points_count,
                'vector_size': info.config.params.vectors.size,
                'distance': info.config.params.vectors.distance.name,
                'status': info.status.name
            }
        except Exception as e:
            print(f"❌ Error getting collection info: {e}")
            return {}
    
    def close(self):
        """Close the Qdrant client and release resources."""
        try:
            if hasattr(self.client, 'close'):
                self.client.close()
        except:
            pass


# Convenience function to get client instance
def get_qdrant_client(
    collection_name: Optional[str] = None,
    embedding_dimension: Optional[int] = None,
    mode: Optional[str] = None,
    create_collection: bool = True
) -> QdrantVectorStore:
    """
    Get initialized Qdrant client.
    
    Args:
        collection_name: Collection name (defaults to settings)
        embedding_dimension: Vector dimension (defaults to settings)
        mode: 'local' or 'cloud' (defaults to settings)
        create_collection: Whether to create collection if not exists
        
    Returns:
        QdrantVectorStore: Initialized client
    """
    client = QdrantVectorStore(
        collection_name=collection_name,
        embedding_dimension=embedding_dimension,
        mode=mode
    )
    
    if create_collection:
        client.create_collection()
    
    return client


if __name__ == "__main__":
    # Test the Qdrant client
    print("\n" + "="*70)
    print("🧪 Testing Qdrant Client")
    print("="*70 + "\n")
    
    # Initialize client
    client = get_qdrant_client(create_collection=True)
    
    # Get collection info
    info = client.get_collection_info()
    print(f"\n📊 Collection Info:")
    for key, value in info.items():
        print(f"   {key}: {value}")
    
    print("\n✅ Qdrant client test completed!\n")
