"""
Run User Upload Chunking
-------------------------
Script untuk menjalankan user upload chunking.

Usage:
    1. Edit FILE_TO_PROCESS di bawah
    2. Run: py src/ingestion/run_user_chunking.py

Author: RAG Chatbot Builder
Date: 2025
"""

import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from user_upload_chunking import process_user_file


# ============================================================================
# CONFIGURATION: Set file yang mau di-proses
# ============================================================================

FILE_TO_PROCESS = "SOP_Hotel.pdf"  # Ganti dengan nama file kamu

CHUNK_SIZE = 512      # Maximum characters per chunk
CHUNK_OVERLAP = 50    # Characters overlap between chunks

# ============================================================================


def main():
    """Main function to run user chunking."""
    
    # Process file
    result = process_user_file(
        file_name=FILE_TO_PROCESS,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    
    # Check result
    if result['status'] == 'error':
        print(f"\n❌ ERROR: {result['message']}\n")
        print("💡 TIP: Edit FILE_TO_PROCESS in this script\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
