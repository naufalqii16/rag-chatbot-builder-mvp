"""
DataGlossary Chunking & Preprocessing Module
---------------------------------------------
This module handles chunking and preprocessing specifically for DataGlossary.xlsx

Tasks:
1. Define chunk parameters (chunk size & overlap)
2. Split data by row, paragraph, or long definition
3. Normalize text (lowercase, remove special characters)
4. Save chunking results to data/DataGlossary_Chunks/DataGlossary_chunks.json
5. Generate chunking statistics

Author: RAG Chatbot Builder
Date: 2025
"""

import pandas as pd
import numpy as np
import json
import re
import os
from typing import List, Dict, Tuple
from datetime import datetime


class DataGlossaryChunker:
    """
    A class to handle text chunking and preprocessing for DataGlossary data.
    """
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        """
        Initialize the DataGlossaryChunker with configurable parameters.
        
        Args:
            chunk_size (int): Maximum number of characters per chunk
            chunk_overlap (int): Number of characters to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunks = []
        self.statistics = {
            'total_chunks': 0,
            'average_token_length': 0,
            'min_chunk_length': 0,
            'max_chunk_length': 0,
            'total_rows_processed': 0
        }
    
    def normalize_text(self, text: str) -> str:
        """
        Normalize text by:
        - Converting to lowercase
        - Removing special characters (keeping alphanumeric, spaces, and basic punctuation)
        - Removing extra whitespace
        
        Args:
            text (str): Input text to normalize
            
        Returns:
            str: Normalized text
        """
        if pd.isnull(text) or text is None:
            return ""
        
        # Convert to string and lowercase
        text = str(text).lower()
        
        # Remove special characters but keep: letters, numbers, spaces, commas, periods, hyphens
        text = re.sub(r'[^a-z0-9\s,.\-_]', ' ', text)
        
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def split_by_row(self, df: pd.DataFrame, text_columns: List[str]) -> List[Dict]:
        """
        Split data by row - each row becomes one or more chunks.
        
        Args:
            df (pd.DataFrame): Input dataframe
            text_columns (List[str]): Columns to combine for chunking
            
        Returns:
            List[Dict]: List of chunk dictionaries
        """
        chunks = []
        
        for idx, row in df.iterrows():
            # Combine text from specified columns
            combined_text = ""
            metadata = {}
            
            for col in df.columns:
                value = row[col]
                if pd.notnull(value):
                    if col in text_columns:
                        combined_text += f"{col}: {value}\n"
                    metadata[col] = str(value)
            
            # Normalize the combined text
            normalized_text = self.normalize_text(combined_text)
            
            if normalized_text:
                # Split into chunks if text is too long
                text_chunks = self._split_long_text(normalized_text)
                
                for chunk_idx, chunk_text in enumerate(text_chunks):
                    chunk = {
                        'chunk_id': f"dataglossary_row_{idx}_chunk_{chunk_idx}",
                        'row_index': int(idx),
                        'chunk_index': chunk_idx,
                        'text': chunk_text,
                        'metadata': metadata,
                        'chunk_length': len(chunk_text),
                        'token_count': self._estimate_tokens(chunk_text)
                    }
                    chunks.append(chunk)
        
        return chunks
    
    def split_by_paragraph(self, text: str, metadata: Dict = None) -> List[Dict]:
        """
        Split text by paragraphs (newlines).
        
        Args:
            text (str): Input text
            metadata (Dict): Optional metadata to attach to chunks
            
        Returns:
            List[Dict]: List of chunk dictionaries
        """
        chunks = []
        paragraphs = text.split('\n')
        
        for idx, paragraph in enumerate(paragraphs):
            normalized_para = self.normalize_text(paragraph)
            
            if normalized_para and len(normalized_para) > 10:  # Skip very short paragraphs
                para_chunks = self._split_long_text(normalized_para)
                
                for chunk_idx, chunk_text in enumerate(para_chunks):
                    chunk = {
                        'chunk_id': f"para_{idx}_chunk_{chunk_idx}",
                        'paragraph_index': idx,
                        'chunk_index': chunk_idx,
                        'text': chunk_text,
                        'metadata': metadata or {},
                        'chunk_length': len(chunk_text),
                        'token_count': self._estimate_tokens(chunk_text)
                    }
                    chunks.append(chunk)
        
        return chunks
    
    def _split_long_text(self, text: str) -> List[str]:
        """
        Split long text into chunks with overlap.
        
        Args:
            text (str): Input text
            
        Returns:
            List[str]: List of text chunks
        """
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            # Get chunk
            end = start + self.chunk_size
            chunk = text[start:end]
            
            # Try to break at sentence boundary if possible
            if end < len(text):
                # Look for sentence endings within last 100 chars
                last_period = chunk.rfind('.')
                last_comma = chunk.rfind(',')
                
                if last_period > len(chunk) - 100:
                    end = start + last_period + 1
                    chunk = text[start:end]
                elif last_comma > len(chunk) - 100:
                    end = start + last_comma + 1
                    chunk = text[start:end]
            
            chunks.append(chunk.strip())
            
            # Move start position with overlap
            start = end - self.chunk_overlap
            
            # Prevent infinite loop
            if start >= len(text):
                break
        
        return chunks
    
    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate token count (rough approximation: ~4 chars per token).
        
        Args:
            text (str): Input text
            
        Returns:
            int: Estimated token count
        """
        return len(text) // 4
    
    def calculate_statistics(self, chunks: List[Dict]) -> Dict:
        """
        Calculate statistics for the chunks.
        
        Args:
            chunks (List[Dict]): List of chunks
            
        Returns:
            Dict: Statistics dictionary
        """
        if not chunks:
            return self.statistics
        
        chunk_lengths = [chunk['chunk_length'] for chunk in chunks]
        token_counts = [chunk['token_count'] for chunk in chunks]
        
        self.statistics = {
            'total_chunks': len(chunks),
            'average_chunk_length': np.mean(chunk_lengths),
            'average_token_count': np.mean(token_counts),
            'min_chunk_length': min(chunk_lengths),
            'max_chunk_length': max(chunk_lengths),
            'total_characters': sum(chunk_lengths),
            'total_estimated_tokens': sum(token_counts),
            'chunk_size_setting': self.chunk_size,
            'chunk_overlap_setting': self.chunk_overlap
        }
        
        return self.statistics
    
    def save_chunks(self, chunks: List[Dict], output_path: str) -> None:
        """
        Save chunks to JSON file.
        
        Args:
            chunks (List[Dict]): List of chunks
            output_path (str): Path to save JSON file
        """
        output_data = {
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'chunk_size': self.chunk_size,
                'chunk_overlap': self.chunk_overlap,
                'total_chunks': len(chunks),
                'source': 'DataGlossary.xlsx'
            },
            'statistics': self.statistics,
            'chunks': chunks
        }
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Chunks saved to: {output_path}")
    
    def save_chunks_csv(self, chunks: List[Dict], output_path: str) -> None:
        """
        Save chunks to CSV file (flattened version).
        
        Args:
            chunks (List[Dict]): List of chunks
            output_path (str): Path to save CSV file
        """
        # Flatten chunks for CSV
        flattened_chunks = []
        for chunk in chunks:
            flat_chunk = {
                'chunk_id': chunk['chunk_id'],
                'text': chunk['text'],
                'chunk_length': chunk['chunk_length'],
                'token_count': chunk['token_count']
            }
            # Add metadata fields
            if 'metadata' in chunk:
                for key, value in chunk['metadata'].items():
                    flat_chunk[f'meta_{key}'] = value
            
            flattened_chunks.append(flat_chunk)
        
        df = pd.DataFrame(flattened_chunks)
        df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"✅ Chunks CSV saved to: {output_path}\")")
    
    def print_statistics(self) -> None:
        """Print chunking statistics in a formatted way."""
        print("\n" + "="*60)
        print("📊 DATAGLOSSARY CHUNKING STATISTICS")
        print("="*60)
        for key, value in self.statistics.items():
            if isinstance(value, float):
                print(f"{key:30s}: {value:.2f}")
            else:
                print(f"{key:30s}: {value}")
        print("="*60 + "\n")


def main():
    """
    Main function to execute the DataGlossary chunking pipeline.
    """
    # Configuration
    CHUNK_SIZE = 512  # Maximum characters per chunk
    CHUNK_OVERLAP = 50  # Characters overlap between chunks
    
    # Get project paths
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = os.path.join(project_root, 'data')
    
    # Input and output paths
    input_file = os.path.join(data_dir, 'DataGlossary_clean.xlsx')
    output_dir = os.path.join(data_dir, 'DataGlossary_Chunks')
    os.makedirs(output_dir, exist_ok=True)
    
    output_json = os.path.join(output_dir, 'DataGlossary_chunks.json')
    output_csv = os.path.join(output_dir, 'DataGlossary_chunks.csv')
    
    print("="*70)
    print("🚀 DataGlossary Chunking & Preprocessing Pipeline")
    print("="*70)
    print(f"📁 Input file: {input_file}")
    print(f"⚙️  Chunk size: {CHUNK_SIZE}")
    print(f"⚙️  Chunk overlap: {CHUNK_OVERLAP}\n")
    
    # Load data
    print("📖 Loading DataGlossary data...")
    df = pd.read_excel(input_file)
    print(f"✅ Loaded {len(df)} rows with {len(df.columns)} columns")
    
    # Initialize chunker
    chunker = DataGlossaryChunker(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    
    # Define which columns to include in chunks
    text_columns = [
        'db_type', 'database_server_name', 'source_system_database_name',
        'schema_name', 'source_table', 'pk_column_name', 'pk_data_type',
        'watermark_column_name', 'watermark_data_type', 'is_watermark_null',
        'type_of_table', 'extraction_mode', 'extraction_type_rules'
    ]
    
    # Filter to only existing columns
    text_columns = [col for col in text_columns if col in df.columns]
    
    print(f"📝 Processing {len(text_columns)} columns for chunking...")
    
    # Split by row
    print("✂️  Splitting data by row...")
    chunks = chunker.split_by_row(df, text_columns)
    
    # Calculate statistics
    print("📊 Calculating statistics...")
    statistics = chunker.calculate_statistics(chunks)
    
    # Print statistics
    chunker.print_statistics()
    
    # Save results
    print("💾 Saving chunks...")
    chunker.save_chunks(chunks, output_json)
    chunker.save_chunks_csv(chunks, output_csv)
    
    # Print sample chunks
    print("\n" + "="*60)
    print("📄 SAMPLE CHUNKS (First 3)")
    print("="*60)
    for i, chunk in enumerate(chunks[:3]):
        print(f"\n--- Chunk {i+1} ---")
        print(f"ID: {chunk['chunk_id']}")
        print(f"Length: {chunk['chunk_length']} chars")
        print(f"Tokens: {chunk['token_count']}")
        print(f"Text preview: {chunk['text'][:200]}...")
    
    print("\n" + "="*70)
    print("✨ DataGlossary Chunking & Preprocessing completed successfully!")
    print("="*70)
    print(f"📄 Output JSON: {output_json}")
    print(f"📄 Output CSV: {output_csv}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
