"""
Flow:
1. Cleaning data via ingestion_module.py
2. Chunking dengan overlap
3. Save ke data/User_Upload_Chunks/
"""

import pandas as pd
import numpy as np
import json
import re
from pathlib import Path
from datetime import datetime
from ingestion_module import DataIngestionModule


def normalize_text(text):
    """Normalize text: lowercase, remove special chars, clean whitespace."""
    if pd.isnull(text) or text is None:
        return ""
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s,.\-_]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def split_long_text(text, chunk_size, chunk_overlap):
    """Split text into chunks with overlap."""
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        
        if end < len(text):
            last_period = chunk.rfind('.')
            last_comma = chunk.rfind(',')
            
            if last_period > len(chunk) - 100:
                end = start + last_period + 1
                chunk = text[start:end]
            elif last_comma > len(chunk) - 100:
                end = start + last_comma + 1
                chunk = text[start:end]
        
        chunks.append(chunk.strip())
        start = end - chunk_overlap
        
        if start >= len(text):
            break
    
    return chunks


def chunk_text_data(text, file_name, chunk_size, chunk_overlap):
    """Chunk text from PDF/TXT."""
    chunks = []
    paragraphs = text.split('\n\n')
    chunk_idx = 0
    
    for para_idx, para in enumerate(paragraphs):
        if len(para.strip()) < 20:
            continue
        
        normalized = normalize_text(para)
        if normalized:
            para_chunks = split_long_text(normalized, chunk_size, chunk_overlap)
            
            for local_idx, chunk_text in enumerate(para_chunks):
                chunks.append({
                    'chunk_id': f"{file_name}_para_{para_idx}_chunk_{local_idx}",
                    'source_file': file_name,
                    'source_type': 'text',
                    'paragraph_index': para_idx,
                    'chunk_index': chunk_idx,
                    'text': chunk_text,
                    'chunk_length': len(chunk_text),
                    'token_count': len(chunk_text) // 4
                })
                chunk_idx += 1
    
    return chunks


def chunk_dataframe_data(df, file_name, chunk_size, chunk_overlap):
    """Chunk DataFrame from CSV/XLSX."""
    chunks = []
    
    for idx, row in df.iterrows():
        row_text = ""
        for col in df.columns:
            if pd.notnull(row[col]):
                row_text += f"{col}: {row[col]}\n"
        
        normalized = normalize_text(row_text)
        if normalized:
            text_chunks = split_long_text(normalized, chunk_size, chunk_overlap)
            
            for chunk_idx, chunk_text in enumerate(text_chunks):
                chunks.append({
                    'chunk_id': f"{file_name}_row_{idx}_chunk_{chunk_idx}",
                    'source_file': file_name,
                    'source_type': 'dataframe',
                    'row_index': int(idx),
                    'chunk_index': chunk_idx,
                    'text': chunk_text,
                    'chunk_length': len(chunk_text),
                    'token_count': len(chunk_text) // 4
                })
    
    return chunks


def process_user_file(file_name, chunk_size=512, chunk_overlap=50):
    """
    Main function to process user file: cleaning + chunking.
    
    Args:
        file_name (str): Nama file di folder data/ (e.g., "SOP_Hotel.pdf")
        chunk_size (int): Max characters per chunk (default: 512)
        chunk_overlap (int): Overlap characters (default: 50)
    
    Returns:
        dict: {
            'status': 'success' or 'error',
            'message': str,
            'output_path': str,
            'statistics': dict,
            'chunks': list
        }
    """
    try:
        print("\n" + "="*70)
        print("🚀 USER UPLOAD: CLEANING + CHUNKING")
        print("="*70 + "\n")
        
        # Get file path
        project_root = Path(__file__).parent.parent.parent
        file_path = project_root / 'data' / file_name
        
        if not file_path.exists():
            return {
                'status': 'error',
                'message': f"File not found: {file_path}",
                'output_path': None,
                'statistics': {},
                'chunks': []
            }
        
        print(f"📂 File: {file_name}")
        print(f"⚙️  Chunk Size: {chunk_size} chars")
        print(f"⚙️  Overlap: {chunk_overlap} chars\n")
        
        # STEP 1: CLEANING
        print("="*70)
        print("STEP 1: CLEANING DATA")
        print("="*70 + "\n")
        
        ingestion = DataIngestionModule()
        result = ingestion.ingest_file(str(file_path), clean_data=True)
        
        if result['status'] != 'success':
            return {
                'status': 'error',
                'message': f"Cleaning failed: {result['message']}",
                'output_path': None,
                'statistics': {},
                'chunks': []
            }
        
        print(f"✅ Cleaned successfully!")
        print(f"   Type: {result['metadata']['file_type']}\n")
        
        # STEP 2: CHUNKING
        print("="*70)
        print("STEP 2: CHUNKING DATA")
        print("="*70 + "\n")
        
        data = result['data']
        file_stem = file_path.stem
        
        if isinstance(data, pd.DataFrame):
            print(f"   DataFrame: {len(data)} rows x {len(data.columns)} columns")
            chunks = chunk_dataframe_data(data, file_stem, chunk_size, chunk_overlap)
        elif isinstance(data, str):
            print(f"   Text: {len(data)} characters")
            chunks = chunk_text_data(data, file_stem, chunk_size, chunk_overlap)
        else:
            return {
                'status': 'error',
                'message': f"Unsupported data type: {type(data)}",
                'output_path': None,
                'statistics': {},
                'chunks': []
            }
        
        print(f"✅ Created {len(chunks)} chunks\n")
        
        # STEP 3: STATISTICS
        print("="*70)
        print("STEP 3: STATISTICS")
        print("="*70 + "\n")
        
        lengths = [c['chunk_length'] for c in chunks]
        tokens = [c['token_count'] for c in chunks]
        
        stats = {
            'total_chunks': len(chunks),
            'average_chunk_length': np.mean(lengths),
            'average_token_count': np.mean(tokens),
            'min_chunk_length': min(lengths),
            'max_chunk_length': max(lengths),
            'total_characters': sum(lengths),
            'total_estimated_tokens': sum(tokens),
            'chunk_size_setting': chunk_size,
            'chunk_overlap_setting': chunk_overlap
        }
        
        print(f"   Total Chunks: {stats['total_chunks']}")
        print(f"   Avg Length: {stats['average_chunk_length']:.1f} chars")
        print(f"   Avg Tokens: {stats['average_token_count']:.1f}")
        print(f"   Range: {stats['min_chunk_length']} - {stats['max_chunk_length']} chars\n")
        
        # STEP 4: SAVE
        print("="*70)
        print("STEP 4: SAVING")
        print("="*70 + "\n")
        
        output_dir = project_root / 'data' / 'User_Upload_Chunks'
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{file_stem}_chunks.json"
        
        output_data = {
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'chunk_size': chunk_size,
                'chunk_overlap': chunk_overlap,
                'source_file': file_name,
                'source_file_type': result['metadata']['file_type']
            },
            'statistics': stats,
            'chunks': chunks
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Saved to: {output_file}\n")
        
        # FINAL
        print("="*70)
        print("✅ SUCCESS!")
        print("="*70)
        print(f"📄 Output: {output_file}")
        print(f"📊 Chunks: {stats['total_chunks']}")
        print(f"📏 Avg Length: {stats['average_chunk_length']:.1f} chars")
        print(f"🔢 Avg Tokens: {stats['average_token_count']:.1f}")
        print("="*70 + "\n")
        
        return {
            'status': 'success',
            'message': 'Processing completed successfully',
            'output_path': str(output_file),
            'statistics': stats,
            'chunks': chunks
        }
    
    except Exception as e:
        return {
            'status': 'error',
            'message': f"Processing failed: {str(e)}",
            'output_path': None,
            'statistics': {},
            'chunks': []
        }
