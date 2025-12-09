# 📦 Guide: Handling Large Files

## Overview
This guide explains how to process large files without hitting limits in the RAG Chatbot system.

## 🎯 Quick Configuration

### For Small Files (< 10 MB)
```env
MAX_FILE_SIZE_MB=50
PROCESSING_BATCH_SIZE=100
EMBEDDING_BATCH_SIZE=100
CHUNK_SIZE=512
```

### For Medium Files (10-50 MB)
```env
MAX_FILE_SIZE_MB=100
PROCESSING_BATCH_SIZE=50
EMBEDDING_BATCH_SIZE=100
CHUNK_SIZE=512
```

### For Large Files (50-200 MB)
```env
MAX_FILE_SIZE_MB=200
PROCESSING_BATCH_SIZE=30
EMBEDDING_BATCH_SIZE=50
CHUNK_SIZE=512
```

### For Very Large Files (200-500 MB)
```env
MAX_FILE_SIZE_MB=500
PROCESSING_BATCH_SIZE=20
EMBEDDING_BATCH_SIZE=30
CHUNK_SIZE=1024
```

### For Extremely Large Files (> 500 MB)
**Recommendation:** Split the file into smaller parts or use CHUNK_SIZE=2048

```env
MAX_FILE_SIZE_MB=1000
PROCESSING_BATCH_SIZE=10
EMBEDDING_BATCH_SIZE=20
CHUNK_SIZE=2048
```

## 🔧 Configuration Parameters Explained

### 1. `MAX_FILE_SIZE_MB`
- **Purpose:** Maximum allowed file size in megabytes
- **Default:** 50 MB
- **Effect:** Files larger than this will be rejected
- **Recommendation:** Set to your largest expected file size

### 2. `PROCESSING_BATCH_SIZE`
- **Purpose:** Number of chunks processed at once
- **Default:** 50
- **Effect:** 
  - **Higher (100+):** Faster but uses more memory
  - **Lower (10-30):** Slower but uses less memory
- **Recommendation:** 
  - Normal files: 50-100
  - Large files: 20-30
  - Very large files: 10-20

### 3. `EMBEDDING_BATCH_SIZE`
- **Purpose:** Number of embeddings generated per API call
- **Default:** 100
- **Effect:**
  - **Higher:** Faster but may hit API rate limits
  - **Lower:** Slower but safer
- **Recommendation:**
  - HuggingFace (local): 100-200 (no limits)
  - OpenAI API: 50-100 (rate limited)
  - Groq API: 50-100 (rate limited)

### 4. `CHUNK_SIZE`
- **Purpose:** Characters per chunk
- **Default:** 512
- **Effect:**
  - **Smaller (256-512):** More chunks, better precision, more storage
  - **Larger (1024-2048):** Fewer chunks, less precision, less storage
- **Recommendation:**
  - For Q&A: 512 (default)
  - For summarization: 1024
  - For very large files: 1024-2048

## 💡 Strategies for Handling Large Files

### Strategy 1: Optimize Batch Sizes (Recommended)
**Best for:** Files 50-200 MB
```env
# Reduce batch sizes to use less memory
PROCESSING_BATCH_SIZE=30
EMBEDDING_BATCH_SIZE=50
```

### Strategy 2: Increase Chunk Size
**Best for:** Files > 200 MB
```env
# Larger chunks = fewer total chunks
CHUNK_SIZE=1024
CHUNK_OVERLAP=100
```

### Strategy 3: Use Local Embeddings (FREE)
**Best for:** Avoiding API limits
```env
# HuggingFace runs locally, no rate limits
EMBEDDING_PROVIDER=huggingface
HUGGINGFACE_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DIMENSION=384
```

### Strategy 4: Split Files Manually
**Best for:** Files > 500 MB

Split your file into parts:
```bash
# For PDFs: Use PDF tools to split
# For text files:
split -l 10000 large_file.txt part_

# Then upload each part separately
```

### Strategy 5: Progressive Upload
**Best for:** Multiple large files

Upload files one by one instead of all at once:
1. Upload file 1 → Process → Add to Knowledge Base
2. Upload file 2 → Process → Add to Knowledge Base
3. Continue...

## 📊 Memory Usage Estimation

### Formula
```
Memory (MB) ≈ PROCESSING_BATCH_SIZE × CHUNK_SIZE × 0.01
```

### Examples
| Batch Size | Chunk Size | Est. Memory |
|------------|------------|-------------|
| 100        | 512        | ~512 MB     |
| 50         | 512        | ~256 MB     |
| 30         | 512        | ~154 MB     |
| 20         | 1024       | ~205 MB     |
| 10         | 2048       | ~205 MB     |

## 🚀 Performance Tips

### 1. Use HuggingFace Embeddings (Fastest for Large Files)
```env
EMBEDDING_PROVIDER=huggingface
HUGGINGFACE_MODEL=sentence-transformers/all-MiniLM-L6-v2  # Fastest
EMBEDDING_DIMENSION=384
```

### 2. Optimize Qdrant Mode
```env
# Server mode is faster than local for large datasets
QDRANT_MODE=server
QDRANT_URL=http://localhost:6333
```

### 3. Adjust Retrieval Settings
```env
# Fewer results = faster queries
RETRIEVAL_TOP_K=3

# Higher threshold = fewer candidates
MIN_SIMILARITY_SCORE=0.6
```

## 🐛 Troubleshooting

### Problem: "Out of Memory" Error
**Solutions:**
1. Decrease `PROCESSING_BATCH_SIZE` to 20 or lower
2. Decrease `EMBEDDING_BATCH_SIZE` to 30 or lower
3. Increase `CHUNK_SIZE` to reduce total chunks
4. Split file into smaller parts

### Problem: "File Too Large" Error
**Solutions:**
1. Increase `MAX_FILE_SIZE_MB` in `.env`
2. Split file into multiple smaller files
3. Compress file if possible

### Problem: Processing is Very Slow
**Solutions:**
1. Use HuggingFace local embeddings (faster than API calls)
2. Increase `PROCESSING_BATCH_SIZE` if you have enough memory
3. Use smaller embedding model (all-MiniLM-L6-v2)
4. Switch to `QDRANT_MODE=server` with Docker

### Problem: API Rate Limit Errors
**Solutions:**
1. Decrease `EMBEDDING_BATCH_SIZE` to 20-30
2. Switch to HuggingFace (no rate limits)
3. Add delays between batches (modify `index_builder.py`)

## 📈 Recommended Configurations by Use Case

### Use Case: Corporate Documents (PDF, DOCX)
**File size:** 10-50 MB, High quality needed
```env
MAX_FILE_SIZE_MB=100
PROCESSING_BATCH_SIZE=50
CHUNK_SIZE=512
CHUNK_OVERLAP=50
EMBEDDING_PROVIDER=huggingface
HUGGINGFACE_MODEL=sentence-transformers/all-mpnet-base-v2
```

### Use Case: Large Dataset (CSV, XLSX)
**File size:** 50-200 MB, Many rows
```env
MAX_FILE_SIZE_MB=300
PROCESSING_BATCH_SIZE=30
CHUNK_SIZE=1024
CHUNK_OVERLAP=100
EMBEDDING_PROVIDER=huggingface
HUGGINGFACE_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

### Use Case: Books/Manuals (TXT, PDF)
**File size:** 10-100 MB, Long text
```env
MAX_FILE_SIZE_MB=150
PROCESSING_BATCH_SIZE=40
CHUNK_SIZE=768
CHUNK_OVERLAP=75
EMBEDDING_PROVIDER=huggingface
HUGGINGFACE_MODEL=sentence-transformers/multi-qa-mpnet-base-dot-v1
```

### Use Case: Code Documentation
**File size:** 5-50 MB, Structured text
```env
MAX_FILE_SIZE_MB=100
PROCESSING_BATCH_SIZE=60
CHUNK_SIZE=512
CHUNK_OVERLAP=50
EMBEDDING_PROVIDER=huggingface
HUGGINGFACE_MODEL=sentence-transformers/all-mpnet-base-v2
```

## 🔍 Monitoring File Processing

Check the console output during upload for:
- File sizes
- Number of chunks generated
- Processing speed
- Memory usage warnings

Example output:
```
======================================================================
📦 PROCESSING 2 FILE(S)
======================================================================
📄 large_document.pdf: 85.34 MB
📄 reference.docx: 12.56 MB
📊 Total size: 97.90 MB
⚠️  Large files detected - using batch processing

[1/2] Processing: large_document.pdf
   📝 Content length: 1,245,678 characters
   ✅ Generated 2,432 chunks
[2/2] Processing: reference.docx
   📝 Content length: 145,234 characters
   ✅ Generated 284 chunks

======================================================================
📊 TOTAL CHUNKS: 2,716
======================================================================
```

## 💰 Cost Considerations

### FREE Options (No Limits)
- **Embedding:** HuggingFace (local)
- **LLM:** Groq API (free tier)
- **Vector DB:** Qdrant (local mode)

### Paid Options (With Limits)
- **Embedding:** OpenAI ($0.00002/1K tokens)
- **LLM:** OpenAI GPT-4 ($0.03/1K tokens)
- **Vector DB:** Qdrant Cloud (paid)

**Recommendation:** Use FREE options for large files to avoid costs!

## 📚 Further Resources

- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [HuggingFace Models](https://huggingface.co/sentence-transformers)
- [Groq API Docs](https://console.groq.com/docs)
- [PDF Processing Tips](https://pypdf2.readthedocs.io/)

---

**Need Help?** Check the console output for detailed error messages and processing stats!
