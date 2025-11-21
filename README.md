# 🤖 RAG Chatbot MVP

A lightweight web-based Retrieval-Augmented Generation (RAG) system that allows users to upload documents (PDF, Excel, CSV, TXT) and instantly chat with an AI chatbot powered by the uploaded knowledge.

---

## 🧠 Overview

This project demonstrates how to build a customizable RAG pipeline using **Streamlit**, **LlamaIndex**, and **Qdrant**.  
Each module is organized for easy development, testing, and deployment.

Users can:
- Upload their own knowledge files (finance data, SOP documents, etc.)
- Automatically chunk and embed the text
- Store embeddings in a vector database
- Chat with an AI model that retrieves context-aware answers from the data

---
## ⚙️ Tech Stack

This project is built using a modern **AI + Web stack** to create a flexible **Retrieval-Augmented Generation (RAG)** system and deploy a chatbot accessible through the web.

---

### 🧠 Core AI Frameworks

- **LangChain** — Framework to build and manage LLM-based applications (prompt templates, chains, agents, and retrieval).  
- **LlamaIndex** — Simplifies RAG implementation by connecting data sources to the language model.  
- **Groq API** — Provides ultra-fast LLM inference compatible with OpenAI format (e.g., Mixtral, Llama-3 models).  
- **Tiktoken** — Handles token counting for text chunking and embedding efficiency.

---

### 💾 Vector Database

- **Qdrant** — Open-source and cloud-ready vector database for storing and retrieving document embeddings efficiently.

---

### 🌐 Web Frameworks

- **FastAPI** — Backend API framework used to serve the RAG pipeline and handle chat requests.  
- **Streamlit** — Lightweight front-end framework for building an interactive chatbot interface.

---

### 🧩 Utilities & Data Processing

- **Pandas** — Handles structured data input such as Excel or CSV files.  
- **PyPDF** — Extracts and processes text from PDF documents.  
- **Requests** — Handles API calls and integrations between components.  
- **Pydantic** — Ensures clean and validated data models for FastAPI.

---

### 🔐 Configuration & Environment

- **Python-dotenv** — Manages API keys and environment variables through `.env` files.  
- **Uvicorn** — ASGI server to run FastAPI in development or production environments.  

---

### 🧱 Development & Version Control

- **Virtual Environment (`myvenv/`)** — Keeps dependencies isolated per project.  
- **Git & GitHub** — Used for version control, code collaboration, and project tracking.

---

### 🧩 System Overview

This project enables users to:
1. Upload and index knowledge sources (e.g., PDF or Excel).  
2. Configure chunking parameters and embedding storage in Qdrant.  
3. Interact with an LLM-powered chatbot connected to the indexed data via RAG.


---
## 📁 Folder & File Descriptions

### **Root Directory**

| File | Description |
|------|--------------|
| **.env** | Contains environment variables such as API keys and database URLs. Keep this file private — never commit it to Git. |
| **.gitignore** | Lists files and directories that Git should ignore (e.g., `.env`, cache folders). |
| **exploration.ipynb** | Used for quick testing or experimentation (data ingestion, embeddings, etc.) before integrating into the main code. |
| **README.md** | Main documentation file for setup, usage, and structure. |
| **requirements.txt** | Lists all Python dependencies required to run the project. |

---

### **data/**
Stores all uploaded or sample datasets for RAG testing (e.g., Excel, PDF, CSV files).

### **src/**
Main source code directory containing modular components of the RAG pipeline.

#### **config/**
- **`settings.py`**  
  Loads configuration from `.env` (API keys, embedding model, vector DB URL).  
  Centralized configuration ensures consistent access across modules.

---

#### **ingestion/**
Handles reading and preprocessing of documents.

- **`load_files.py`** — Reads multiple file formats (PDF, Excel, CSV, TXT) and converts them into plain text.  
- **`chunk_text.py`** — Splits text into smaller overlapping chunks for vectorization and retrieval.

---

#### **vectorstore/**
Manages embedding generation and storage in the vector database.

- **`qdrant_client.py`** — Connects to the Qdrant instance (local or cloud) and performs CRUD operations.  
- **`index_builder.py`** — Converts text chunks into embeddings and stores them in Qdrant for retrieval.

---

#### **rag/**
Implements the RAG pipeline.

- **`retriever.py`** — Retrieves the most relevant text chunks from the vector database using semantic similarity.  
- **`query_engine.py`** — Combines retrieved context with user queries and sends them to the LLM for final response generation.

---

#### **ui/**
Streamlit interface for user interaction.

- **`app.py`** — Main entry point for the web app.  
  Provides file upload, indexing, and chat interface.


---

## ⚙️ How It Works

1. **Upload Knowledge** — User uploads one or more files (PDF, Excel, etc.) via Streamlit UI.  
2. **Text Extraction & Chunking** — The system converts the files to text and splits them into small chunks.  
3. **Embedding & Storage** — Each chunk is embedded and stored in the Qdrant vector database.  
4. **User Query** — User enters a question in the chatbot interface.  
5. **Context Retrieval** — The system retrieves top relevant chunks from Qdrant.  
6. **Answer Generation** — The retrieved context is combined with the question and passed to the LLM to generate a response.

---

## 🚀 Quick Start

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/yourusername/rag-chatbot-mvp.git
cd rag-chatbot-mvp
```

### 2️⃣ Create a Virtual Environment

```bash
python -m venv myvenv
source myvenv/bin/activate     # macOS/Linux
myvenv\Scripts\activate        # Windows
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Configure Environment
```bash
# Copy template
cp .env.example .env

# Edit .env and add your Groq API key (FREE at https://console.groq.com)
GROQ_API_KEY=gsk_your_key_here
```

### Step 5: Start Qdrant Web UI (Optional)
```bash
# Windows
./start_qdrant_server.bat

# Linux/Mac
bash start_qdrant_server.sh

# Then update .env:
QDRANT_MODE=server
QDRANT_URL=http://localhost:6333
```

# Inspect database
```bash
python src/vectorstore/inspect_qdrant.py
```

### Step 6: Index Data Glossary
```bash
python src/ingestion/index_dataglossary.py
```


### Step 7: Index User Uploaded Files
```bash
python src/ingestion/index_user_upload.py
```

### Step 8: Test Everything
```bash
python test_embedding_retrieval.py
```

**Expected:** All 6 tests pass ✅


### Step 9 Run the App

```bash
streamlit run src/ui/app.py
```

---

## 💰 Our Setup

| Component | Provider | Cost |
|-----------|----------|------|
| **Embeddings** | HuggingFace `all-mpnet-base-v2` | $0 |
| **Vector DB** | Qdrant (local/server) | $0 |
| **LLM** | Groq `llama-3.3-70b-versatile` | $0  |
| **Total** | | **$0/month** 🎉 |