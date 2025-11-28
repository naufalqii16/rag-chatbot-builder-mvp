"""
RAG Chatbot Streamlit App - Redesigned
---------------------------------------
Two-page structure: Home and Chat
"""


import streamlit as st
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from config.settings import settings
from rag.retriever import Retriever
from rag.query_engine import QueryEngine
from utils.style_loader import inject_custom_css

# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(
    page_title="🧠 RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ----------------------------
# Load Custom CSS
# ----------------------------
inject_custom_css()

# ----------------------------
# Initialize session state
# ----------------------------
if "current_page" not in st.session_state:
    st.session_state["current_page"] = "home"
if "chat_state" not in st.session_state:
    st.session_state["chat_state"] = "upload"  # "upload" or "chat"
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "query_engine" not in st.session_state:
    st.session_state["query_engine"] = None
if "uploaded_files_list" not in st.session_state:
    st.session_state["uploaded_files_list"] = []

# ----------------------------
# Initialize RAG Engine
# ----------------------------
@st.cache_resource
def initialize_query_engine():
    """Initialize RAG engine for User Upload data."""
    try:
        engine = QueryEngine(collection_name=settings.QDRANT_USER_UPLOAD_COLLECTION)
        return engine, True, "✅ RAG engine initialized!"
    except Exception as e:
        return None, False, f"❌ Error initializing RAG: {str(e)}"

# ----------------------------
# Helper function to reset chatbot
# ----------------------------
def reset_chatbot():
    """Reset chatbot to initial state"""
    st.session_state["current_page"] = "home"
    st.session_state["chat_state"] = "upload"
    st.session_state["chat_history"] = []
    st.session_state["uploaded_files_list"] = []
    # Don't reset query_engine (keep it cached)

# ============================
# HOME PAGE
# ============================
if st.session_state["current_page"] == "home":
    
    # Header with Start Chat button
    col1, col2 = st.columns([6, 1])
    with col1:
        st.title("🧠 RAG Chatbot")
    with col2:
        if st.button("💬 Start Chat", key="start_chat_top", use_container_width=True, type="primary"):
            st.session_state["current_page"] = "chat"
            st.rerun()
    
    st.markdown('<div class="div-hr"></div>', unsafe_allow_html=True)
    
    # About Section
    st.markdown("""
    <div class="info-box">
        <h2>📖 About This Chatbot</h2>
        <p>Selamat datang di <strong>RAG-powered Chatbot</strong>! Sistem ini memungkinkan Anda untuk:</p>
        <ul>
            <li>📤 <strong>Upload dokumen</strong> dalam berbagai format (PDF, DOCX, TXT, CSV, XLSX)</li>
            <li>🤖 <strong>Bertanya</strong> tentang isi dokumen Anda</li>
            <li>⚡ <strong>Mendapatkan jawaban</strong> yang didukung oleh AI dan semantic search</li>
            <li>🎯 <strong>Akurat</strong> dengan referensi dari dokumen asli</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("&nbsp;")
    
    # How to Use Section
    st.markdown("""
    <div class="info-box">
        <h2>🚀 How to Use</h2>
        <ol>
            <li><strong>Klik "Start Chat"</strong> untuk memulai</li>
            <li><strong>Upload dokumen</strong> Anda (bisa lebih dari satu file)</li>
            <li><strong>Klik "Process & Index"</strong> untuk memproses dokumen</li>
            <li><strong>Mulai bertanya</strong> tentang isi dokumen</li>
            <li><strong>Upload dokumen tambahan</strong> kapan saja untuk memperluas knowledge base</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("&nbsp;")
    
    # System Info Section
    st.markdown(f"""
    <div class="info-box">
        <h2>⚙️ System Information</h2>
        <table style="width: 100%; color: #ffffff;">
            <tr>
                <td><strong>🤖 LLM Provider:</strong></td>
                <td>{settings.LLM_PROVIDER.upper()} ({settings.GROQ_LLM_MODEL if settings.LLM_PROVIDER == 'groq' else settings.OPENAI_LLM_MODEL})</td>
            </tr>
            <tr>
                <td><strong>🔤 Embedding Model:</strong></td>
                <td>{settings.HUGGINGFACE_MODEL if settings.EMBEDDING_PROVIDER == 'huggingface' else settings.OPENAI_EMBEDDING_MODEL}</td>
            </tr>
            <tr>
                <td><strong>💾 Vector Database:</strong></td>
                <td>Qdrant ({settings.QDRANT_MODE} mode)</td>
            </tr>
            <tr>
                <td><strong>📂 Collection:</strong></td>
                <td>{settings.QDRANT_COLLECTION_NAME}</td>
            </tr>
            <tr>
                <td><strong>🔎 Top-K Results:</strong></td>
                <td>{settings.RETRIEVAL_TOP_K}</td>
            </tr>
            <tr>
                <td><strong>📊 Min Similarity Score:</strong></td>
                <td>{settings.MIN_SIMILARITY_SCORE}</td>
            </tr>
        </table>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("&nbsp;")
    st.markdown("&nbsp;")
    
    # Big Start Chat Button
    col1, col2, col3 = st.columns([2, 3, 2])
    with col2:
        if st.button("🚀 Start Chat Now", key="start_chat_bottom", use_container_width=True, type="primary"):
            st.session_state["current_page"] = "chat"
            st.rerun()

# ============================
# CHAT PAGE
# ============================
elif st.session_state["current_page"] == "chat":
    
    # Header with Home button
    col1, col2 = st.columns([1, 11])
    with col1:
        if st.button("🏠", key="home_button",  use_container_width=True
                     ):
            reset_chatbot()
            st.rerun()
  
    with col2:
        st.title("💬 Chat with Your Documents")
    
    st.markdown('<div class="div-hr"></div>', unsafe_allow_html=True)
    
    # ============================
    # STATE 1: UPLOAD DOCUMENTS
    # ============================
    if st.session_state["chat_state"] == "upload":
        
        st.markdown("""
        <div class="upload-container">
            <h2 style="text-align: center; margin-bottom: 1rem;">📤 Upload Your Documents</h2>
            <p style="text-align: center; color: rgba(255, 255, 255, 0.7); margin-bottom: 2rem;">
                Upload dokumen untuk memulai. Anda bisa upload lebih dari satu file sekaligus.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # File uploader
        uploaded_files = st.file_uploader(
            "Pilih file",
            accept_multiple_files=True,
            type=["pdf", "csv", "txt", "xlsx", "xls", "docx", "doc"],
            help="Format yang didukung: PDF, CSV, TXT, XLSX, DOCX",
            label_visibility="collapsed"
        )
        
        # Display uploaded files
        if uploaded_files:
            st.markdown("---")
            st.subheader(f"📋 Files Selected ({len(uploaded_files)})")
            
            for idx, uploaded_file in enumerate(uploaded_files, 1):
                file_size = uploaded_file.size / 1024  # KB
                st.markdown(f"**{idx}.** `{uploaded_file.name}` ({file_size:.2f} KB)")
            
            st.markdown("&nbsp;")
            
            # Process button
            col1, col2, col3 = st.columns([2, 3, 2])
            with col2:
                if st.button("🚀 Process & Index Files", use_container_width=True, type="primary", key="process_btn"):
                    from ingestion.ingestion_module import process_and_index_files
                    
                    with st.spinner("📝 Processing documents..."):
                        try:
                            # Save uploaded files temporarily
                            temp_dir = Path("data/temp_uploads")
                            temp_dir.mkdir(parents=True, exist_ok=True)
                            
                            file_paths = []
                            for uploaded_file in uploaded_files:
                                file_path = temp_dir / uploaded_file.name
                                with open(file_path, "wb") as f:
                                    f.write(uploaded_file.getbuffer())
                                file_paths.append(file_path)
                            
                            # Process and index
                            st.info("🔄 Chunking and indexing documents...")
                            result = process_and_index_files(file_paths)
                            
                            if result.get('success', False):
                                st.success(f"✅ Successfully processed {result['total_chunks']} chunks from {len(uploaded_files)} file(s)!")
                                st.balloons()
                                
                                # Show stats
                                st.markdown("### 📊 Indexing Statistics:")
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Files Processed", len(uploaded_files))
                                with col2:
                                    st.metric("Total Chunks", result['total_chunks'])
                                with col3:
                                    st.metric("Vectors Added", result.get('vectors_indexed', result['total_chunks']))
                                
                                # Save uploaded files list
                                st.session_state["uploaded_files_list"].extend([f.name for f in uploaded_files])
                                
                                # Clean up temp files
                                import shutil
                                shutil.rmtree(temp_dir)
                                
                                # Initialize engine
                                st.info("🔧 Initializing RAG engine...")
                                engine, success, message = initialize_query_engine()
                                if success:
                                    st.session_state["query_engine"] = engine
                                    st.session_state["chat_history"] = [
                                        ("bot", "Hi! Saya sudah membaca dokumen Anda. Silakan tanyakan apapun tentang isi dokumen! 📚")
                                    ]
                                    st.session_state["chat_state"] = "chat"
                                    st.success("✅ Ready to chat!")
                                    st.rerun()
                                else:
                                    st.error(message)
                            else:
                                st.error(f"❌ Error: {result.get('error', 'Unknown error')}")
                                
                        except Exception as e:
                            st.error(f"❌ Error processing files: {str(e)}")
                            import traceback
                            with st.expander("Show error details"):
                                st.code(traceback.format_exc())
        else:
            st.info("👆 Pilih file untuk memulai")
    
    # ============================
    # STATE 2: CHAT INTERFACE
    # ============================
    elif st.session_state["chat_state"] == "chat":
        
        # Initialize engine if not already done
        if st.session_state["query_engine"] is None:
            with st.spinner("🔧 Initializing RAG engine..."):
                engine, success, message = initialize_query_engine()
                if success:
                    st.session_state["query_engine"] = engine
                    if not st.session_state["chat_history"]:
                        st.session_state["chat_history"] = [
                            ("bot", "Hi! Silakan tanyakan apapun tentang dokumen Anda! 📚")
                        ]
                else:
                    st.error(message)
                    st.stop()
        
        # Chat container
        chat_container = st.container(height=400)
        
        with chat_container:
            for sender, msg in st.session_state["chat_history"]:
                escaped_msg = msg.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")
                if sender == "user":
                    st.markdown(f'<div class="chat-message-wrapper user"><div class="user-msg">{escaped_msg}</div></div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="chat-message-wrapper bot"><div class="bot-msg">{escaped_msg}</div></div>', unsafe_allow_html=True)
        
        st.markdown("&nbsp;")
        
        # Input area
        col1, col2 = st.columns([8, 1])
        
        with col1:
            user_input = st.text_area(
                "You:", 
                value="",
                key="input_box",
                label_visibility="collapsed",
                placeholder="💭 Tanyakan sesuatu tentang dokumen Anda...",
                height=100
            )
        
        with col2:
            send_clicked = st.button("Send", use_container_width=True, type="primary")
        
        # Handle send action
        if send_clicked and user_input.strip():
            st.session_state["chat_history"].append(("user", user_input))
            
            with st.spinner("🤖 Thinking..."):
                try:
                    engine = st.session_state["query_engine"]
                    result = engine.query(user_input)
                    
                    if result.get('success', False):
                        answer = result['answer']
                        num_sources = result.get('num_sources', 0)
                        
                        if num_sources > 0:
                            avg_score = result['avg_score']
                            response = f"{answer}\n\n📚 Sources: {num_sources} chunks (avg score: {avg_score:.2f})"
                        else:
                            response = f"{answer}\n\n💡 Tip: Coba ubah pertanyaan atau turunkan MIN_SIMILARITY_SCORE di .env"
                        
                        st.session_state["chat_history"].append(("bot", response))
                    else:
                        error_msg = f"❌ Sorry, I encountered an error: {result.get('error', 'Unknown error')}"
                        st.session_state["chat_history"].append(("bot", error_msg))
                    
                except Exception as e:
                    error_msg = f"❌ Error processing your question: {str(e)}"
                    st.session_state["chat_history"].append(("bot", error_msg))
            
            st.rerun()
        
        st.markdown('<div class="div-hr"></div>', unsafe_allow_html=True)
        
        # Two columns: Upload additional docs + Example questions
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### 📤 Upload More Documents")
            
            additional_files = st.file_uploader(
                "Add more documents",
                accept_multiple_files=True,
                type=["pdf", "csv", "txt", "xlsx", "xls", "docx", "doc"],
                key="additional_upload",
                label_visibility="collapsed"
            )
            
            if additional_files:
                st.markdown(f"**{len(additional_files)} file(s) selected**")
                
                if st.button("➕ Add to Knowledge Base", use_container_width=True, type="secondary"):
                    from ingestion.ingestion_module import process_and_index_files
                    
                    with st.spinner("Processing additional documents..."):
                        try:
                            temp_dir = Path("data/temp_uploads")
                            temp_dir.mkdir(parents=True, exist_ok=True)
                            
                            file_paths = []
                            for uploaded_file in additional_files:
                                file_path = temp_dir / uploaded_file.name
                                with open(file_path, "wb") as f:
                                    f.write(uploaded_file.getbuffer())
                                file_paths.append(file_path)
                            
                            result = process_and_index_files(file_paths)
                            
                            if result.get('success', False):
                                st.success(f"✅ Added {result['total_chunks']} new chunks!")
                                st.session_state["uploaded_files_list"].extend([f.name for f in additional_files])
                                
                                # Reinitialize engine
                                st.cache_resource.clear()
                                engine, success, _ = initialize_query_engine()
                                if success:
                                    st.session_state["query_engine"] = engine
                                
                                import shutil
                                shutil.rmtree(temp_dir)
                                st.rerun()
                            else:
                                st.error(f"Error: {result.get('error', 'Unknown error')}")
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
        
        with col2:
            st.markdown("### 💡 Example Questions")
            
            example_questions = [
                "Apa isi utama dari dokumen ini?",
                "Berikan ringkasan singkat",
                "Apa poin-poin penting?",
                "Jelaskan topik utama"
            ]
            
            for idx, question in enumerate(example_questions):
                if st.button(question, key=f"example_{idx}", use_container_width=True):
                    st.session_state["chat_history"].append(("user", question))
                    
                    with st.spinner("🤖 Thinking..."):
                        try:
                            engine = st.session_state["query_engine"]
                            result = engine.query(question)
                            
                            if result['success']:
                                answer = result['answer']
                                num_sources = result['num_sources']
                                avg_score = result['avg_score']
                                
                                response = f"{answer}\n\n📚 Sources: {num_sources} chunks (avg score: {avg_score:.2f})"
                                st.session_state["chat_history"].append(("bot", response))
                            else:
                                error_msg = f"❌ Sorry, I encountered an error: {result.get('error', 'Unknown error')}"
                                st.session_state["chat_history"].append(("bot", error_msg))
                        except Exception as e:
                            error_msg = f"❌ Error: {str(e)}"
                            st.session_state["chat_history"].append(("bot", error_msg))
                    
                    st.rerun()
        
        # Show uploaded files info
        if st.session_state["uploaded_files_list"]:
            st.markdown("---")
            with st.expander(f"📋 Uploaded Files ({len(st.session_state['uploaded_files_list'])})"):
                for idx, filename in enumerate(st.session_state["uploaded_files_list"], 1):
                    st.markdown(f"**{idx}.** {filename}")
