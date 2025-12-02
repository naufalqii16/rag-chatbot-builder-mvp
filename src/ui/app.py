"""
RAG Chatbot Streamlit App - Redesigned
---------------------------------------
Two-page structure: Home and Chat
ULTRA-FIXED AUTO-SCROLL VERSION
"""


import streamlit as st
import streamlit.components.v1 as components
import sys
from pathlib import Path
import time

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
    page_title="🧠 BitMate AI",
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
if "scroll_trigger" not in st.session_state:
    st.session_state["scroll_trigger"] = 0
if "indexing_stats" not in st.session_state:
    st.session_state["indexing_stats"] = None
if "show_stats_popup" not in st.session_state:
    st.session_state["show_stats_popup"] = False
if "upload_counter" not in st.session_state:
    st.session_state["upload_counter"] = 0
if "is_processing" not in st.session_state:
    st.session_state["is_processing"] = False

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
    """Reset chatbot to initial state - COMPLETE RESET including Qdrant data"""
    try:
        # Delete Qdrant collection to remove all old documents
        from qdrant_client import QdrantClient
        
        if settings.QDRANT_MODE == "local":
            client = QdrantClient(path=settings.QDRANT_PATH)
        else:
            client = QdrantClient(
                url=settings.QDRANT_URL,
                api_key=settings.QDRANT_API_KEY
            )
        
        # Delete the collection if it exists
        try:
            client.delete_collection(collection_name=settings.QDRANT_USER_UPLOAD_COLLECTION)
            print(f"✅ Deleted Qdrant collection: {settings.QDRANT_USER_UPLOAD_COLLECTION}")
        except Exception as e:
            print(f"⚠️ Collection doesn't exist or already deleted: {e}")
        
    except Exception as e:
        print(f"❌ Error deleting Qdrant collection: {e}")
    
    # Reset all session state
    st.session_state["current_page"] = "home"
    st.session_state["chat_state"] = "upload"
    st.session_state["chat_history"] = []
    st.session_state["uploaded_files_list"] = []
    st.session_state["scroll_trigger"] = 0
    st.session_state["query_engine"] = None
    st.session_state["indexing_stats"] = None
    st.session_state["show_stats_popup"] = False
    st.session_state["upload_counter"] = 0
    st.session_state["is_processing"] = False
    
    # Clear all caches
    st.cache_resource.clear()
    st.cache_data.clear()

# ============================
# HOME PAGE
# ============================
if st.session_state["current_page"] == "home":
    
    # Header with Start Chat button
    col1, col2 = st.columns([6, 1])
    with col1:
        st.title("🧠 BitMate AI")
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
        if st.button("🏠", key="home_button", help="Go Home (resets the chatbot)",  use_container_width=True):
            reset_chatbot()
            st.rerun()
  
    with col2:
        st.title("💬 Chat with Your BitMate")
    
    st.markdown('<div class="div-hr"></div>', unsafe_allow_html=True)
    
    # ============================
    # STATE 1: UPLOAD DOCUMENTS
    # ============================
    if st.session_state["chat_state"] == "upload" and st.session_state["indexing_stats"] is None:
        
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
            label_visibility="collapsed",
            key=f"initial_upload_{st.session_state['upload_counter']}"
        )
        
        # Display uploaded files
        if uploaded_files:
            st.markdown("&nbsp;")
            
            # Process button
            col1, col2, col3 = st.columns([2, 3, 2])
            with col2:
                if st.button("🚀 Process & Index Files", use_container_width=True, type="primary", key="process_btn"):
                    from ingestion.ingestion_module import process_and_index_files
                    
                    # Create progress bar
                    progress_text = st.empty()
                    progress_bar = st.progress(0)
                    
                    try:
                        # Step 1: Save files (20%)
                        progress_text.text("📁 Saving uploaded files...")
                        progress_bar.progress(20)
                        
                        temp_dir = Path("data/temp_uploads")
                        temp_dir.mkdir(parents=True, exist_ok=True)
                        
                        file_paths = []
                        for idx, uploaded_file in enumerate(uploaded_files):
                            file_path = temp_dir / uploaded_file.name
                            with open(file_path, "wb") as f:
                                f.write(uploaded_file.getbuffer())
                            file_paths.append(file_path)
                            
                            # Update progress for each file
                            file_progress = 20 + (idx + 1) / len(uploaded_files) * 20
                            progress_bar.progress(int(file_progress))
                        
                        # Step 2: Process and index (40-80%)
                        progress_text.markdown('<p style="color: white; font-size: 16px;">🔄 Chunking and indexing documents...</p>', unsafe_allow_html=True)
                        progress_bar.progress(50)
                        
                        result = process_and_index_files(file_paths)
                        progress_bar.progress(80)
                        
                        if result.get('success', False):
                            # Step 3: Initialize engine (80-100%)
                            progress_text.text("🔧 Initializing RAG engine...")
                            progress_bar.progress(90)
                            
                            engine, success, message = initialize_query_engine()
                            
                            if success:
                                progress_bar.progress(100)
                                progress_text.markdown('<p style="color: white; font-size: 16px;">✅ Processing complete!</p>', unsafe_allow_html=True)
                                
                                # Save uploaded files list
                                st.session_state["uploaded_files_list"].extend([f.name for f in uploaded_files])
                                
                                # Clean up temp files
                                import shutil
                                shutil.rmtree(temp_dir)
                                
                                # Set engine and prepare for stats display
                                st.session_state["query_engine"] = engine
                                st.session_state["indexing_stats"] = {
                                    'num_files': len(uploaded_files),
                                    'total_chunks': result['total_chunks'],
                                    'vectors_indexed': result.get('vectors_indexed', result['total_chunks'])
                                }
                                st.session_state["chat_history"] = [
                                    ("bot", "Hi! Saya sudah membaca dokumen Anda. Silakan tanyakan apapun tentang isi dokumen! 📚")
                                ]
                                st.session_state["scroll_trigger"] = 1
                                st.session_state["upload_counter"] += 1  # Increment to clear file uploader
                                st.rerun()
                            else:
                                progress_bar.empty()
                                progress_text.empty()
                                st.error(message)
                        else:
                            progress_bar.empty()
                            progress_text.empty()
                            st.error(f"❌ Error: {result.get('error', 'Unknown error')}")
                            
                    except Exception as e:
                        progress_bar.empty()
                        progress_text.empty()
                        st.error(f"❌ Error processing files: {str(e)}")
                        import traceback
                        with st.expander("Show error details"):
                            st.code(traceback.format_exc())
        else:
            st.info("👆 Pilih file untuk memulai")
    
    # ============================
    # STATE 1.5: SHOW INDEXING STATS
    # ============================
    elif st.session_state["chat_state"] == "upload" and st.session_state["indexing_stats"] is not None:
        
        st.success(f"✅ Successfully processed {st.session_state['indexing_stats']['total_chunks']} chunks from {st.session_state['indexing_stats']['num_files']} file(s)!")
        st.balloons()
        
        st.markdown("&nbsp;")
        
        # Show stats
        st.markdown("### 📊 Indexing Statistics:")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Files Processed", st.session_state['indexing_stats']['num_files'])
        with col2:
            st.metric("Total Chunks", st.session_state['indexing_stats']['total_chunks'])
        with col3:
            st.metric("Vectors Added", st.session_state['indexing_stats']['vectors_indexed'])
        
        st.markdown("&nbsp;")
        st.markdown("&nbsp;")
        
        # Button to go to chatbot
        col1, col2, col3 = st.columns([2, 3, 2])
        with col2:
            if st.button("🚀 Go to Chatbot", use_container_width=True, type="primary", key="go_to_chat_btn"):
                st.session_state["chat_state"] = "chat"
                st.session_state["show_stats_popup"] = False
                st.rerun()
    
    # ============================
    # STATE 2: CHAT INTERFACE
    # ============================
    elif st.session_state["chat_state"] == "chat":
    
        # Show stats popup if needed
        if st.session_state["show_stats_popup"] and st.session_state["indexing_stats"] is not None:
            
            @st.dialog("✅ Indexing Complete!")
            def show_stats_dialog():
                st.success(f"Successfully processed {st.session_state['indexing_stats']['total_chunks']} chunks from {st.session_state['indexing_stats']['num_files']} file(s)!")
                
                st.markdown("### 📊 Indexing Statistics:")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Files Processed", st.session_state['indexing_stats']['num_files'])
                with col2:
                    st.metric("Total Chunks", st.session_state['indexing_stats']['total_chunks'])
                with col3:
                    st.metric("Vectors Added", st.session_state['indexing_stats']['vectors_indexed'])
                
                st.markdown("&nbsp;")
                
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    if st.button("OK", use_container_width=True, type="primary", key="ok_stats_btn"):
                        st.session_state["show_stats_popup"] = False
                        st.rerun()
            
            show_stats_dialog()
        
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
        
        # Chat container with fixed height and unique ID
        chat_container = st.container(height=500)
        
        with chat_container:
            for idx, (sender, msg) in enumerate(st.session_state["chat_history"]):
                escaped_msg = msg.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")
                if sender == "user":
                    st.markdown(f'<div class="chat-message-wrapper user" data-msg-id="{idx}"><div class="user-msg">{escaped_msg}</div></div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="chat-message-wrapper bot" data-msg-id="{idx}"><div class="bot-msg">{escaped_msg}</div></div>', unsafe_allow_html=True)
            
            # Add invisible marker at the end
            st.markdown(f'<div id="chat-end-marker" style="height: 1px;" data-trigger="{st.session_state["scroll_trigger"]}"></div>', unsafe_allow_html=True)
        
        # Reduced spacing
        st.markdown('<div style="height: 8px;"></div>', unsafe_allow_html=True)
        
        # ============================================
        # ULTRA-AGGRESSIVE AUTO-SCROLL
        # ============================================
        scroll_js = f"""
        <script>
        (function() {{
            const scrollTrigger = {st.session_state["scroll_trigger"]};
            const messageCount = {len(st.session_state["chat_history"])};
            
            console.log('🔄 Scroll script loaded - Trigger:', scrollTrigger, 'Messages:', messageCount);
            
            let scrollAttempts = 0;
            let scrollSuccess = false;
            
            function forceScroll() {{
                scrollAttempts++;
                const doc = window.parent.document;
                
                // Method 1: Find by chat messages
                const containers = doc.querySelectorAll('[data-testid="stVerticalBlockBorderWrapper"]');
                
                for (let container of containers) {{
                    const messages = container.querySelectorAll('.chat-message-wrapper');
                    
                    if (messages.length > 0) {{
                        const oldScroll = container.scrollTop;
                        container.scrollTop = container.scrollHeight + 9999;
                        
                        console.log('📍 Attempt', scrollAttempts, '- Scrolled from', oldScroll, 'to', container.scrollTop, '/', container.scrollHeight);
                        
                        if (container.scrollTop > oldScroll || container.scrollTop > container.scrollHeight - container.clientHeight - 50) {{
                            scrollSuccess = true;
                        }}
                        return true;
                    }}
                }}
                
                // Method 2: Find by marker
                const marker = doc.querySelector('#chat-end-marker');
                if (marker) {{
                    marker.scrollIntoView({{ behavior: 'auto', block: 'end' }});
                    console.log('📍 Scrolled using marker');
                    return true;
                }}
                
                return false;
            }}
            
            // Ultra-aggressive retry mechanism
            function aggressiveScroll() {{
                const intervals = [0, 50, 100, 150, 200, 300, 400, 500, 700, 1000, 1500, 2000, 2500, 3000];
                
                intervals.forEach((delay, index) => {{
                    setTimeout(() => {{
                        if (!scrollSuccess || index < 5) {{
                            forceScroll();
                        }}
                    }}, delay);
                }});
            }}
            
            // Start aggressive scrolling
            if (messageCount > 0) {{
                aggressiveScroll();
            }}
            
            // MutationObserver - watch for any DOM changes
            const observer = new MutationObserver((mutations) => {{
                forceScroll();
            }});
            
            setTimeout(() => {{
                const main = window.parent.document.querySelector('main');
                if (main) {{
                    observer.observe(main, {{ 
                        childList: true, 
                        subtree: true,
                        attributes: true,
                        characterData: true
                    }});
                    console.log('👀 MutationObserver active');
                }}
            }}, 100);
            
            // ============================================
            // KEYBOARD SHORTCUTS
            // ============================================
            window.parent.document.addEventListener('keydown', function(e) {{
                if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {{
                    e.preventDefault();
                    const btns = window.parent.document.querySelectorAll('button[data-testid="stBaseButton-primary"]');
                    for (let btn of btns) {{
                        if (btn.textContent.includes('➤')) {{
                            btn.click();
                            break;
                        }}
                    }}
                }}
            }});
            
            setTimeout(() => {{
                const textareas = window.parent.document.querySelectorAll('textarea');
                for (let ta of textareas) {{
                    if (ta.placeholder && ta.placeholder.includes('Tanyakan')) {{
                        ta.addEventListener('keydown', e => {{
                            if (e.key === 'Enter' && !e.ctrlKey && !e.metaKey) e.preventDefault();
                        }});
                        break;
                    }}
                }}
            }}, 100);
        }})();
        </script>
        """
        
        components.html(scroll_js, height=0)
        
        # Input area
        col1, col2 = st.columns([10, 1])
        
        with col1:
            input_key = f"input_box_{len(st.session_state['chat_history'])}"
            user_input = st.text_area(
                "You:", 
                value="",
                key=input_key,
                label_visibility="collapsed",
                placeholder="💭 Tanyakan sesuatu tentang dokumen Anda...",
                height=50
            )
        
        with col2:
            send_clicked = st.button("➤", use_container_width=True, type="primary", help="Send message (Ctrl+Enter)", key="send_button", disabled=st.session_state["is_processing"])
        
        # Handle send action
        if send_clicked and user_input.strip():
            user_query = user_input.strip()
            st.session_state["chat_history"].append(("user", user_query))
            
            with st.spinner("🤖 Thinking..."):
                try:
                    engine = st.session_state["query_engine"]
                    result = engine.query(user_query)
                    
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
            
            # Increment scroll trigger to force new scroll
            st.session_state["scroll_trigger"] += 1
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
                key=f"additional_upload_{st.session_state['upload_counter']}",
                label_visibility="collapsed",
                disabled=st.session_state["is_processing"]
            )
            
            if additional_files:
                st.markdown(f"**{len(additional_files)} file(s) selected**")
                
                if st.button("➕ Add to Knowledge Base", use_container_width=True, type="secondary", disabled=st.session_state["is_processing"]):
                    from ingestion.ingestion_module import process_and_index_files
                    
                    # Set processing flag
                    st.session_state["is_processing"] = True
                    
                    progress_text = st.empty()
                    progress_bar = st.progress(0)
                    
                    try:
                        progress_text.text("📁 Saving files...")
                        progress_bar.progress(20)
                        
                        temp_dir = Path("data/temp_uploads")
                        temp_dir.mkdir(parents=True, exist_ok=True)
                        
                        file_paths = []
                        for idx, uploaded_file in enumerate(additional_files):
                            file_path = temp_dir / uploaded_file.name
                            with open(file_path, "wb") as f:
                                f.write(uploaded_file.getbuffer())
                            file_paths.append(file_path)
                            
                            file_progress = 20 + (idx + 1) / len(additional_files) * 20
                            progress_bar.progress(int(file_progress))
                        
                        progress_text.markdown('<p style="color: white; font-size: 16px;">🔄 Processing documents...</p>', unsafe_allow_html=True)
                        progress_bar.progress(50)
                        
                        result = process_and_index_files(file_paths)
                        progress_bar.progress(80)
                        
                        if result.get('success', False):
                            progress_text.text("🔧 Updating engine...")
                            progress_bar.progress(90)
                            
                            st.session_state["uploaded_files_list"].extend([f.name for f in additional_files])
                            
                            st.cache_resource.clear()
                            engine, success, _ = initialize_query_engine()
                            if success:
                                st.session_state["query_engine"] = engine
                            
                            import shutil
                            shutil.rmtree(temp_dir)
                            
                            progress_bar.progress(100)
                            progress_text.markdown('<p style="color: white; font-size: 16px;">✅ Complete!</p>', unsafe_allow_html=True)
                            
                            # Store stats for popup
                            st.session_state["indexing_stats"] = {
                                'num_files': len(additional_files),
                                'total_chunks': result['total_chunks'],
                                'vectors_indexed': result.get('vectors_indexed', result['total_chunks'])
                            }
                            st.session_state["show_stats_popup"] = True
                            st.session_state["upload_counter"] += 1  # Increment to clear file uploader
                            st.session_state["is_processing"] = False  # Reset processing flag
                            
                            st.success(f"✅ Added {result['total_chunks']} new chunks!")
                            st.rerun()
                        else:
                            st.session_state["is_processing"] = False  # Reset on error
                            progress_bar.empty()
                            progress_text.empty()
                            st.error(f"Error: {result.get('error', 'Unknown error')}")
                    except Exception as e:
                        st.session_state["is_processing"] = False  # Reset on exception
                        progress_bar.empty()
                        progress_text.empty()
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
                if st.button(question, key=f"example_{idx}", use_container_width=True, disabled=st.session_state["is_processing"]):
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
                    
                    st.session_state["scroll_trigger"] += 1
                    st.rerun()
        
        # Show uploaded files info
        if st.session_state["uploaded_files_list"]:
            st.markdown("---")
            with st.expander(f"📋 Uploaded Files ({len(st.session_state['uploaded_files_list'])})"):
                for idx, filename in enumerate(st.session_state["uploaded_files_list"], 1):
                    st.markdown(f"**{idx}.** {filename}")
