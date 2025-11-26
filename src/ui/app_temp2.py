"""
RAG Chatbot Streamlit App
--------------------------
Interactive UI for RAG-powered Q&A system.
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
    page_title="🧠 RAG Builder",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# st.markdown("""
# <style>
# .block-container {
#     padding-top: 2.5rem;
#     padding-bottom: 1rem;
# }
# </style>
# """, unsafe_allow_html=True)

# ----------------------------
# Load Custom CSS
# ----------------------------
inject_custom_css()

# ----------------------------
# Initialize session state
# ----------------------------
if "current_page" not in st.session_state:
    st.session_state["current_page"] = "home"
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = [
        ("bot", "Hi! I'm your RAG-powered assistant. Ask me anything from your dataset.")
    ]
if "query_engine" not in st.session_state:
    st.session_state["query_engine"] = None
if "is_loading" not in st.session_state:
    st.session_state["is_loading"] = False

# ----------------------------
# Initialize RAG Engine (on first load)
# ----------------------------
@st.cache_resource
def initialize_rag_engine():
    """Initialize RAG engine once and cache it."""
    try:
        engine = QueryEngine()
        return engine, True, "✅ RAG engine initialized successfully!"
    except Exception as e:
        return None, False, f"❌ Error initializing RAG: {str(e)}"

# ----------------------------
# Sidebar navigation
# ----------------------------
st.sidebar.title("Navigation")

# Show system info
with st.sidebar.expander("🔧 System Info", expanded=False):
    st.markdown(f"""
    **Embedding:** {settings.EMBEDDING_PROVIDER}  
    **Model:** {settings.HUGGINGFACE_MODEL if settings.EMBEDDING_PROVIDER == 'huggingface' else settings.OPENAI_EMBEDDING_MODEL}  
    **LLM:** {settings.LLM_PROVIDER} ({settings.GROQ_LLM_MODEL if settings.LLM_PROVIDER == 'groq' else settings.OPENAI_LLM_MODEL})  
    **Vector DB:** Qdrant ({settings.QDRANT_MODE})
    """)
    
    if settings.EMBEDDING_PROVIDER == "huggingface" and settings.LLM_PROVIDER == "groq":
        st.success("💰 100% FREE Setup!")

st.sidebar.markdown("---")

# Navigation buttons
if st.session_state["current_page"] == "home":
    if st.sidebar.button("💬 Go to Chatbot", use_container_width=True):
        st.session_state["current_page"] = "chatbot"
        st.rerun()
else:
    if st.sidebar.button("🏠 Go to Home", use_container_width=True):
        st.session_state["current_page"] = "home"
        st.rerun()

# Clear chat button
if st.session_state["current_page"] == "chatbot":
    st.sidebar.markdown("---")
    if st.sidebar.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state["chat_history"] = [
            ("bot", "Hi! I'm your RAG-powered assistant. Ask me anything from your dataset.")
        ]
        st.rerun()

# ----------------------------
# Home page
# ----------------------------
if st.session_state["current_page"] == "home":
    st.header("🏠 Home - RAG System Overview")
    st.markdown('<div class="div-hr"></div>', unsafe_allow_html=True)

        # Display current configuration
    st.markdown(f"""
    <div class="info-box">
        <h3>📊 Current Configuration</h3>
        <p><strong>Vector Database:</strong> Qdrant ({settings.QDRANT_MODE} mode)</p>
        <p><strong>Collection:</strong> {settings.QDRANT_COLLECTION_NAME}</p>
        <p><strong>Embedding Model:</strong> {settings.HUGGINGFACE_MODEL if settings.EMBEDDING_PROVIDER == 'huggingface' else settings.OPENAI_EMBEDDING_MODEL}</p>
        <p><strong>LLM Model:</strong> {settings.GROQ_LLM_MODEL if settings.LLM_PROVIDER == 'groq' else settings.OPENAI_LLM_MODEL}</p>
        <p><strong>Top-K Results:</strong> {settings.RETRIEVAL_TOP_K}</p>
        <p><strong>Min Similarity Score:</strong> {settings.MIN_SIMILARITY_SCORE}</p>
    </div>
    """, unsafe_allow_html=True)
    # st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("&nbsp;")
    
    # File upload section (for future enhancement)
    st.subheader("📁 Upload Additional Documents (Coming Soon)")
    uploaded_files = st.file_uploader(
        "Drag & drop your files (PDF, CSV, TXT)",
        accept_multiple_files=True,
        type=["pdf", "csv", "txt"],
        disabled=True,
        help="This feature will allow you to add more documents to the knowledge base."
    )
    
    st.markdown("&nbsp;")
    st.markdown('<div class="div-hr"></div>', unsafe_allow_html=True)
    
    # Quick actions
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("💬 Start Chatting", use_container_width=True, type="primary"):
            st.session_state["current_page"] = "chatbot"
            st.rerun()
    
    with col2:
        if st.button("📊 View System Stats", use_container_width=True):
            st.info("System statistics feature coming soon!")

# ----------------------------
# Chatbot page
# ----------------------------
elif st.session_state["current_page"] == "chatbot":
    st.header("💬 RAG Chatbot")
    st.markdown('<div class="div-hr"></div>', unsafe_allow_html=True)
    
    # Initialize RAG engine if not already done
    if st.session_state["query_engine"] is None:
        with st.spinner("🔧 Initializing RAG engine..."):
            engine, success, message = initialize_rag_engine()
            if success:
                st.session_state["query_engine"] = engine
                st.success(message)
            else:
                st.error(message)
                st.stop()
    
    # Chat container with height
    chat_container = st.container(height=500)

    # chat_container = st.container()
        
    with chat_container:
        # Display chat history
        for sender, msg in st.session_state["chat_history"]:
            # Escape HTML and replace newlines with <br>
            escaped_msg = msg.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")
            if sender == "user":
                st.markdown(f'<div class="chat-message-wrapper user"><div class="user-msg">{escaped_msg}</div></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-message-wrapper bot"><div class="bot-msg">{escaped_msg}</div></div>', unsafe_allow_html=True)

    # ----------------------------
    # Input + Send button (side by side)
    # ----------------------------
    st.markdown("&nbsp;")
    
    col1, col2 = st.columns([8, 1])
    
    with col1:
        user_input = st.text_area(
            "You:", 
            value="",
            key="input_box",
            label_visibility="collapsed",
            placeholder="💭 Type your question here... (e.g., 'Which tables use incremental extraction?')",
            height=15
        )
    
    with col2:
        send_clicked = st.button("Send", use_container_width=True, type="primary")

    # ----------------------------
    # Handle send action with RAG
    # ----------------------------
    if send_clicked and user_input.strip():
        # Add user message to chat
        st.session_state["chat_history"].append(("user", user_input))
        
        # Show loading state
        with st.spinner("🤖 Thinking..."):
            try:
                # Query the RAG engine
                engine = st.session_state["query_engine"]
                result = engine.query(user_input)
                
                if result.get('success', False):
                    # Format response
                    answer = result['answer']
                    num_sources = result.get('num_sources', 0)
                    
                    if num_sources > 0:
                        # Has sources
                        avg_score = result['avg_score']
                        response = f"{answer} 📚 Sources: {num_sources} chunks (avg score: {avg_score:.2f})"
                    else:
                        # No sources found
                        response = f"{answer} 💡 Tip: Try rephrasing your question or lower MIN_SIMILARITY_SCORE in .env"
                    
                    st.session_state["chat_history"].append(("bot", response))
                else:
                    error_msg = f"❌ Sorry, I encountered an error: {result.get('error', 'Unknown error')}"
                    st.session_state["chat_history"].append(("bot", error_msg))
                
            except Exception as e:
                import traceback
                error_detail = traceback.format_exc()
                error_msg = f"❌ Error processing your question: {str(e)}"
                st.session_state["chat_history"].append(("bot", error_msg))
        
        # Rerun to update UI
        st.rerun()

    # ----------------------------
    # Example questions
    # ----------------------------
    st.markdown('<div class="div-hr"></div>', unsafe_allow_html=True)
    st.markdown("### 💡 Example Questions:")
    
    example_questions = [
        "Which tables use incremental extraction with watermark datetime?",
        "What are the tables in the EMR database?",
        "Show me tables with full load extraction mode",
        "Which tables have UUID as primary key?"
    ]
    
    cols = st.columns(2)
    for idx, question in enumerate(example_questions):
        with cols[idx % 2]:
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
                            
                            response = f"{answer}"
                            response += f"📚 Sources: {num_sources} chunks (avg score: {avg_score:.2f})"
                            
                            st.session_state["chat_history"].append(("bot", response))
                        else:
                            error_msg = f"❌ Sorry, I encountered an error: {result.get('error', 'Unknown error')}"
                            st.session_state["chat_history"].append(("bot", error_msg))
                    except Exception as e:
                        error_msg = f"❌ Error: {str(e)}"
                        st.session_state["chat_history"].append(("bot", error_msg))
                
                st.rerun()
