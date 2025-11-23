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

# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(
    page_title="🧠 RAG Builder",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

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
# Enhanced Dark Mode CSS with Glassmorphism & Animations
# ----------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* Main App Styling */
.stApp { 
    background: linear-gradient(135deg, #0a0a0f 0%, #1a1a2e 100%);
    color: #ffffff; 
    font-family: 'Inter', sans-serif;
}

/* Sidebar Styling */
section[data-testid="stSidebar"] {
    background: rgba(20, 20, 30, 0.8);
    backdrop-filter: blur(10px);
    border-right: 1px solid rgba(0, 191, 165, 0.1);
}

section[data-testid="stSidebar"] h1 {
    color: #00BFA5;
    font-weight: 700;
    text-align: center;
    padding: 1rem 0;
    border-bottom: 2px solid rgba(0, 191, 165, 0.3);
    margin-bottom: 2rem;
}

/* Button Styling with Hover Effects */
div.stButton > button {
    background: linear-gradient(135deg, #00BFA5 0%, #00897B 100%);
    color: #ffffff;
    border: none;
    border-radius: 12px;
    padding: 0.75rem 2rem;
    font-weight: 600;
    font-size: 16px;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(0, 191, 165, 0.3);
}

div.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 25px rgba(0, 191, 165, 0.5);
    background: linear-gradient(135deg, #00D4B8 0%, #00BFA5 100%);
}

div.stButton > button:active {
    transform: translateY(0px);
}

/* File Uploader Styling */
.stFileUploader {
    background: rgba(255, 255, 255, 0.03);
    border: 2px dashed rgba(0, 191, 165, 0.3);
    border-radius: 16px;
    padding: 2rem;
    transition: all 0.3s ease;
}

.stFileUploader:hover {
    border-color: rgba(0, 191, 165, 0.6);
    background: rgba(255, 255, 255, 0.05);
}

.stFileUploader label {
    color: #00BFA5 !important;
    font-weight: 600;
}

/* Select Box Styling */
.stSelectbox label {
    color: #00BFA5 !important;
    font-weight: 600;
}

/* Header Styling */
h1, h2, h3 {
    color: #ffffff;
    font-weight: 700;
}

h1 {
    background: linear-gradient(135deg, #00BFA5 0%, #00D4B8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

/* Chat Messages with Glassmorphism */
.user-msg { 
    background: linear-gradient(135deg, rgba(0, 191, 165, 0.15) 0%, rgba(0, 191, 165, 0.1) 100%);
    backdrop-filter: blur(10px);
    color: #ffffff; 
    padding: 12px 18px; 
    border-radius: 18px 18px 4px 18px;
    margin: 8px 0; 
    max-width: 70%; 
    float: right; 
    clear: both; 
    font-size: 15px; 
    font-family: 'Inter', sans-serif;
    border: 1px solid rgba(0, 191, 165, 0.2);
    box-shadow: 0 4px 15px rgba(0, 191, 165, 0.1);
    animation: slideInRight 0.3s ease;
}

.bot-msg { 
    background: rgba(255, 255, 255, 0.05);
    backdrop-filter: blur(10px);
    color: #ffffff; 
    padding: 12px 18px; 
    border-radius: 18px 18px 18px 4px;
    margin: 8px 0; 
    max-width: 70%; 
    float: left; 
    clear: both; 
    font-size: 15px; 
    font-family: 'Inter', sans-serif;
    border: 1px solid rgba(255, 255, 255, 0.1);
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    animation: slideInLeft 0.3s ease;
}

/* Sources Badge */
.source-badge {
    display: inline-block;
    background: rgba(0, 191, 165, 0.2);
    color: #00BFA5;
    padding: 4px 12px;
    border-radius: 12px;
    font-size: 12px;
    font-weight: 600;
    margin-top: 8px;
    border: 1px solid rgba(0, 191, 165, 0.3);
}

/* Animations */
@keyframes slideInRight {
    from {
        opacity: 0;
        transform: translateX(30px);
    }
    to {
        opacity: 1;
        transform: translateX(0);
    }
}

@keyframes slideInLeft {
    from {
        opacity: 0;
        transform: translateX(-30px);
    }
    to {
        opacity: 1;
        transform: translateX(0);
    }
}

/* Loading Animation */
@keyframes pulse {
    0%, 100% {
        opacity: 1;
    }
    50% {
        opacity: 0.5;
    }
}

.loading {
    animation: pulse 1.5s ease-in-out infinite;
}

/* Textarea Styling */
textarea {
    font-size: 15px !important;
    font-family: 'Inter', sans-serif !important;
    background: rgba(255, 255, 255, 0.05) !important;
    border: 2px solid rgba(0, 191, 165, 0.2) !important;
    border-radius: 12px !important;
    color: #ffffff !important;
    padding: 12px !important;
    transition: all 0.3s ease !important;
}

textarea:focus {
    border-color: rgba(0, 191, 165, 0.6) !important;
    box-shadow: 0 0 20px rgba(0, 191, 165, 0.2) !important;
    background: rgba(255, 255, 255, 0.08) !important;
}

textarea::placeholder {
    color: rgba(255, 255, 255, 0.4) !important;
}

/* Success/Warning Messages */
.element-container div[data-testid="stMarkdownContainer"] > div[data-testid="stAlert"] {
    border-radius: 12px;
    border-left: 4px solid #00BFA5;
    background: rgba(0, 191, 165, 0.1);
}

/* Divider Styling */
hr {
    border: none;
    height: 2px;
    background: linear-gradient(90deg, transparent, rgba(0, 191, 165, 0.3), transparent);
    margin: 2rem 0;
}

/* Chat Container */
.chat-container {
    background: rgba(255, 255, 255, 0.02);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    border: 1px solid rgba(255, 255, 255, 0.05);
    min-height: 400px;
    max-height: 500px;
    overflow-y: auto;
}

/* Custom Scrollbar */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}

::-webkit-scrollbar-track {
    background: rgba(255, 255, 255, 0.05);
    border-radius: 10px;
}

::-webkit-scrollbar-thumb {
    background: rgba(0, 191, 165, 0.3);
    border-radius: 10px;
}

::-webkit-scrollbar-thumb:hover {
    background: rgba(0, 191, 165, 0.5);
}

/* Info Box */
.info-box {
    background: rgba(0, 191, 165, 0.1);
    border: 1px solid rgba(0, 191, 165, 0.3);
    border-radius: 12px;
    padding: 1rem;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

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
    st.markdown("---")
    
    # Display current configuration
    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.markdown(f"""
    ### 📊 Current Configuration
    
    **Vector Database:** Qdrant ({settings.QDRANT_MODE} mode)  
    **Collection:** `{settings.QDRANT_COLLECTION_NAME}`  
    **Embedding Model:** {settings.HUGGINGFACE_MODEL if settings.EMBEDDING_PROVIDER == 'huggingface' else settings.OPENAI_EMBEDDING_MODEL}  
    **LLM Model:** {settings.GROQ_LLM_MODEL if settings.LLM_PROVIDER == 'groq' else settings.OPENAI_LLM_MODEL}  
    **Top-K Results:** {settings.RETRIEVAL_TOP_K}  
    **Min Similarity Score:** {settings.MIN_SIMILARITY_SCORE}
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
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
    st.markdown("---")
    
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
    st.markdown("---")
    
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
    
    # Chat container with custom class
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    chat_container = st.container()

    # Display chat history
    with chat_container:
        for sender, msg in st.session_state["chat_history"]:
            # Escape HTML and replace newlines with <br>
            escaped_msg = msg.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")
            if sender == "user":
                st.markdown(f'<div class="user-msg">{escaped_msg}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="bot-msg">{escaped_msg}</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

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
            height=100
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
                        response = f"{answer}\n\n📚 Sources: {num_sources} chunks (avg score: {avg_score:.2f})"
                    else:
                        # No sources found
                        response = f"{answer}\n\n💡 Tip: Try rephrasing your question or lower MIN_SIMILARITY_SCORE in .env"
                    
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
    st.markdown("---")
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
                            
                            response = f"{answer}\n\n"
                            response += f"📚 Sources: {num_sources} chunks (avg score: {avg_score:.2f})"
                            
                            st.session_state["chat_history"].append(("bot", response))
                        else:
                            error_msg = f"❌ Sorry, I encountered an error: {result.get('error', 'Unknown error')}"
                            st.session_state["chat_history"].append(("bot", error_msg))
                    except Exception as e:
                        error_msg = f"❌ Error: {str(e)}"
                        st.session_state["chat_history"].append(("bot", error_msg))
                
                st.rerun()