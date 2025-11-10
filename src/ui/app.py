import streamlit as st

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
</style>
""", unsafe_allow_html=True)

# ----------------------------
# Sidebar navigation
# ----------------------------
st.sidebar.title("Navigation")
if st.session_state["current_page"] == "home":
    st.sidebar.button("Go to Chatbot", on_click=lambda: st.session_state.update({"current_page": "chatbot"}))
else:
    st.sidebar.button("Go to Home", on_click=lambda: st.session_state.update({"current_page": "home"}))

# ----------------------------
# Home page
# ----------------------------
if st.session_state["current_page"] == "home":
    st.header("🏠 Home - Upload Dataset & Create Chatbot")
    st.markdown("---")

    uploaded_files = st.file_uploader(
        "📁 Drag & drop your files (PDF, CSV, TXT)",
        accept_multiple_files=True,
        type=["pdf", "csv", "txt"],
    )
    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} file(s) uploaded successfully!")

    st.markdown("&nbsp;")
    model = st.selectbox("🤖 Select embedding model:", ["text-embedding-3-small", "text-embedding-3-large"])

    st.markdown("&nbsp;")
    if st.button("Create Chatbot ▶"):
        if uploaded_files:
            st.session_state["current_page"] = "chatbot"
            st.rerun()
        else:
            st.warning("⚠️ Please upload at least one file before creating the chatbot.")

# ----------------------------
# Chatbot page
# ----------------------------
elif st.session_state["current_page"] == "chatbot":
    st.header("💬 RAG Chatbot")
    st.markdown("---")
    
    # Chat container with custom class
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    chat_container = st.container()

    # Display chat history
    with chat_container:
        for sender, msg in st.session_state["chat_history"]:
            # Escape HTML dan replace newlines dengan <br>
            escaped_msg = msg.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")
            if sender == "user":
                st.markdown(f'<div class="user-msg">{escaped_msg}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="bot-msg">{escaped_msg}</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

    # ----------------------------
    # Input + Send button (side by side, button di kanan bawah)
    # ----------------------------
    st.markdown("&nbsp;")
    
    col1, col2 = st.columns([8, 1])
    
    with col1:
        user_input = st.text_area(
            "You:", 
            value="",
            key="input_box",
            label_visibility="collapsed",
            placeholder="💭 Type your message here...",
            height=100
        )
    
    with col2:
        send_clicked = st.button("Send", use_container_width=True)

    # ----------------------------
    # Handle send action
    # ----------------------------
    if send_clicked and user_input.strip():
        # Tambah ke chat_history
        st.session_state["chat_history"].append(("user", user_input))
        st.session_state["chat_history"].append(("bot", "🤖 This is a placeholder response. Your RAG system will provide real answers here!"))
        
        # Rerun untuk update UI
        st.rerun()