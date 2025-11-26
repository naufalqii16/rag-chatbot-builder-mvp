"""
Style Loader Utility
---------------------
Utility module for loading and injecting CSS styles into Streamlit app.
"""

from pathlib import Path
import streamlit as st


def load_css(css_file_path: str) -> str:
    """
    Load CSS file and return its content.
    
    Args:
        css_file_path: Path to CSS file (relative or absolute)
    
    Returns:
        CSS content as string
    """
    css_path = Path(css_file_path)
    
    if not css_path.exists():
        # Try relative to this file's directory
        css_path = Path(__file__).parent.parent / css_file_path
    
    if css_path.exists():
        with open(css_path, 'r', encoding='utf-8') as f:
            return f.read()
    else:
        st.warning(f"⚠️ CSS file not found: {css_file_path}")
        return ""


def inject_custom_css(css_file_path: str = "styles/app_styles.css"):
    """
    Inject custom CSS into Streamlit app.
    
    Args:
        css_file_path: Path to CSS file (default: styles/app_styles.css)
    """
    css_content = load_css(css_file_path)
    
    if css_content:
        st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)
