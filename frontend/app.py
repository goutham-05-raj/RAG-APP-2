"""
app.py
------
Fallstaff - THE FINAL BOSS EDITION.
- Sky Blue Premium Background.
- Search Input: Indigo background with WHITE text.
- Auto-Ingestion: No 'Index' button, just a progress bar on upload.
- Elegant B2B SaaS layout (900px centered).
"""

import time
import requests
import streamlit as st
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
BACKEND_URL = "http://localhost:8000"
UPLOAD_ENDPOINT = f"{BACKEND_URL}/upload"
ASK_ENDPOINT = f"{BACKEND_URL}/ask"
REQUEST_TIMEOUT = 120

# ── Page Setup ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fallstaff | Intelligence Search",
    page_icon="⚜️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS — Fallstaff Final Boss Aesthetic ───────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

    /* Global Typography & Sky Blue Background */
    html, body, [class*="st-"] {
        font-family: 'Inter', sans-serif;
        color: #0f172a;
    }
    .stApp {
        background: radial-gradient(circle at 50% 50%, #f0f9ff 0%, #e0f2fe 100%);
        background-attachment: fixed;
    }

    /* Hide Default Elements */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}

    /* Centered Main Container */
    .block-container {
        max-width: 900px !important;
        padding-top: 3rem !important;
        padding-bottom: 10rem !important;
        margin: 0 auto;
    }

    /* Header Section */
    .header-section {
        text-align: center;
        margin-bottom: 3rem;
    }
    .header-top h1 {
        font-size: 4.8rem;
        font-weight: 900;
        color: #ef4444; /* Branding Red */
        letter-spacing: -0.06em;
        margin-bottom: 0.2rem;
    }
    .header-top p {
        font-size: 1.25rem;
        font-weight: 500;
        color: #475569;
        margin-top: -10px;
    }

    /* Premium Cards */
    .premium-card {
        background: rgba(255, 255, 255, 0.8);
        border: 1px solid rgba(255, 255, 255, 0.4);
        border-radius: 20px;
        padding: 30px;
        backdrop-filter: blur(20px);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.04);
        margin-bottom: 2.5rem;
    }

    /* BROWSE FILES PANEL - White Button Logic */
    .stFileUploader section {
        background-color: white !important;
        border: 2px solid #e2e8f0 !important;
        border-radius: 16px !important;
    }
    .stFileUploader button {
        background-color: white !important;
        background: white !important;
        color: #0f172a !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05) !important;
    }
    .stFileUploader button:hover {
        background-color: #f8fafc !important;
        border-color: #94a3b8 !important;
    }

    /* SEARCH INPUT - INDIGO with WHITE TEXT */
    /* This targets the actual input within the chat_input widget */
    .stChatInput textarea {
        background-color: #4f46e5 !important; /* Indigo Background */
        color: white !important; /* WHITE TEXT WHILE TYPING */
        caret-color: white !important;
        border-radius: 30px !important;
        padding: 15px 25px !important;
        border: none !important;
        box-shadow: 0 20px 40px rgba(79, 70, 229, 0.2) !important;
    }
    /* Placeholder for the Indigo input */
    .stChatInput textarea::placeholder {
        color: rgba(255, 255, 255, 0.7) !important;
    }
    
    /* Progress Bar Color */
    .stProgress > div > div > div > div {
        background-color: #4f46e5 !important;
    }

    /* Message Bubbles */
    .chat-row {
        display: flex;
        width: 100%;
        margin-bottom: 1.5rem;
        animation: fadeIn 0.4s ease-out;
    }
    .user-row { justify-content: flex-end; }
    .assistant-row { justify-content: flex-start; }

    .bubble {
        padding: 14px 20px;
        border-radius: 18px;
        font-size: 1rem;
        line-height: 1.6;
        max-width: 85%;
        box-shadow: 0 4px 6px rgba(0,0,0,0.02);
    }
    .user-bubble {
        background: #4f46e5;
        color: white;
        border-bottom-right-radius: 4px;
        font-weight: 500;
    }
    .assistant-bubble {
        background-color: white;
        color: #0f172a;
        border: 1px solid #e2e8f0;
        border-bottom-left-radius: 4px;
    }

    /* Intelligence Trace */
    .source-trace {
        margin-top: 10px;
        font-size: 0.85rem;
        color: #64748b;
        border-top: 1px solid #f1f5f9;
        padding-top: 10px;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(15px); }
        to { opacity: 1; transform: translateY(0); }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── State Initialisation ─────────────────────────────────────────────────────

def _init_state():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "indexed_files" not in st.session_state:
        st.session_state.indexed_files = []
    if "last_processed" not in st.session_state:
        st.session_state.last_processed = None


_init_state()


# ── Logic ─────────────────────────────────────────────────────────────────────

def _handle_auto_ingestion(file):
    # If this file was already processed in this toggle, skip
    if st.session_state.last_processed == file.name:
        return

    progress_bar = st.progress(0)
    status_text = st.empty()
    
    status_text.markdown(f"🛰️ **Initialising Fallstaff context for {file.name}...**")
    time.sleep(0.5)
    progress_bar.progress(30)
    
    try:
        response = requests.post(
            UPLOAD_ENDPOINT,
            files={"file": (file.name, file.getvalue(), "application/pdf")},
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()
        
        progress_bar.progress(100)
        status_text.markdown(f"✅ **{file.name} successfully indexed.**")
        
        st.session_state.indexed_files.append({
            "name": file.name,
            "chunks": data["chunks_indexed"]
        })
        st.session_state.last_processed = file.name
        time.sleep(1)
        st.rerun()
    except Exception as e:
        status_text.error(f"Failed to index: {e}")
        progress_bar.empty()


def _handle_ask(question):
    if not question or not question.strip():
        return

    if not st.session_state.indexed_files:
        st.warning("Please upload a PDF source first.")
        return

    st.session_state.chat_history.append({"role": "user", "content": question})

    with st.spinner("💭 Fallstaff is thinking..."):
        try:
            response = requests.post(
                ASK_ENDPOINT,
                json={"question": question},
                timeout=REQUEST_TIMEOUT,
            )
            response.raise_for_status()
            data = response.json()
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": data["answer"],
                "sources": data["sources"]
            })
        except Exception as e:
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": f"AI Engine Link Failure: {e}",
                "sources": []
            })
    st.rerun()


# ── Main UI Rendering ─────────────────────────────────────────────────────────

# Header
st.markdown(
    """
    <div class="header-top" style="text-align: center; padding: 40px 0;">
        <h1>Fallstaff</h1>
        <p style="color: #64748b; font-size: 1.2rem;">Analyze and Make Your Documents Wisely</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Upload Card (Auto-Processing)
st.markdown('<div class="premium-card">', unsafe_allow_html=True)
st.markdown('<h4 style="font-weight: 700; margin-bottom: 1.5rem; color: #1e293b;">📥 Knowledge Repository</h4>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Upload Source PDF",
    type=["pdf"],
    label_visibility="collapsed",
    key="final_uploader"
)

if uploaded_file:
    _handle_auto_ingestion(uploaded_file)

if st.session_state.indexed_files:
    st.markdown('<div style="margin-top: 1rem; display: flex; flex-wrap: wrap; gap: 8px;">', unsafe_allow_html=True)
    for f in st.session_state.indexed_files:
        st.markdown(f'<div style="background: #e0f2fe; color: #0369a1; padding: 6px 14px; border-radius: 10px; font-size: 0.85rem; font-weight: 600;">📎 {f["name"]}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Chat History
if not st.session_state.chat_history:
    st.markdown(
        """
        <div style="text-align: center; margin-top: 4rem; color: #94a3b8; font-size: 1.1rem;">
            Ready for your analytical queries.
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    for msg in st.session_state.chat_history:
        role = msg["role"]
        row_class = "user-row" if role == "user" else "assistant-row"
        bubble_class = "user-bubble" if role == "user" else "assistant-bubble"
        
        sources_html = ""
        if role == "assistant" and msg.get("sources"):
            sources_list = "".join([f"<li>{s}</li>" for s in msg["sources"][:3]])
            sources_html = f'<div class="source-trace"><details><summary>Intelligence Context</summary><ul>{sources_list}</ul></details></div>'

        st.markdown(
            f"""
            <div class="chat-row {row_class}">
                <div class="bubble {bubble_class}">
                    <div>{msg['content']}</div>
                    {sources_html}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

# Sticky Bottom Pill Input (Indigo background, White Text)
query = st.chat_input("Ask Fallstaff about your documents...")
if query:
    _handle_ask(query)
