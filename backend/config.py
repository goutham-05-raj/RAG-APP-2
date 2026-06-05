"""
config.py
---------
Central configuration for the RAG application.
All tuneable parameters are read from a .env file so the app
can be reconfigured without touching source code.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from the project root (one level above backend/)
_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_ROOT / ".env")


# ── LLM / Embedding ────────────────────────────────────────────────────────────
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL: str = os.getenv("LLM_MODEL", "llama3")
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# ── Chunking ───────────────────────────────────────────────────────────────────
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))

# ── Retrieval ──────────────────────────────────────────────────────────────────
TOP_K: int = int(os.getenv("TOP_K", "3"))

# ── Storage ────────────────────────────────────────────────────────────────────
DATA_DIR: Path = _ROOT / os.getenv("DATA_DIR", "data")
VECTORSTORE_DIR: Path = _ROOT / os.getenv("VECTORSTORE_DIR", "vectorstore")

# ── CORS ───────────────────────────────────────────────────────────────────────
# Comma-separated list of allowed origins, e.g. "http://localhost:8501,https://myapp.com"
CORS_ORIGINS: list[str] = ["*"]
