"""
ingest.py
---------
Handles the full document ingestion pipeline:
  1. Load PDF from disk
  2. Split text into overlapping chunks
  3. Generate embeddings with sentence-transformers
  4. Build and persist a FAISS index
  5. Load an existing index from disk
"""

import pickle
from pathlib import Path
from typing import List, Tuple
import os

# Limit thread usage to save memory on free tier cloud instances
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

from config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    EMBEDDING_MODEL,
    VECTORSTORE_DIR,
)
from utils import get_logger

logger = get_logger(__name__)

# ── Singleton embedding model (loaded once per process) ────────────────────────
_embedder = None


def _get_embedder():
    """Lazily load the SentenceTransformer model and cache it in module scope."""
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading embedding model: %s", EMBEDDING_MODEL)
        _embedder = SentenceTransformer(EMBEDDING_MODEL)
    return _embedder


# ── PDF Loading ────────────────────────────────────────────────────────────────

def load_pdf(pdf_path: Path) -> str:
    """
    Extract all text from a PDF file.

    Args:
        pdf_path: Absolute path to the PDF file.

    Returns:
        Concatenated text of all pages.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If no text could be extracted.
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    from pypdf import PdfReader
    reader = PdfReader(str(pdf_path))
    pages_text: List[str] = []

    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            pages_text.append(text)

    if not pages_text:
        raise ValueError("No extractable text found in PDF.")

    full_text = "\n".join(pages_text)
    logger.info("Extracted %d characters from %d pages", len(full_text), len(pages_text))
    return full_text


# ── Text Splitting ────────────────────────────────────────────────────────────

def split_text(text: str) -> List[str]:
    """
    Split raw text into overlapping chunks suitable for embedding.

    Args:
        text: Full document text.

    Returns:
        List of text chunk strings.
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks = splitter.split_text(text)
    logger.info("Split text into %d chunks (size=%d, overlap=%d)", len(chunks), CHUNK_SIZE, CHUNK_OVERLAP)
    return chunks


# ── FAISS Index ───────────────────────────────────────────────────────────────

def build_vectorstore(chunks: List[str]) -> Tuple["faiss.IndexFlatL2", List[str]]:
    """
    Embed chunks and build a FAISS L2 index.

    Args:
        chunks: List of text chunks.

    Returns:
        Tuple of (faiss_index, chunks) where chunks is preserved for lookup.
    """
    import faiss
    import numpy as np
    
    embedder = _get_embedder()
    # Use a small batch_size to prevent Out Of Memory crashes on free tiers
    embeddings = embedder.encode(chunks, batch_size=8, show_progress_bar=False, convert_to_numpy=True)
    embeddings = np.array(embeddings, dtype="float32")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    logger.info("Built FAISS index with %d vectors (dim=%d)", index.ntotal, dimension)
    return index, chunks


def save_vectorstore(index: "faiss.IndexFlatL2", chunks: List[str], name: str = "index") -> None:
    """
    Persist FAISS index and associated chunk metadata to disk.

    Args:
        index:  FAISS index to save.
        chunks: Corresponding text chunks (saved as pickle).
        name:   Base filename (without extension) for the saved files.
    """
    import faiss
    
    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(VECTORSTORE_DIR / f"{name}.faiss"))
    with open(VECTORSTORE_DIR / f"{name}.pkl", "wb") as f:
        pickle.dump(chunks, f)
    logger.info("Vectorstore saved → %s", VECTORSTORE_DIR / name)


def load_vectorstore(name: str = "index") -> Tuple["faiss.IndexFlatL2", List[str]]:
    """
    Load a previously saved FAISS index and chunk metadata from disk.

    Args:
        name: Base filename (without extension) matching what was used in save_vectorstore.

    Returns:
        Tuple of (faiss_index, chunks).

    Raises:
        FileNotFoundError: If the index files do not exist on disk.
    """
    index_path = VECTORSTORE_DIR / f"{name}.faiss"
    meta_path = VECTORSTORE_DIR / f"{name}.pkl"

    if not index_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            f"No vectorstore found at {VECTORSTORE_DIR}. "
            "Please upload and process a PDF first."
        )

    import faiss
    index = faiss.read_index(str(index_path))
    with open(meta_path, "rb") as f:
        chunks = pickle.load(f)

    logger.info("Loaded vectorstore: %d vectors, %d chunks", index.ntotal, len(chunks))
    return index, chunks


# ── High-Level Ingestion Entry Point ──────────────────────────────────────────

def ingest_pdf(pdf_path: Path) -> int:
    """
    Full pipeline: load PDF → split → embed → save vectorstore.

    Args:
        pdf_path: Path to the uploaded PDF.

    Returns:
        Number of chunks indexed.
    """
    text = load_pdf(pdf_path)
    chunks = split_text(text)
    index, chunks = build_vectorstore(chunks)
    save_vectorstore(index, chunks)
    return len(chunks)
