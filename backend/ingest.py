"""
ingest.py
---------
Handles the full document ingestion pipeline using lightweight TF-IDF:
  1. Load PDF from disk
  2. Split text into chunks
  3. Fit a TfidfVectorizer
  4. Save the vectorizer, tfidf_matrix, and chunks to disk
"""

import pickle
from pathlib import Path
from typing import List, Tuple

from config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    VECTORSTORE_DIR,
)
from utils import get_logger

logger = get_logger(__name__)


# ── PDF Loading ────────────────────────────────────────────────────────────────

def load_pdf(pdf_path: Path) -> str:
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    import fitz  # PyMuPDF
    doc = fitz.open(str(pdf_path))
    pages_text: List[str] = []

    for page in doc:
        text = page.get_text() or ""
        if text.strip():
            pages_text.append(text)

    if not pages_text:
        raise ValueError("No extractable text found in PDF.")

    full_text = "\n".join(pages_text)
    logger.info("Extracted %d characters from %d pages", len(full_text), len(pages_text))
    return full_text


# ── Text Splitting ────────────────────────────────────────────────────────────

def split_text(text: str) -> List[str]:
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


# ── TF-IDF Index ───────────────────────────────────────────────────────────────

def build_vectorstore(chunks: List[str]):
    """
    Fit a TF-IDF vectorizer on the chunks.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(chunks)

    logger.info("Built TF-IDF index with %d chunks", len(chunks))
    return vectorizer, tfidf_matrix, chunks


def save_vectorstore(vectorizer, tfidf_matrix, chunks: List[str], name: str = "index") -> None:
    """
    Persist the vectorizer, matrix, and chunks to disk using pickle.
    """
    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    
    data = {
        "vectorizer": vectorizer,
        "tfidf_matrix": tfidf_matrix,
        "chunks": chunks
    }
    
    with open(VECTORSTORE_DIR / f"{name}.pkl", "wb") as f:
        pickle.dump(data, f)
        
    logger.info("Vectorstore saved → %s", VECTORSTORE_DIR / name)


def load_vectorstore(name: str = "index"):
    """
    Load the saved TF-IDF data from disk.
    """
    meta_path = VECTORSTORE_DIR / f"{name}.pkl"

    if not meta_path.exists():
        raise FileNotFoundError(
            f"No vectorstore found at {VECTORSTORE_DIR}. "
            "Please upload and process a PDF first."
        )

    with open(meta_path, "rb") as f:
        data = pickle.load(f)

    logger.info("Loaded vectorstore: %d chunks", len(data["chunks"]))
    return data["vectorizer"], data["tfidf_matrix"], data["chunks"]


# ── High-Level Ingestion Entry Point ──────────────────────────────────────────

def ingest_pdf(pdf_path: Path) -> int:
    def _log(msg):
        with open(VECTORSTORE_DIR.parent / "debug.log", "a", encoding="utf-8") as f:
            f.write(msg + "\n")
            
    _log("1. Loading PDF...")
    text = load_pdf(pdf_path)
    
    _log(f"2. Splitting text (length: {len(text)})...")
    chunks = split_text(text)
    
    _log(f"3. Building TF-IDF vectorstore with {len(chunks)} chunks...")
    vectorizer, tfidf_matrix, chunks = build_vectorstore(chunks)
    
    _log("4. Saving TF-IDF to disk...")
    save_vectorstore(vectorizer, tfidf_matrix, chunks)
    
    _log("5. Done!")
    return len(chunks)
