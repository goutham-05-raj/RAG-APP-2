"""
rag.py
------
Core RAG logic:
  1. Retrieve semantically relevant chunks from the FAISS index
  2. Build a context-grounded prompt
  3. Call Ollama LLM and return the answer
"""

from typing import List, Tuple

import numpy as np
import requests

from config import OLLAMA_BASE_URL, LLM_MODEL, TOP_K
from ingest import load_vectorstore, _get_embedder
from utils import get_logger

logger = get_logger(__name__)

# ── Retrieval ─────────────────────────────────────────────────────────────────

def retrieve(query: str, top_k: int = TOP_K) -> Tuple[List[str], List[float]]:
    """
    Embed the query and return the top_k most relevant chunks.

    Args:
        query:  User question string.
        top_k:  Number of chunks to retrieve.

    Returns:
        Tuple of (list_of_chunks, list_of_distances).

    Raises:
        FileNotFoundError: Propagated from load_vectorstore when no index exists.
    """
    index, chunks = load_vectorstore()
    embedder = _get_embedder()

    query_vec = embedder.encode([query], convert_to_numpy=True).astype("float32")
    distances, indices = index.search(query_vec, top_k)

    retrieved = [chunks[i] for i in indices[0] if i < len(chunks)]
    dists = distances[0].tolist()

    logger.info("Retrieved %d chunks for query: %.60s...", len(retrieved), query)
    return retrieved, dists


# ── Prompt Construction ────────────────────────────────────────────────────────

def build_prompt(query: str, context_chunks: List[str]) -> str:
    """
    Construct a RAG prompt that instructs the LLM to answer strictly
    from the provided context, avoiding hallucinations.

    Args:
        query:          User's question.
        context_chunks: Relevant text passages retrieved from the document.

    Returns:
        Formatted prompt string ready to send to the LLM.
    """
    context = "\n\n---\n\n".join(context_chunks)
    prompt = (
        "You are a precise and factual assistant. "
        "Answer the question ONLY from the provided context below. "
        "If the answer is not found in the context, respond with exactly: "
        "'I don't know based on the provided document.'\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        "Answer:"
    )
    return prompt


# ── LLM Interaction ───────────────────────────────────────────────────────────

def ask_llm(prompt: str) -> str:
    """
    Send a prompt to Ollama and return the generated answer.

    Args:
        prompt: Full prompt string including context and question.

    Returns:
        LLM-generated answer string.

    Raises:
        ConnectionError: If Ollama is unreachable.
        RuntimeError:    If the API returns an unexpected response.
    """
    url = f"{OLLAMA_BASE_URL}/api/generate"
    payload = {
        "model": LLM_MODEL,
        "prompt": prompt,
        "stream": False,
    }

    try:
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
    except requests.exceptions.ConnectionError as exc:
        raise ConnectionError(
            f"Cannot reach Ollama at {OLLAMA_BASE_URL}. "
            "Make sure Ollama is running (`ollama serve`)."
        ) from exc
    except requests.exceptions.HTTPError as exc:
        raise RuntimeError(f"Ollama API error: {exc}") from exc

    data = response.json()
    answer = data.get("response", "").strip()

    if not answer:
        raise RuntimeError("Ollama returned an empty response.")

    logger.info("LLM answered %d characters", len(answer))
    return answer


# ── Orchestration ─────────────────────────────────────────────────────────────

def answer_question(query: str) -> dict:
    """
    Full RAG pipeline: retrieve → build prompt → ask LLM.

    Args:
        query: User's question.

    Returns:
        Dict with keys:
          - "answer"  (str)  : LLM's response
          - "sources" (list) : Top-k chunk excerpts used as context
    """
    chunks, distances = retrieve(query)
    prompt = build_prompt(query, chunks)
    llm_answer = ask_llm(prompt)

    # Return short excerpts (first 200 chars) as source evidence
    sources = [c[:200].strip() + ("…" if len(c) > 200 else "") for c in chunks]

    return {"answer": llm_answer, "sources": sources}
