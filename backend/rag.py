"""
rag.py
------
Core RAG logic:
  1. Retrieve semantically relevant chunks from the FAISS index
  2. Build a context-grounded prompt
  3. Call Ollama LLM and return the answer
"""

from typing import List, Tuple
from config import TOP_K
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

import os
from groq import Groq

# Initialize the Groq client (requires GROQ_API_KEY environment variable)
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def ask_llm(prompt: str) -> str:
    chat = client.chat.completions.create(
        messages=[
            {"role": "user", "content": prompt}
        ],
        model="llama-3.1-8b-instant"
    )
    return chat.choices[0].message.content


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
