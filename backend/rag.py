"""
rag.py
------
Core RAG logic using TF-IDF:
  1. Retrieve relevant chunks using cosine similarity
  2. Build a context-grounded prompt
  3. Call Groq LLM and return the answer
"""

from typing import List, Tuple
from config import TOP_K
from ingest import load_vectorstore
from utils import get_logger
import os
from groq import Groq

logger = get_logger(__name__)

# Initialize the Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ── Retrieval ─────────────────────────────────────────────────────────────────

def retrieve(query: str, top_k: int = TOP_K) -> Tuple[List[str], List[float]]:
    """
    Find the most relevant chunks using TF-IDF cosine similarity.
    """
    vectorizer, tfidf_matrix, chunks = load_vectorstore()
    
    from sklearn.metrics.pairwise import cosine_similarity
    
    query_vec = vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    
    # Get indices of top_k highest similarities
    indices = similarities.argsort()[-top_k:][::-1]
    
    retrieved = [chunks[i] for i in indices]
    scores = [float(similarities[i]) for i in indices]

    logger.info("Retrieved %d chunks for query: %.60s...", len(retrieved), query)
    return retrieved, scores


# ── Prompt Construction ────────────────────────────────────────────────────────

def build_prompt(query: str, context_chunks: List[str]) -> str:
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
    chat = client.chat.completions.create(
        messages=[
            {"role": "user", "content": prompt}
        ],
        model="llama-3.1-8b-instant"
    )
    return chat.choices[0].message.content


# ── Orchestration ─────────────────────────────────────────────────────────────

def answer_question(query: str) -> dict:
    chunks, scores = retrieve(query)
    prompt = build_prompt(query, chunks)
    llm_answer = ask_llm(prompt)

    sources = [c[:200].strip() + ("…" if len(c) > 200 else "") for c in chunks]

    return {"answer": llm_answer, "sources": sources}
