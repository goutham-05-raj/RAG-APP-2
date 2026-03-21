"""
main.py
-------
FastAPI application entry point.
All business logic is delegated to ingest.py and rag.py — this file
contains only routing, middleware, and request/response modelling.
"""

import shutil
import sys
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Ensure the backend directory is on sys.path when launched from any CWD
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import CORS_ORIGINS, DATA_DIR
from ingest import ingest_pdf
from rag import answer_question
from utils import ensure_dirs, get_logger

logger = get_logger(__name__)

# ── App Bootstrap ─────────────────────────────────────────────────────────────
ensure_dirs()

app = FastAPI(
    title="RAG API",
    description="Retrieval-Augmented Generation backend powered by Ollama + FAISS.",
    version="1.0.0",
)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response Models ─────────────────────────────────────────────────

class QuestionRequest(BaseModel):
    """Payload for the /ask endpoint."""
    question: str = Field(..., min_length=1, description="The question to ask the RAG system.")


class UploadResponse(BaseModel):
    """Response returned after a successful PDF upload and indexing."""
    message: str
    filename: str
    chunks_indexed: int


class AnswerResponse(BaseModel):
    """Response returned by the /ask endpoint."""
    answer: str
    sources: list[str]


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/")
def home():
    return {"status": "running 🚀"}

@app.get("/health", tags=["Utility"])
async def health_check() -> dict:
    """Lightweight liveness check."""
    return {"status": "ok"}

@app.get("/routes", tags=["Utility"])
def get_routes():
    return [route.path for route in app.routes]


@app.post("/upload", response_model=UploadResponse, tags=["Ingestion"])
async def upload_pdf(file: UploadFile = File(...)) -> UploadResponse:
    """
    Accept a PDF file, save it to disk, extract text, generate embeddings,
    and persist the FAISS index.

    Args:
        file: Uploaded PDF (multipart/form-data).

    Returns:
        UploadResponse with filename and number of indexed chunks.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    save_path = DATA_DIR / file.filename
    try:
        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info("Saved uploaded file → %s", save_path)
    finally:
        await file.close()

    try:
        chunk_count = ingest_pdf(save_path)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Ingestion failed for %s", file.filename)
        raise HTTPException(status_code=500, detail=f"Ingestion error: {exc}") from exc

    return UploadResponse(
        message="PDF processed and indexed successfully.",
        filename=file.filename,
        chunks_indexed=chunk_count,
    )


@app.post("/ask", response_model=AnswerResponse, tags=["RAG"])
async def ask(request: QuestionRequest) -> AnswerResponse:
    """
    Answer a question using the RAG pipeline (retrieve → prompt → LLM).

    Args:
        request: JSON body containing the question string.

    Returns:
        AnswerResponse with the LLM answer and source excerpts.
    """
    try:
        result = answer_question(request.question)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ConnectionError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Error answering question: %s", request.question)
        raise HTTPException(status_code=500, detail=f"RAG error: {exc}") from exc

    return AnswerResponse(**result)
