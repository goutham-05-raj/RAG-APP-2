# CloudRag / Fallstaff Intelligence Search

🚀 **Live Demo:** [https://rag-app-2.vercel.app/](https://rag-app-2.vercel.app/)

A production-ready **Retrieval-Augmented Generation (RAG)** app offering both a standalone Streamlit application and a scalable FastAPI backend architecture. The project has evolved to utilize **Groq's fast LLM APIs** (`llama-3.1-8b-instant`) for inference.

## Architectures

This repository contains two distinct ways to run the application:

### 1. Standalone Streamlit App
A simple, all-in-one frontend and backend using LangChain, HuggingFace embeddings (`all-MiniLM-L6-v2`), FAISS vector store, and Groq LLM.
- **File:** `streamlit_app.py`
- **Features:** Direct PDF upload, chunking, and interactive chat in a single Streamlit interface.

### 2. Client-Server Architecture (Fallstaff Edition)
A robust, decoupled setup using a FastAPI backend and a premium Streamlit frontend.
- **Backend (`backend/`):** FastAPI + Uvicorn. Uses TF-IDF for fast, lightweight document retrieval and Groq API for LLM inference.
- **Frontend (`frontend/app.py`):** A beautifully designed "Fallstaff Edition" Streamlit UI that communicates with the backend via REST API.

---

## Prerequisites

1. **Python 3.10+**
2. A **Groq API Key** — [Get one here](https://console.groq.com/).

---

## Quick Start

### Option 1: Standalone App
```powershell
# Install root dependencies
pip install -r requirements.txt

# Set your Groq API Key
set GROQ_API_KEY=your_api_key_here  # Windows
# export GROQ_API_KEY="your_api_key_here"  # Mac/Linux

# Run the app
streamlit run streamlit_app.py
```

### Option 2: Client-Server Architecture

**1. Start the Backend:**
```powershell
cd backend
pip install -r requirements.txt

# The backend relies on environment variables, set them up:
# Make sure to add your GROQ_API_KEY inside your environment variables or .env file.

uvicorn main:app --reload --port 8000
```
*API docs available at **http://localhost:8000/docs***

**2. Start the Frontend:**
Open a new terminal:
```powershell
cd frontend
# (Requires Streamlit and Requests)
streamlit run app.py
```
*Opens at **http://localhost:8501***

---

## API Reference (FastAPI Backend)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/health` | Liveness check |
| `POST` | `/upload` | Upload + index a PDF using TF-IDF |
| `POST` | `/ask`    | Ask a question (RAG with Groq) |

---

## Deployment Notes

- This project includes a `render.yaml` file for easy cloud deployment of the backend on [Render](https://render.com/). 
- Set the `GROQ_API_KEY` as an environment secret in your deployment environment.

---

## Local Testing
A CLI testing script (`app.py`) is also available in the root directory if you wish to run a quick terminal-based query against a specific PDF file using FAISS and LangChain.

## License

MIT
