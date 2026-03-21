# RAG Application

A production-ready **Retrieval-Augmented Generation (RAG)** app built with:

| Layer | Technology |
|---|---|
| Backend | FastAPI + Uvicorn |
| Frontend | Streamlit |
| LLM | Ollama (llama3) |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) |
| Vector Store | FAISS (local, persistent) |

---

## Project Structure

```
rag-app/
├── backend/
│   ├── main.py          # FastAPI routes (no business logic)
│   ├── rag.py           # Retrieval, prompt building, LLM call
│   ├── ingest.py        # PDF load, chunk, embed, FAISS index
│   ├── config.py        # Env-driven settings
│   ├── utils.py         # Logging + directory helpers
│   └── requirements.txt
├── frontend/
│   └── app.py           # Streamlit chat UI
├── data/                # Uploaded PDFs land here
├── vectorstore/         # FAISS index persisted here
└── .env.example         # Copy → .env before running
```

---

## Prerequisites

1. **Python 3.10+**
2. **Ollama** — [install](https://ollama.com/download) and pull the model:
   ```bash
   ollama pull llama3
   ollama serve          # keep this running in the background
   ```
3. **Streamlit** (frontend only):
   ```bash
   pip install streamlit
   ```

---

## Quick Start

### 1 — Clone & configure
```powershell
cd rag-app
copy .env.example .env   # Windows
# Edit .env if you need non-default values (e.g. different model or ports)
```

### 2 — Install backend dependencies
```powershell
cd backend
pip install -r requirements.txt
```

### 3 — Start the backend
```powershell
cd backend
uvicorn main:app --reload --port 8000
```
API docs available at **http://localhost:8000/docs**

### 4 — Start the frontend (new terminal)
```powershell
cd frontend
streamlit run app.py
```
Opens at **http://localhost:8501**

---

## How to Use

1. Open the Streamlit app in your browser.
2. In the **sidebar**, upload a PDF and click **Upload & Process**.
3. Type your question in the chat input and click **Ask ➤**.
4. The assistant answers using only the document's content.

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/health` | Liveness check |
| `POST` | `/upload` | Upload + index a PDF |
| `POST` | `/ask`    | Ask a question (RAG) |

### POST `/upload`
- **Content-Type**: `multipart/form-data`
- **Field**: `file` (PDF)
- **Response**: `{ message, filename, chunks_indexed }`

### POST `/ask`
- **Body**: `{ "question": "string" }`
- **Response**: `{ "answer": "string", "sources": ["..."] }`

---

## Configuration (`.env`)

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `LLM_MODEL` | `llama3` | Ollama model tag |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | HuggingFace encoder |
| `CHUNK_SIZE` | `500` | Chars per text chunk |
| `CHUNK_OVERLAP` | `50` | Overlap between chunks |
| `TOP_K` | `3` | Chunks retrieved per query |
| `DATA_DIR` | `data` | PDF storage directory |
| `VECTORSTORE_DIR` | `vectorstore` | FAISS index directory |
| `CORS_ORIGINS` | `http://localhost:8501` | Allowed CORS origins |

---

## Deployment Notes

- The vectorstore is file-system based — swap `faiss-cpu` for `pinecone-client` or `weaviate-client` for cloud deployments.
- Set `OLLAMA_BASE_URL` to your remote Ollama endpoint for cloud inference.
- The `.env` file is gitignored — use environment secrets on AWS / Render.

---

## License

MIT
