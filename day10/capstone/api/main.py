"""FastAPI service for AskThePaper: /health, /ingest, /search, /chat (streaming)."""
import json
import logging
import shutil
from pathlib import Path

import requests
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from day10.capstone.api import rag
from day10.capstone.api.config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("askthepaper")

app = FastAPI(title="AskThePaper API", version="1.0")


class Question(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)


class Source(BaseModel):
    source: str
    page: int
    score: float
    text: str


class SearchResponse(BaseModel):
    sources: list[Source]


@app.get("/health")
def health() -> dict:
    """Liveness check: is the API up, is Ollama reachable, which PDFs are indexed."""
    try:
        ollama_ok = requests.get(f"{settings.ollama_url}/api/tags", timeout=3).ok
    except requests.RequestException:
        ollama_ok = False
    return {"status": "ok", "ollama": ollama_ok, "documents": rag.list_documents()}


@app.post("/ingest")
def ingest(file: UploadFile = File(...)) -> dict:
    """Save an uploaded PDF and index it."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported.")
    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    path = upload_dir / Path(file.filename).name      # drop any folder parts from the name
    with path.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    try:
        n = rag.index_pdf(str(path))
    except ValueError as e:                            # e.g. scanned PDF with no text
        raise HTTPException(422, str(e))
    except Exception:
        log.exception("Indexing failed for %s", path.name)
        raise HTTPException(500, "Indexing failed. Check the API logs.")
    return {"indexed": path.name, "chunks": n}


@app.post("/search", response_model=SearchResponse)
def search(q: Question) -> dict:
    """Retrieve + re-rank only (no LLM). Useful for debugging and evaluation."""
    return {"sources": rag.search(q.question)}


@app.post("/chat")
def chat(q: Question) -> StreamingResponse:
    """RAG answer, streamed as NDJSON: first line = sources, then one line per token."""
    hits = rag.search(q.question)
    refused = not hits or hits[0]["score"] < settings.min_rerank_score

    def stream():
        sources = [] if refused else [
            {"source": h["source"], "page": h["page"], "score": round(h["score"], 2), "text": h["text"]}
            for h in hits]
        yield json.dumps({"sources": sources}) + "\n"
        try:
            for token in rag.stream_answer(q.question, hits):
                yield json.dumps({"token": token}) + "\n"
        except requests.RequestException:
            log.exception("Ollama call failed")
            yield json.dumps({"error": "The language model is unavailable. Is Ollama running?"}) + "\n"

    return StreamingResponse(stream(), media_type="application/x-ndjson")