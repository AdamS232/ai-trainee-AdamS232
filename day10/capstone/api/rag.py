"""RAG pipeline: PDF -> chunks -> embeddings -> Chroma -> re-rank -> Ollama."""
import json
import logging
from pathlib import Path
from typing import Iterator

import chromadb
import requests
from chromadb.config import Settings as ChromaSettings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from sentence_transformers import CrossEncoder, SentenceTransformer

from day10.capstone.api.config import settings

log = logging.getLogger("askthepaper")
IDK = "I don't know based on the provided documents."
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "  # bge-v1.5 query instruction
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)  # silence telemetry noise

# Loaded once when the module is imported (slow to load, fast to reuse)
_embedder = SentenceTransformer(settings.embed_model)
_reranker = CrossEncoder(settings.rerank_model)
_client = chromadb.PersistentClient(path=settings.chroma_dir,
                                    settings=ChromaSettings(anonymized_telemetry=False))
_col = _client.get_or_create_collection("askthepaper", metadata={"hnsw:space": "cosine"})
_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)

PROMPT = """You are AskThePaper, a precise assistant that answers questions about the user's documents.
Use ONLY the numbered sources below. Cite the sources you use inline, like [S1] or [S2].
If the sources do not contain the answer, reply exactly: "{idk}"

Sources:
{context}

Question: {question}
Answer:"""


def index_pdf(path: str) -> int:
    """Chunk a PDF page by page, embed the chunks, and store them in Chroma. Returns the chunk count."""
    name = Path(path).name
    _col.delete(where={"source": name})          # re-uploading a file replaces its old chunks
    ids, docs, metas = [], [], []
    for page_no, page in enumerate(PdfReader(path).pages, start=1):
        text = page.extract_text() or ""
        for i, chunk in enumerate(_splitter.split_text(text)):
            ids.append(f"{name}::p{page_no}::c{i}")
            docs.append(chunk)
            metas.append({"source": name, "page": page_no})
    if not docs:
        raise ValueError(f"No extractable text in {name} (scanned PDF?)")
    embs = _embedder.encode(docs, normalize_embeddings=True, batch_size=64).tolist()
    _col.upsert(ids=ids, documents=docs, embeddings=embs, metadatas=metas)
    log.info("Indexed %s: %d chunks", name, len(docs))
    return len(docs)


def list_documents() -> list[str]:
    """Names of all indexed PDFs."""
    metas = _col.get(include=["metadatas"])["metadatas"]
    return sorted({m["source"] for m in metas})


def retrieve(question: str, k: int | None = None) -> list[dict]:
    """Vector search: top-k chunks closest in meaning to the question."""
    if _col.count() == 0:
        return []
    k = min(k or settings.retrieve_k, _col.count())
    q = _embedder.encode([QUERY_PREFIX + question], normalize_embeddings=True).tolist()
    r = _col.query(query_embeddings=q, n_results=k)
    return [{"text": d, "source": m["source"], "page": m["page"], "distance": dist}
            for d, m, dist in zip(r["documents"][0], r["metadatas"][0], r["distances"][0])]


def rerank(question: str, hits: list[dict], n: int | None = None) -> list[dict]:
    """Cross-encoder re-ranking: score each (question, chunk) pair together, keep the best n."""
    if not hits:
        return []
    scores = _reranker.predict([(question, h["text"]) for h in hits])
    for h, s in zip(hits, scores):
        h["score"] = float(s)
    return sorted(hits, key=lambda h: -h["score"])[: n or settings.top_n]


def search(question: str) -> list[dict]:
    """Full retrieval: vector search, then re-rank."""
    return rerank(question, retrieve(question))


def build_prompt(question: str, hits: list[dict]) -> str:
    """Number the sources [S1], [S2]... and fill in the prompt template."""
    context = "\n\n".join(f"[S{i}] ({h['source']}, p.{h['page']})\n{h['text']}"
                          for i, h in enumerate(hits, start=1))
    return PROMPT.format(idk=IDK, context=context, question=question)


def stream_answer(question: str, hits: list[dict]) -> Iterator[str]:
    """Yield the answer token by token from Ollama. Refuses without calling the LLM if nothing relevant was found."""
    if not hits or hits[0]["score"] < settings.min_rerank_score:
        log.info("Refused (best score %s): %s", hits[0]["score"] if hits else None, question)
        yield IDK
        return
    payload = {"model": settings.llm_model, "prompt": build_prompt(question, hits),
               "stream": True, "options": {"temperature": 0.1}}
    with requests.post(f"{settings.ollama_url}/api/generate", json=payload,
                       stream=True, timeout=180) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line:
                continue
            data = json.loads(line)
            if data.get("response"):
                yield data["response"]
            if data.get("done"):
                break


def answer(question: str) -> dict:
    """Non-streaming helper (used by tests/eval): returns the answer text and its sources."""
    hits = search(question)
    text = "".join(stream_answer(question, hits))
    return {"answer": text, "sources": hits}


if __name__ == "__main__":   # quick self-test
    logging.basicConfig(level=logging.INFO)
    print("chunks:", index_pdf("data/raw/my_document.pdf"))
    out = answer("What two RAG formulations does the paper compare?")
    print(out["answer"])
    for s in out["sources"]:
        print(f"  [{s['score']:.2f}] {s['source']} p.{s['page']}")