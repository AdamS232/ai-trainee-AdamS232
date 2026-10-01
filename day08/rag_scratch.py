import re, requests, chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder

CE = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
EMB = SentenceTransformer("BAAI/bge-small-en-v1.5")
client = chromadb.PersistentClient(path="day08/chroma")
col = client.get_or_create_collection("kb_docs")
 
def load_pdf(path: str) -> str:
    return "\n".join(p.extract_text() or "" for p in PdfReader(path).pages)
 
def chunk(text: str, size: int = 800, overlap: int = 100):
    paras = re.split(r"\n+", text)
    chunks, buf = [], ""
    for p in paras:
        if len(buf) + len(p) < size:
            buf += ("\n\n" + p) if buf else p
        else:
            chunks.append(buf)
            buf = buf[-overlap:] + "\n\n" + p
    if buf: chunks.append(buf)
    return [c.strip() for c in chunks if c.strip()]
 
def index_pdf(path: str):
    text   = load_pdf(path)
    chunks = chunk(text)
    embs   = EMB.encode(chunks, normalize_embeddings=True).tolist()
    ids    = [f"{path}::{i}" for i in range(len(chunks))]
    col.upsert(ids=ids, documents=chunks, embeddings=embs,
            metadatas=[{"source": path, "chunk": i} for i in range(len(chunks))])
 
def retrieve(query: str, k: int = 5):
    q = EMB.encode([query], normalize_embeddings=True).tolist()
    r = col.query(query_embeddings=q, n_results=k)
    return list(zip(r["documents"][0], r["metadatas"][0], r["distances"][0]))

def rerank(query: str, hits, top_k: int = 5):
    pairs  = [(query, doc) for doc, m, _ in hits]
    scores = CE.predict(pairs)
    ranked = sorted(zip(hits, scores), key=lambda x: -x[1])
    return [h for h, s in ranked[:top_k]]
 
PROMPT = """You are a precise assistant. Answer the user's question using ONLY the context below.
If the answer is not in the context, say: "I don't know based on the provided documents."
Cite sources as [source_name chunk_id] inline.
 
Context:
{context}
 
Question: {question}
 
Answer:"""
 
def ask(question: str):
    hits = rerank(question, retrieve(question, k=50), top_k=5)
    context = "\n\n---\n\n".join(
        f"[{m['source']} chunk {m['chunk']}]\n{doc}" for doc, m, _ in hits
    )
    prompt = PROMPT.format(context=context, question=question)
    r = requests.post("http://localhost:11434/api/generate",
                      json={"model": "llama3.2:3b", "prompt": prompt, "stream": False})
    return r.json()["response"]
 
if __name__ == "__main__":
    index_pdf("data/raw/my_document.pdf")
    print(ask("What two RAG formulations does the paper compare?"))