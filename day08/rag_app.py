import glob, os
import gradio as gr
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from sentence_transformers import CrossEncoder

# 1. Load every PDF in data/raw
docs = []
for path in glob.glob("data/raw/*.pdf"):
    docs.extend(PyPDFLoader(path).load())

# 2. Chunk, embed, store (in memory - rebuilt each time the app starts)
chunks = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100).split_documents(docs)
emb  = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
db   = Chroma.from_documents(chunks, emb, collection_name="rag_app")
retr = db.as_retriever(search_kwargs={"k": 20})

llm = ChatOllama(model="llama3.2:3b", temperature=0)
ce  = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

PROMPT = """You are a precise assistant. Answer the question using ONLY the context below.
If the answer is not in the context, say: "I don't know based on the provided documents."
Cite sources as [file p.N] inline.

Context:
{context}

Question: {question}

Answer:"""

def label(d):
    return f"{os.path.basename(d.metadata['source'])} p.{d.metadata['page'] + 1}"

def rerank(query, docs, top_k=3):
    scores = ce.predict([(query, d.page_content) for d in docs])
    ranked = sorted(zip(docs, scores), key=lambda x: -x[1])
    return ranked[:top_k]

def answer(question, history):
    hits     = retr.invoke(question)                 # top 20 (fast bi-encoder)
    reranked = rerank(question, hits, top_k=3)       # best 3 (cross-encoder)
    ctx = "\n\n---\n\n".join(f"[{label(d)}]\n{d.page_content}" for d, s in reranked)
    reply = llm.invoke(PROMPT.format(context=ctx, question=question)).content
    sources = "\n\n---\n\n".join(
        f"**Score {s:.2f}** - {label(d)}\n\n{d.page_content[:300]}..." for d, s in reranked
    )
    return reply + "\n\n<details><summary>Sources</summary>\n\n" + sources + "\n\n</details>"

gr.ChatInterface(fn=answer, title="Day 8 - PDF RAG").launch()