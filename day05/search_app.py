import gradio as gr, faiss, numpy as np, pickle
from sentence_transformers import SentenceTransformer

model  = SentenceTransformer("all-MiniLM-L6-v2")
index  = faiss.read_index("day05/wiki.faiss")
corpus = pickle.load(open("day05/corpus.pkl", "rb"))

def search(query, k=5):
    q = model.encode([query], normalize_embeddings=True).astype("float32")
    D, I = index.search(q, int(k))   # int(): the slider can send 5.0, FAISS needs 5
    return "\n\n".join([f"**Score {d:.3f}**\n{corpus[i][:400]}..." for d, i in zip(D[0], I[0])])

gr.Interface(
    fn=search,
    inputs=[gr.Textbox(label="Query"), gr.Slider(1, 20, value=5, step=1, label="Top-K")],
    outputs=gr.Markdown(label="Results"),
    title="Day 5 Semantic Search",
).launch()