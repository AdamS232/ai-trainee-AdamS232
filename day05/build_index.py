"""Build a FAISS index over chunked Simple English Wikipedia."""
import pickle, time
import numpy as np, faiss
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

N_CHUNKS = 20000
CHUNK_WORDS = 150   # ~200 tokens, safely under MiniLM's 256-token limit

# 1. Stream articles (downloads only what we read, not all of Wikipedia)
ds = load_dataset("wikimedia/wikipedia", "20231101.simple", split="train", streaming=True)

# 2. Chunk each article into ~150-word pieces
corpus = []
for article in ds:
    words = article["text"].split()
    for start in range(0, len(words), CHUNK_WORDS):
        piece = " ".join(words[start:start + CHUNK_WORDS])
        if len(piece.split()) >= 20:          # skip tiny fragments
            corpus.append(f"{article['title']}: {piece}")
    if len(corpus) >= N_CHUNKS:
        break
corpus = corpus[:N_CHUNKS]
print(f"Collected {len(corpus)} chunks")

# 3. Embed
model = SentenceTransformer("all-MiniLM-L6-v2")
t = time.perf_counter()
emb = model.encode(corpus, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
emb = np.array(emb, dtype="float32")
print(f"Embedded in {time.perf_counter() - t:.0f} s, shape {emb.shape}")

# 4. Index + save
index = faiss.IndexFlatIP(emb.shape[1])
index.add(emb)
faiss.write_index(index, "day05/wiki.faiss")
with open("day05/corpus.pkl", "wb") as f:
    pickle.dump(corpus, f)
print("Saved day05/wiki.faiss and day05/corpus.pkl with", index.ntotal, "vectors")