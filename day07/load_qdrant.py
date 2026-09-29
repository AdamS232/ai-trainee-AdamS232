"""Load the same 5,000 IMDB reviews into Qdrant (we embed them ourselves)."""
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from sentence_transformers import SentenceTransformer
from datasets import load_dataset

df = load_dataset("stanfordnlp/imdb", split="train").shuffle(seed=42).select(range(5000)).to_pandas()
docs = df["text"].tolist()
sents = ["positive" if l == 1 else "negative" for l in df["label"]]
vecs = SentenceTransformer("all-MiniLM-L6-v2").encode(docs, normalize_embeddings=True, show_progress_bar=True)

qc = QdrantClient("http://localhost:6333")
qc.recreate_collection(
    collection_name="imdb",
    vectors_config=qm.VectorParams(size=384, distance=qm.Distance.COSINE),
)
qc.upload_points(
    collection_name="imdb",
    points=[qm.PointStruct(id=i, vector=v.tolist(), payload={"text": d, "sentiment": s})
            for i, (d, v, s) in enumerate(zip(docs, vecs, sents))],
    batch_size=256,
)
print("Qdrant docs:", qc.count("imdb").count)