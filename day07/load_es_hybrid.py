"""Load the same 5,000 IMDB reviews into Elasticsearch with text + embeddings (for hybrid search)."""
from elasticsearch import Elasticsearch, helpers
from sentence_transformers import SentenceTransformer
from datasets import load_dataset

df = load_dataset("stanfordnlp/imdb", split="train").shuffle(seed=42).select(range(5000)).to_pandas()
docs = df["text"].tolist()
sents = ["positive" if l == 1 else "negative" for l in df["label"]]
vecs = SentenceTransformer("all-MiniLM-L6-v2").encode(docs, normalize_embeddings=True, show_progress_bar=True)

es = Elasticsearch("http://localhost:9200")
if es.indices.exists(index="imdb_hybrid"):
    es.indices.delete(index="imdb_hybrid")

es.indices.create(index="imdb_hybrid", mappings={"properties": {
    "text":      {"type": "text", "analyzer": "english"},
    "sentiment": {"type": "keyword"},
    "embedding": {"type": "dense_vector", "dims": 384, "index": True, "similarity": "cosine"},
}})

helpers.bulk(es, ({"_index": "imdb_hybrid", "_id": i,
                   "_source": {"text": d, "sentiment": s, "embedding": v.tolist()}}
                  for i, (d, v, s) in enumerate(zip(docs, vecs, sents))), chunk_size=500)
es.indices.refresh(index="imdb_hybrid")
print("ES docs:", es.count(index="imdb_hybrid")["count"])