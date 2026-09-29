"""Run the same queries through BM25 (Elasticsearch) and semantic search (FAISS)."""
import pickle
import faiss
from elasticsearch import Elasticsearch, helpers
from sentence_transformers import SentenceTransformer

# Load the Day 5 corpus + FAISS index
corpus = pickle.load(open("day05/corpus.pkl", "rb"))
faiss_index = faiss.read_index("day05/wiki.faiss")
model = SentenceTransformer("all-MiniLM-L6-v2")

# Put the SAME 20,000 chunks into Elasticsearch, so both engines search identical data
es = Elasticsearch("http://localhost:9200")
ES_INDEX = "wiki"
title_of = lambda chunk: chunk.split(": ", 1)[0]

if not es.indices.exists(index=ES_INDEX) or es.count(index=ES_INDEX)["count"] != len(corpus):
    if es.indices.exists(index=ES_INDEX):
        es.indices.delete(index=ES_INDEX)
    es.indices.create(index=ES_INDEX, mappings={"properties": {
        "title": {"type": "keyword"},
        "text":  {"type": "text", "analyzer": "english"},
    }})
    helpers.bulk(es, ({"_index": ES_INDEX, "_id": i,
                       "_source": {"title": title_of(c), "text": c}}
                      for i, c in enumerate(corpus)), chunk_size=1000)
    es.indices.refresh(index=ES_INDEX)
print("ES docs:", es.count(index=ES_INDEX)["count"])

def bm25(q):
    r = es.search(index=ES_INDEX, query={"match": {"text": q}}, size=5)
    return [h["_source"]["title"] for h in r["hits"]["hits"]]

def semantic(q):
    qv = model.encode([q], normalize_embeddings=True).astype("float32")
    _, I = faiss_index.search(qv, 5)
    return [title_of(corpus[i]) for i in I[0]]

# Your 10 Day 5 test queries
tests = [
    ("the fourth month of the year",                               "April"),
    ("month named after the Roman emperor Augustus",               "August"),
    ("painting, sculpture and other creative things people make",  "Art"),
    ("the mixture of gases that we breathe",                       "Air"),
    ("British mathematician who broke German codes in World War 2","Alan Turing"),
    ("Canadian singer who made the album Jagged Little Pill",      "Alanis Morissette"),
    ("computer program for drawing vector graphics",               "Adobe Illustrator"),
    ("spicy smoked sausage from France and Louisiana",             "Andouille"),
    ("growing crops and raising animals for food",                 "Farming"),
    ("adding, subtracting, multiplying and dividing numbers",      "Arithmetic"),
]

score = {"BM25": 0, "Semantic": 0}
for q, want in tests:
    print(f"\nQUERY: {q}   (want: {want})")
    for name, fn in [("BM25", bm25), ("Semantic", semantic)]:
        titles = fn(q)
        hit = want in titles
        score[name] += hit
        print(f"  {name:<8} {'HIT ' if hit else 'MISS'}  {titles}")

print(f"\nRecall@5 — BM25: {score['BM25']}/10   Semantic: {score['Semantic']}/10")