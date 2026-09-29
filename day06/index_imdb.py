import pandas as pd
from datasets import load_dataset
from elasticsearch import Elasticsearch, helpers
from tqdm import tqdm

es = Elasticsearch("http://localhost:9200")
INDEX = "imdb_reviews"

if es.indices.exists(index=INDEX):
    es.indices.delete(index=INDEX)

es.indices.create(index=INDEX, mappings={
    "properties": {
        "text":      {"type": "text",    "analyzer": "english"},
        "sentiment": {"type": "keyword"},
        "length":    {"type": "integer"},
    }
})

# CHANGED: the CSV isn't on this laptop, so load the same 25k reviews from Hugging Face
df = load_dataset("stanfordnlp/imdb", split="train").to_pandas()
df["length"] = df["text"].str.len()
df["sentiment"] = df["label"].map({0: "negative", 1: "positive"})

actions = (
    {"_index": INDEX, "_id": i,
     "_source": {"text": row.text, "sentiment": row.sentiment, "length": int(row.length)}}
    for i, row in enumerate(tqdm(df.itertuples(index=False), total=len(df)))
)
helpers.bulk(es, actions, chunk_size=500)
es.indices.refresh(index=INDEX)
print("Docs indexed:", es.count(index=INDEX)["count"])