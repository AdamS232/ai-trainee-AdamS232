"""Load 5,000 IMDB reviews into ChromaDB (embeds automatically)."""
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from datasets import load_dataset

df = load_dataset("stanfordnlp/imdb", split="train").shuffle(seed=42).select(range(5000)).to_pandas()

client = chromadb.PersistentClient(path="day07/chroma_store",
                                   settings=Settings(anonymized_telemetry=False))
ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
col = client.get_or_create_collection("imdb", embedding_function=ef)

col.upsert(
    ids=[f"doc-{i}" for i in range(len(df))],
    documents=df["text"].tolist(),
    metadatas=[{"sentiment": "positive" if l == 1 else "negative"} for l in df["label"]],
)
print("Chroma docs:", col.count())