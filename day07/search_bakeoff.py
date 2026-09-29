"""BM25 vs semantic vs hybrid (RRF) on the same 5,000 IMDB reviews in Elasticsearch."""
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer

es = Elasticsearch("http://localhost:9200")
model = SentenceTransformer("all-MiniLM-L6-v2")
INDEX = "imdb_hybrid"

# ---------- the three search functions: (query, k) -> ranked list of doc IDs ----------
def search_bm25(query, k=10):
    r = es.search(index=INDEX, query={"match": {"text": query}}, size=k, source=False)
    return [h["_id"] for h in r["hits"]["hits"]]

def search_semantic(query, k=10):
    v = model.encode([query], normalize_embeddings=True)[0].tolist()
    r = es.search(index=INDEX, knn={"field": "embedding", "query_vector": v,
                                    "k": k, "num_candidates": max(100, k * 10)},
                  size=k, source=False)
    return [h["_id"] for h in r["hits"]["hits"]]

def rrf(result_lists, k=60, top_n=10):
    scores = {}
    for ranks in result_lists:
        for rank, doc_id in enumerate(ranks, start=1):
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank)
    return [d for d, _ in sorted(scores.items(), key=lambda x: -x[1])[:top_n]]

def search_hybrid(query, k=10):
    return rrf([search_bm25(query, 50), search_semantic(query, 50)], top_n=k)

# ---------- 20 hand-crafted queries: (type, query, relevant doc ID) ----------
tests = [
    # exact keywords: rare names and titles
    ("keyword",    "Pamela Anderson slasher",                                   "3883"),
    ("keyword",    "Sondra Locke Clint Eastwood",                               "3586"),
    ("keyword",    "Andreas Schnaas zombies",                                   "860"),
    ("keyword",    "Boringus Maximus",                                          "712"),
    ("keyword",    "Richard Brooks The Professionals",                          "467"),
    ("keyword",    "Smallville Tom Welling Michael Rosenbaum",                  "906"),
    ("keyword",    "Josie Lawrence Whose Line",                                 "1220"),
    # paraphrases: describe the review without using its words
    ("paraphrase", "actress faking an eastern European accent opposite a timid office worker", "3187"),
    ("paraphrase", "obsolete soldier abandoned on a distant world after being replaced by engineered troops", "375"),
    ("paraphrase", "female pilot like a famous missing aviator who has never fallen in love", "2100"),
    ("paraphrase", "reminds me of campus antiwar demonstrations in the sixties", "1474"),
    ("paraphrase", "cartoon series with an adorable small mechanical character", "477"),
    ("paraphrase", "aging Scottish star looks lost in his role while his costars shine", "949"),
    ("paraphrase", "watched it as a young teen in Germany expecting action but it was dull", "2165"),
    # compound: several ideas at once
    ("compound",   "high school football players covering up a crime to protect friends in a narrow-minded small town", "4132"),
    ("compound",   "Japanese director follows up his college sumo comedy with another hit", "4288"),
    ("compound",   "remake with great masks and sets but a wooden lead actor", "565"),
    ("compound",   "brain surgery drama duller than House or Grey's Anatomy", "680"),
    ("compound",   "baseball sequel that moves the team from Cleveland to Minnesota", "4200"),
    ("compound",   "slow dull characters for most of the film but patience pays off with a decent ending", "612"),]

# ---------- metrics ----------
def rank_of(target, results):
    return results.index(target) + 1 if target in results else None

methods = {"BM25 only": search_bm25, "Semantic only": search_semantic, "Hybrid (RRF)": search_hybrid}
ranks = {name: [rank_of(t, fn(q, 10)) for _, q, t in tests] for name, fn in methods.items()}

def recall_at_10(rs): return sum(r is not None for r in rs) / len(rs)
def mrr(rs):          return sum(1 / r for r in rs if r) / len(rs)

rows = [(name, recall_at_10(rs), mrr(rs)) for name, rs in ranks.items()]
for name, rec, m in rows:
    print(f"{name:<14} Recall@10 = {rec:.2f}   MRR = {m:.2f}")

# ---------- write day07/eval.md ----------
fmt = lambda r: str(r) if r else "–"
lines = ["# Day 7 — Hybrid Search Bake-off", "",
         "5,000 IMDB reviews in Elasticsearch (`imdb_hybrid`), all-MiniLM-L6-v2 embeddings, 20 hand-crafted queries.", "",
         "| Method | Recall@10 | MRR |", "|---|---|---|"]
lines += [f"| {n} | {rec:.2f} | {m:.2f} |" for n, rec, m in rows]
lines += ["", "## Rank of the relevant review per query (– = not in top 10)", "",
          "| Type | Query | BM25 | Semantic | Hybrid |", "|---|---|---|---|---|"]
for i, (typ, q, _) in enumerate(tests):
    lines.append(f"| {typ} | {q} | " + " | ".join(fmt(ranks[n][i]) for n in methods) + " |")
open("day07/eval.md", "w", encoding="utf-8").write("\n".join(lines) + "\n")
print("\nWrote day07/eval.md")