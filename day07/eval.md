# Day 7 — Hybrid Search Bake-off

5,000 IMDB reviews in Elasticsearch (`imdb_hybrid`), all-MiniLM-L6-v2 embeddings, 20 hand-crafted queries.

| Method | Recall@10 | MRR |
|---|---|---|
| BM25 only | 0.75 | 0.68 |
| Semantic only | 0.65 | 0.53 |
| Hybrid (RRF) | 0.85 | 0.63 |

## Rank of the relevant review per query (– = not in top 10)

| Type | Query | BM25 | Semantic | Hybrid |
|---|---|---|---|---|
| keyword | Pamela Anderson slasher | 1 | 1 | 1 |
| keyword | Sondra Locke Clint Eastwood | 1 | 1 | 1 |
| keyword | Andreas Schnaas zombies | 2 | 8 | 2 |
| keyword | Boringus Maximus | 1 | 1 | 1 |
| keyword | Richard Brooks The Professionals | 1 | – | 5 |
| keyword | Smallville Tom Welling Michael Rosenbaum | 1 | 1 | 1 |
| keyword | Josie Lawrence Whose Line | 1 | 1 | 1 |
| paraphrase | actress faking an eastern European accent opposite a timid office worker | – | – | – |
| paraphrase | obsolete soldier abandoned on a distant world after being replaced by engineered troops | 1 | – | 7 |
| paraphrase | female pilot like a famous missing aviator who has never fallen in love | – | – | 3 |
| paraphrase | reminds me of campus antiwar demonstrations in the sixties | – | 1 | 1 |
| paraphrase | cartoon series with an adorable small mechanical character | – | 2 | – |
| paraphrase | aging Scottish star looks lost in his role while his costars shine | – | – | – |
| paraphrase | watched it as a young teen in Germany expecting action but it was dull | 2 | – | 3 |
| compound | high school football players covering up a crime to protect friends in a narrow-minded small town | 1 | 2 | 1 |
| compound | Japanese director follows up his college sumo comedy with another hit | 1 | 1 | 1 |
| compound | remake with great masks and sets but a wooden lead actor | 2 | – | 10 |
| compound | brain surgery drama duller than House or Grey's Anatomy | 1 | 1 | 1 |
| compound | baseball sequel that moves the team from Cleveland to Minnesota | 1 | 1 | 1 |
| compound | slow dull characters for most of the film but patience pays off with a decent ending | 1 | 2 | 1 |

## Hits by query type

| Query type | BM25 | Semantic | Hybrid |
|---|---|---|---|
| Keyword (7) | 7 | 6 | 7 |
| Paraphrase (7) | 2 | 2 | 4 |
| Compound (6) | 6 | 5 | 6 |
| **Total (20)** | **15** | **13** | **17** |

## Findings

- **Hybrid wins on Recall@10 (0.85, 17/20)**. Its biggest gain is on paraphrases: 4/7, vs 2/7 for each method alone. For "female pilot like a famous missing aviator", neither BM25 nor semantic had the review in their top 10, but hybrid ranked it #3 — it was mid-ranked in *both* top-50 lists, and RRF rewards that agreement.
- **BM25 wins on MRR (0.68)**. It ranked the target #1 on 14 of its 15 hits. RRF sometimes pulled those down when semantic search disagreed: Richard Brooks went from #1 to #5, the soldier review from #1 to #7, and the Planet of the Apes remake from #2 to #10.
- **Semantic handled keywords better than expected** (6/7, including "Boringus Maximus" at #1), but it missed "Richard Brooks The Professionals" completely. Its weakest area was paraphrases, where it found only 2/7.
- **Hybrid can also lose a hit.** "Cartoon series with an adorable small mechanical character" was semantic #2, but BM25 ranked it low, so RRF pushed it out of the top 10.
- **Two queries failed everywhere**: the Kidman accent query and the "aging Scottish star" query. Both paraphrases were too indirect: "eastern European" and "Scottish star" never appear in the reviews, and the embedding model didn't connect them to "Russian accent" or "Connery".
- **Why BM25 beat semantic, unlike the doc's expectation**: the reviews are short and full of distinctive names, which suits exact matching. A set with more pure paraphrases would likely favor semantic search.