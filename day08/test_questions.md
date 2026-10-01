# Day 8 — RAG App Test Questions

**Corpus:** `attention.pdf` (Transformer), `sbert.pdf` (Sentence-BERT), `my_document.pdf` (RAG paper)
**Pipeline:** bge-small retrieval (top 20) → cross-encoder re-rank (top 3) → llama3.2:3b

| # | Type | Question | Model answer | Result |
|---|------|----------|--------------|--------|
| 1 | Positive fact | How many attention heads does the base Transformer use? | "The Transformer uses 8 parallel attention layers, or heads." | Correct |
| 2 | Paraphrase | In the Transformer paper, how many parallel attention layers are there? | "...there are 8 parallel attention layers, or heads. [file p.5]" | Correct, but citation used the placeholder "file" instead of the real file name |
| 3 | Out-of-corpus | What is the population of Canada? | "I don't know based on the provided documents." | Correctly refused |
| 4 | Synthesis (two papers) | How does Sentence-BERT's approach to embeddings differ from how RAG retrieves passages? | "I don't know based on the provided documents." | Expected failure — top-3 chunks came from SBERT only (top score 2.83, sbert.pdf p.2); no chunk covered both papers |

## Observations
- Retrieval + re-ranking found the right page for single-fact questions, even when reworded.
- The "ONLY use context / say I don't know" prompt rule worked: no made-up answer for Canada.
- Synthesis fails because each chunk comes from one paper and top-3 is too small to cover both. Possible fixes: retrieve top-k per document, raise top_k, or split the question into two sub-questions.
- Citations are unreliable with a 3B model: answer 1 had none, answer 2 copied the template's "file" placeholder.