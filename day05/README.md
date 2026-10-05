# Day 5 — Semantic Search over Wikipedia

Searches 20,000 chunks of Simple English Wikipedia by meaning, using
MiniLM embeddings and a FAISS index, with a Gradio web page.

## Files
- `build_index.py` — downloads articles, splits them into ~150-word chunks,
  embeds them, and saves `wiki.faiss` + `corpus.pkl`
- `search_app.py` — Gradio search page
- `embeddings.ipynb` — experiments, model comparison, and Recall@5 evaluation

## Regenerate the index
`wiki.faiss` and `corpus.pkl` are not in git (too large). To rebuild them,
run from the `ai-trainee` folder (about 10 minutes on CPU):

    python day05/build_index.py

## Run the app

    python day05/search_app.py

Then open http://127.0.0.1:7860

Note: Gradio 4.41 needs `fastapi==0.112.2`, `starlette==0.38.2`, and `pydantic<2.11`.

## Results
- Recall@5 = 0.8 on 10 test queries (target ≥ 0.7)
- Search time about 2 ms per query (target < 50 ms)
