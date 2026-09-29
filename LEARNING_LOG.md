# Learning Log

## Day 1 — Environment setup

### Self-reflection answers
1. The difference between pip and mamba is that pip installs Python packages only, while mamba manages whole environments. People tend to use mamba for environment/system level stuff, while pip for Python packages ontop.

2. Pinning makes sure all installs are reproducible, so it makes sure that everyone gets the same versions. Its important because if the library updates later it doesn't silently break or change.

3. It would return False because without "--gpus all", the container can't see the host's GPU at all.

## Day 2 — Classical NLP & IMDB baseline
- Built text-cleaning functions with NLTK and spaCy (`src/preprocess_nltk.py`, `src/preprocess_spacy.py`), each with an `if __name__ == "__main__":` smoke test
- Smoke test showed the spaCy version leaves a stray "br" from HTML tags because it has no HTML-stripping step
- EDA on 25k IMDB reviews: class balance, review lengths, top words and bigrams, duplicates
- TF-IDF + Logistic Regression baseline: **89.0% accuracy / 0.8907 F1**
- Model saved to `day02/models/day02_baseline.joblib` (gitignored)
- Saved a 1,000-review sample to SQLite (`day02/reviews.db`) and queried it with SQL

### Self-reflection answers
1. Stemming just chops off word endings, so it often makes fake words ("studies" → "studi"). Lemmatization turns words into real dictionary forms ("studies" → "study"), which keeps the meaning clearer.

2. Keep stopwords when small words change the meaning, like in sentiment analysis ("not good" vs "good"), or when the model needs full sentences, like transformers.

3. IDF gives less weight to words that appear in almost every document (like "movie") and more weight to rare words that actually tell documents apart.

## Day 3 — Hugging Face & DistilBERT fine-tune
- Ran 5 `pipeline()` tasks: sentiment, zero-shot, NER, question answering, summarization (`day03/pipelines_tour.ipynb`)
- W&B run: https://wandb.ai/adam234-datum-analysis/ai-trainee/runs/8bbafhqg
- Result: **91.3% accuracy / 0.9130 F1** (vs. 89.0% / 0.8907 for the Day 2 TF-IDF baseline)
- Trained on all 25k reviews on a Colab T4 GPU in 7 min 44 sec, because my laptop has no GPU
- A first run on only 8k reviews got 0.9009 F1, just under the 0.91 target, so I retrained on the full data
- Write-up: day03/WRITEUP.md

### Self-reflection answers
1. The RTX 3060 has tensor cores that do 16-bit math very fast. Older Pascal GPUs don't, so fp16 isn't faster there, and 16-bit numbers are less precise, so very small or very large values can get rounded off and hurt training.

2. An attention mask is a list of 1s and 0s that tells the model which tokens are real words and which are just padding. A random mask would make the model ignore real words and read padding, so its outputs would be wrong and unpredictable.

3. - **Pretraining:** the model learns language by predicting words in huge amounts of unlabeled text.
   - **Fine-tuning:** further training on a smaller labeled dataset for one task, like IMDB sentiment.
   - **Instruction tuning:** fine-tuning on instruction → answer examples so the model follows requests.
   - **RLHF:** humans rank the model's answers, and the model is trained to give the kind of answers people rank higher.

## Day 4 — Running LLMs locally
- Ollama serving llama3.2:3b and phi3.5:3.8b, called from Python via the REST API
- llama.cpp loaded a Q4_K_M GGUF and generated text (CPU only, since this laptop has no GPU for offload)
- transformers + bitsandbytes loaded Llama-3.2-3B in 4-bit on Colab: 2.3 GB GPU memory (vs ~6.4 GB full size)
- Built `day04/chatbot.py`: streams replies, keeps history, supports /clear, /exit, /swap (screenshot: `day04/demo.png`)
- Prompting experiments in `day04/prompting_experiments.ipynb`: few-shot fixed the output format, temperature 0 gave identical answers, and the model returned valid JSON
- Q4_K_M vs fp16: fp16 stores each number with 16 bits, Q4_K_M with about 4 bits, so the model is ~4x smaller and faster with a small drop in quality.

### Self-reflection answers
1. Different hardware does floating-point math in slightly different ways, so tiny rounding differences can change which word comes out on top when two words are nearly tied. Once one word changes, the rest of the answer goes a different way.

2. The KV-cache saves the model's work on tokens it has already read, so it doesn't redo it for every new word. It grows with the conversation because every new token adds another entry, so longer chats use more memory.

3. Use Ollama for quick setup, trying models, and apps that just need a local server to call. Use llama-cpp-python when you want full control inside Python, like picking the exact GGUF file and settings, without running a separate server.

## Day 5 — Embeddings & semantic search
- Compared MiniLM, BGE-small, and MPNet on my own test pairs (MiniLM won on my set; BGE is best on the MTEB benchmark)
- Built a FAISS index over 20,000 chunks of Simple English Wikipedia (`day05/build_index.py`)
- Recall@5 = 0.8 on 10 test queries (target ≥ 0.7), about 2 ms per search (target < 50 ms)
- Gradio search app: `day05/search_app.py`
- Lesson: embeddings match topic, not meaning, so opposites like "hot" vs "cold" still score as similar

### Self-reflection answers
1. The word "bank" alone gives the model no context, so it always gets the same embedding. Putting it in a sentence, or using a cross-encoder that reads the query and document together, fixes this.

2. Flat search checks every vector: 10 million comparisons per search. HNSW follows shortcuts and only checks a few thousand. With only 1,000 vectors, checking all of them is so fast that HNSW's extra steps cost more than they save.

3. Without normalizing, longer vectors get higher scores just because of their size, not their meaning, so the search ranks the wrong results higher.

## Day 6 — Elasticsearch & BM25
- Ran Elasticsearch 8.15 + Kibana in Docker (`day06/docker-compose.yml`) on my personal laptop, since Docker virtualization is blocked on the work laptop
- Tried mappings, match, bool/filter, and aggregations in Kibana Dev Tools; connected from Python and bulk-indexed with `helpers.bulk`
- Indexed all 25,000 IMDB training reviews (`day06/index_imdb.py`) and ran 5 queries (`day06/queries.md`)
- "waste of time" matched 787 reviews, 94% negative; the english analyzer drops "of" but keeps its position, so "waste your time" matched too
- BM25 vs semantic on the same 20k Wikipedia chunks: BM25 9/10, semantic 8/10 (`day06/compare_bm25_vs_semantic.md`). BM25 wins on rare exact words; semantic ranks paraphrases better

### Self-reflection answers
1. A `keyword` field isn't analyzed, so `match` compares the word to the **whole stored value, exactly**, including case. "positive" matches a sentiment of "positive," but "Positive" or "pos" returns nothing, and "science" won't match a value of "Science Fiction."

2. The `english` analyzer **removes English stopwords** ("the," "of," "is"), while `standard` keeps them by default. It also **stems words** ("running" → "run," "movies" → "movi"), while `standard` leaves words as they are.

3. IDF measures how few documents contain a word. A rare word like "cinematography" appears in few reviews, so matching it is strong evidence that a review is relevant and it gets a high weight. A common word like "movie" appears almost everywhere, so matching it tells you little and it gets a low weight.