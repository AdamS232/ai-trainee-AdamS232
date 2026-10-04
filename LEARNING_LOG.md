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

## Day 7 — Vector databases & hybrid search
- pgvector: stored vectors in PostgreSQL and ranked by cosine with `<=>` (cat 1.00, dog 0.995, car 0.07)
- Loaded the same 5,000 IMDB reviews into ChromaDB, Qdrant, and Elasticsearch (`dense_vector`); all three returned the same nearest neighbors
- With a sentiment filter, ChromaDB missed a review that Qdrant ranked #1: approximate (HNSW) search + filtering can skip the true best match
- Bake-off on 20 queries (Recall@10 / MRR): BM25 0.75 / 0.68, semantic 0.65 / 0.53, hybrid (RRF) **0.85** / 0.63. See `day07/eval.md`
- Hybrid's biggest gain was on paraphrase queries (4/7 vs 2/7 for each method alone); BM25 kept the best MRR because RRF sometimes pulled its #1 results down

### Self-reflection answers
1. RRF only uses ranks, and the same `k` is applied to every document, so changing it rarely changes the final order much. A smaller `k` just gives top ranks more weight; a larger one flattens the differences. That's why the default of 60 works well almost everywhere.

2. Qdrant runs as a proper server built to scale (sharding, replication, vector compression) and applies filters during the search itself. ChromaDB is mainly for local prototypes. I saw it miss the best match when filtering by sentiment, which would get worse at 10 million documents.

3. kNN can fail on rare exact names: in my bake-off, semantic search completely missed "Richard Brooks The Professionals," which BM25 ranked #1, because the embedding captures the general meaning rather than specific names. BM25 fails when the query shares no words with the document, like paraphrases: "reminds me of campus antiwar demonstrations in the sixties" found nothing with BM25 but was semantic's #1.

## Day 8 — RAG (Retrieval-Augmented Generation)

**What I built**
- `day08/rag_scratch.py` — RAG from scratch: pypdf → custom chunker (800 chars, 100 overlap) → bge-small embeddings → ChromaDB → cross-encoder re-rank (top 50 → top 5) → llama3.2:3b via Ollama.
- `day08/rag_langchain.py` — same pipeline with LangChain (PyPDFLoader, RecursiveCharacterTextSplitter, Chroma, ChatOllama, LCEL chain).
- `day08/rag_llamaindex.py` — same pipeline in ~8 lines with LlamaIndex (in-memory index).
- `day08/rag_app.py` — Gradio PDF Q&A chatbot over 3 papers (RAG, Transformer, Sentence-BERT) with re-ranking and a collapsible Sources panel.
- `day08/test_questions.md` — 4 test questions and actual answers.

**Key results**
- All three versions answered "RAG-Sequence and RAG-Token" correctly.
- Re-ranking changed the answer from just naming the two formulations to explaining the difference between them, because the cross-encoder picked more explanatory chunks.
- App: 3/4 tests passed as expected; the two-paper synthesis question returned "I don't know" (expected failure).

**Problems I hit and fixed**
- Chroma rejected the collection name `"kb"` (minimum 3 characters) → renamed to `"kb_docs"`.
- The whole PDF became **1 chunk** because the chunker split on blank lines and pypdf output has none → split on any newline instead.
- The LangChain install upgraded websockets to 17.1 again (breaks Gradio) → pinned back to 12.0.
- The doc's Gradio sketch built a re-ranked prompt but then called `chain.invoke(question)`, which ignores it → rewrote `answer()` to send the re-ranked context straight to the LLM.

**Self-reflection**

1. *Why does chunk-with-overlap usually outperform fixed-size-no-overlap?*
   Fixed-size chunks cut wherever the character count runs out, often mid-sentence. Then a fact gets split in half, and neither chunk has the whole thing, so neither matches the question well. Overlap repeats the last ~100 characters at the start of the next chunk, so anything near a boundary appears whole in at least one chunk. I saw how much chunking matters when my first run produced just 1 chunk: retrieval had nothing to choose between and sent the whole paper.

2. *Why is re-ranking cheap enough to add, but retrieval with a cross-encoder alone is not?*
   A bi-encoder embeds every chunk once ahead of time, so a search is one question embedding plus a fast vector lookup. A cross-encoder has to read the question *together with* each chunk, so nothing can be precomputed. Using it alone means running the model on every chunk in the database for every question. Re-ranking runs it only on the top 20–50 candidates, which costs about 50–100 ms instead of seconds or minutes, and you still get the accuracy boost where it matters.

3. *Name two hallucination modes you observed today, and what you'd do to mitigate each.*
   - **Fake/placeholder citations:** in the app, the model cited `[file p.5]`, copying the word "file" from the prompt's example instead of the real file name. Mitigation: don't let the LLM write citations at all. Attach sources in code from the retrieved chunks' metadata (like my Sources panel does), or give a concrete example citation with a real file name in the prompt.
   - **Ignoring instructions (missing citations):** the scratch version and app answer 1 gave no citation even though the prompt required one. A 3B model doesn't follow every rule reliably. Mitigation: check the output in code (e.g., reject or retry if no `[...]` citation appears), or use a larger model.
   - (The synthesis question was a *safe* failure: the model refused rather than inventing an answer. The fix there is retrieval, not the prompt: retrieve per document, raise top_k, or split the question into sub-questions.)

   ## Day 9 — Fine-Tuning with LoRA / QLoRA

**What I built**
- `day09/prepare_data.py` — converts OpenAssistant guanaco (`### Human:/### Assistant:`) into Llama 3.2's chat template.
- `day09/finetune_qlora.py` — QLoRA fine-tune of Llama-3.2-1B-Instruct (NF4 4-bit base, LoRA r=16, alpha=32, all 7 linear layers), 2 epochs, logged to W&B.
- `day09/eval_compare.py` + `eval_qualitative.md` — 10-prompt side-by-side, base vs fine-tuned, scored 1–5.
- `day09/eval_rouge.py` — ROUGE on 50 held-out test conversations, base vs fine-tuned.
- `day09/merge.py` → merged fp16 model → GGUF (llama.cpp) → Q4_K_M → served in Ollama via `day09/Modelfile` as `my-tuned`.

**Key results**
- Trained locally on an RTX 4070 Laptop (8 GB) in 54 min. Train loss 2.18 → ~1.49; eval loss 1.552 (epoch 1) → 1.545 (epoch 2).
- ROUGE-L: base 0.187 → fine-tuned 0.230 (+23% relative); improved on ROUGE-1/2/Lsum too.
- Qualitative averages (Helpfulness / Style / Factuality): base 4.1 / 3.8 / 4.4, fine-tuned 3.6 / 4.1 / 4.0. Fine-tuned won 4/10 prompts, base 5/10, 1 tie — **did not meet the ≥6/10 target**.
- Final Q4_K_M GGUF: **763 MB** (from 2,357 MB fp16); Ollama lists it at 807 MB.

**What I learned from the results**
- Fine-tuning changed *style* (more concise, closer to the dataset's answers — confirmed by ROUGE) but not *knowledge*; factuality slightly dropped (e.g., wrong list/tuple facts).
- The base was already instruction-tuned, so generic chat data had little to add. A small domain-specific dataset would be the better use of fine-tuning; for facts, RAG (Day 8) is the right tool.
- Epoch 2 barely improved eval loss (1.552 → 1.545) while train loss kept dropping → early overfitting; 1 epoch would have been almost as good at half the time.
- ROUGE measures word overlap, not correctness — it rose even though factuality fell.

**Problems I hit and fixed**
- PyTorch was the CPU-only build (`2.14.0+cpu`) → reinstalled the CUDA 12.6 build.
- The doc's install was missing `bitsandbytes`; the script imported a `prepare_data` module that didn't exist → created it.
- OOM during evaluation: default eval batch size 8 × 1024 tokens × 128k vocab needed 3.9 GB → `per_device_eval_batch_size=1`.
- The simple parser silently dropped ~25% of conversations (9,846 → 7,376).
- The doc's ROUGE code split on the word "assistant", which corrupted prompts and references → rewrote it using the raw first human/assistant turn, and scored the base model too for comparison.
- Skipped `llama.cpp`'s `requirements.txt` (it could have replaced my CUDA PyTorch); `llama-quantize` needs compiling → used the prebuilt Windows release instead. Ollama's `--quantize` doesn't accept GGUF input.
- The doc's Modelfile template had no Llama 3 chat markers → replaced it with the proper Llama 3 template.

**Self-reflection**

1. *Why do we double-quantize in NF4?*
   NF4 stores weights in blocks of 64, and each block needs its own scaling number, stored in 32-bit. That adds about 0.5 extra bits per weight, which is a lot when the weights themselves are only 4 bits. Double quantization compresses those scaling numbers too (to 8-bit), cutting the overhead to about 0.13 bits per weight. It's free memory savings with almost no quality cost. On my 1B model it only saves about 45 MB, but on a 65B model it saves around 3 GB, which can decide whether training fits on the GPU at all.

2. *Why does `packing=True` matter so much for short-sample instruction datasets?*
   Most guanaco conversations are much shorter than the 1,024-token limit. Without packing, every short example gets padded with filler up to the longest one in its batch, so the GPU spends most of its time on padding that teaches nothing. Packing joins several conversations into full 1,024-token blocks. In my run, 7,376 conversations became 2,869 packed blocks, about 2.6 conversations per block, so training took roughly 2.5x fewer steps. That's the difference between about 54 minutes and over 2 hours.

3. *What would break if you set `lora_alpha` much larger than 2 × r?*
   The adapter's changes are multiplied by alpha ÷ r. With r=16 and alpha=32 that's ×2. With, say, alpha=512 it would be ×32, so every update would be 16x stronger. That works like a much higher learning rate: loss and gradient spikes, possible NaN (not-a-number) errors in fp16, and the adapter overpowering the base model, which makes it forget what it knew and produce repetitive or garbled text. To compensate, you'd have to lower the learning rate by about the same factor.

   **Case study: Python Tutor specialist**
- Wrote a 100-example domain dataset (`day09/python_tutor.jsonl`): Python/NumPy/pandas concepts, each answered in exactly two paragraphs with the same system prompt.
- QLoRA fine-tune in 65 seconds (66 steps). Train loss 2.86 → 0.78; eval loss best at epoch 2 (1.182), slightly worse at epoch 3 (1.195) → overfitting; 2 epochs would have been enough.
- Evaluated on 10 Python topics *not* in the training data (`day09/case_study_eval.md`), same system prompt for both models.
- Style: base 2.0 → fine-tuned **5.0** (followed the two-paragraph format 10/10 vs 0/10). Factuality: 3.0 → 1.9 (invented APIs like `@abc.init`).
- Fine-tuned won 7/10 on total score, but only through Style; on helpfulness + factuality, base won 9/10.
- Lesson: a small focused dataset changes tone and format very effectively, but can't add knowledge. Combine fine-tuning (for format) with RAG (for facts).