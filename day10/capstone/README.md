# AskThePaper

## 1. Pitch

AskThePaper lets you upload a PDF and ask questions about it in a chat. Every answer comes only from the document and shows the file and page it came from, and when the answer isn't in the PDF it says "I don't know" instead of making something up. It's for anyone who needs quick, checkable answers from long documents like research papers, manuals or reports. Everything runs locally, so documents never leave your machine.

## 2. Architecture

```
┌─────────────────────────────┐
│  Streamlit UI  :8501        │   upload PDF · chat · sources panel
└──────────────┬──────────────┘
               │ HTTP
┌──────────────▼──────────────┐
│  FastAPI  :8000             │
│  GET  /health  liveness     │
│  POST /ingest  upload+index │
│  POST /search  top-k chunks │
│  POST /chat    RAG answer   │
└──────┬───────────────┬──────┘
       │               │
┌──────▼──────┐  ┌─────▼───────────┐
│  ChromaDB   │  │  Ollama :11434  │
│  (vectors)  │  │  llama3.2:3b    │
└─────────────┘  └─────────────────┘
```

**Pipeline:** PDF → 800-character chunks (100 overlap) → embed with `BAAI/bge-small-en-v1.5` → ChromaDB → retrieve top 50 → re-rank with `cross-encoder/ms-marco-MiniLM-L-6-v2` → keep top 4 → `llama3.2:3b` → answer + sources. If the best re-rank score is below the threshold, it answers "I don't know" without calling the LLM.

## 3. Prerequisites

| Requirement | Version |
|---|---|
| Python | 3.11.5 |
| Docker | Not required |
| Ollama | 0.35.1 |
| Ollama model | `llama3.2:3b` (~2 GB) |
| OS | Windows 11 (tested) |
| GPU | Not required |

## 4. Setup

From the repo root:

```powershell
git clone https://github.com/AdamS232/ai-trainee-AdamS232.git
cd ai-trainee-AdamS232
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r day10/capstone/requirements.txt
ollama pull llama3.2:3b
copy day10\capstone\.env.example day10\capstone\.env
```

Make sure Ollama is running before you start the API. The first run downloads the embedding and re-ranking models (~150 MB), so it takes a minute.

## 5. Usage

Run each in its own terminal, from the repo root.

API (docs at http://localhost:8000/docs):

```powershell
uvicorn day10.capstone.api.main:app --port 8000
```

UI (opens at http://localhost:8501):

```powershell
streamlit run day10/capstone/ui/app.py
```

In the UI, upload a PDF in the sidebar, click **Index document**, then ask questions in the chat. Sources show under each answer.

Sample requests (on Windows use `curl.exe`):

```powershell
curl.exe http://localhost:8000/health
curl.exe -F "file=@paper.pdf" http://localhost:8000/ingest
curl.exe -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d "{\"question\": \"What is the main contribution?\"}"
```

## 6. Known limitations

- The 3B model sometimes skips `[S#]` citations in its answer. The sources panel still shows where the context came from.
- The "I don't know" threshold was tuned by hand on a few PDFs. Set too high, it refuses answerable questions; set too low, it answers questions it shouldn't.
- Only text PDFs work well. Scanned pages, tables and equations are lost or garbled.
- Every question searches all indexed PDFs; there's no per-document filter.
- The eval checks answers by keyword matching on only 17 questions.

## 7. Evaluation

Measured with `day10/capstone/eval/run_eval.py` on the 17 questions in `eval/questions.json` (9 direct, 4 paraphrased, 4 out-of-scope) across 3 PDFs: the Transformer, RAG and Sentence-BERT papers. Run locally on Windows 11 with an RTX 4070 Laptop GPU.

| Metric | Result |
|---|---|
| Retrieval hit rate (answerable) | 11/13 (85%) |
| Answer accuracy (answerable) | 8/13 (62%) |
| ↳ direct questions | 6/9 (67%) |
| ↳ paraphrased questions | 2/4 (50%) |
| Correct refusals (out-of-scope) | 4/4 (100%) |
| False refusals (answerable) | 1/13 (8%) |
| Inline `[S#]` citation rate | 11/12 (92%) |
| Time to first token, median / p95 | 4.83 s / 4.95 s |
| Total latency, median / p95 | 5.06 s / 5.77 s |

**What failed:**

- **Retrieval misses (2):** questions about the optimizer (Adam) and RAG's knowledge source (Wikipedia). The right chunk didn't make the top 4.
- **False refusal (1):** the BLEU score question. The answer sits in a table, which pypdf extracts as messy text, so it scored below the "I don't know" threshold.
- **Wrong answers with the right chunk (2):** the d_model question and the SBERT speed-up question (the model answered "9% faster" instead of 65 hours → 5 seconds). The 3B model misread the context.

It never answered an out-of-scope question, which was the main safety goal.

## 8. What I'd do with another week

- Hybrid search (BM25 + vectors) for exact terms, names and numbers.
- A per-document filter so users can pick which PDF to ask about.
- Better PDF parsing for tables and scanned pages.
- A bigger eval set with LLM-graded answers and automatic threshold tuning.
- Docker Compose to start everything with one command.