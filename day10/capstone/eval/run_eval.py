"""End-to-end evaluation of AskThePaper through the running API.

Usage (API must be running, with attention.pdf, my_document.pdf and sbert.pdf indexed):
    python day10/capstone/eval/run_eval.py
"""
import json
import re
import statistics
import time
from pathlib import Path

import requests

API = "http://localhost:8000"
HERE = Path(__file__).parent
IDK_PREFIX = "I don't know"


def contains(text: str, keyword: str) -> bool:
    """Case-insensitive match; numbers must match as whole numbers (so '8' doesn't match '18')."""
    if re.fullmatch(r"[\d.]+", keyword):
        return re.search(rf"(?<![\d.]){re.escape(keyword)}(?![\d.])", text) is not None
    return keyword.lower() in text.lower()


def ask(question: str) -> dict:
    """Call /chat, collect the streamed answer, and time it."""
    t0 = time.perf_counter()
    first_token = None
    answer, sources = "", []
    with requests.post(f"{API}/chat", json={"question": question}, stream=True, timeout=300) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line:
                continue
            msg = json.loads(line)
            if "sources" in msg:
                sources = msg["sources"]
            elif "token" in msg:
                if first_token is None:
                    first_token = time.perf_counter() - t0
                answer += msg["token"]
            elif "error" in msg:
                raise RuntimeError(msg["error"])
    return {"answer": answer.strip(), "sources": sources,
            "ttft": first_token or 0.0, "total": time.perf_counter() - t0}


def p95(values: list[float]) -> float:
    s = sorted(values)
    return s[min(len(s) - 1, round(0.95 * (len(s) - 1)))]


def main() -> None:
    questions = json.loads((HERE / "questions.json").read_text(encoding="utf-8"))
    rows = []
    for q in questions:
        out = ask(q["question"])
        refused = out["answer"].startswith(IDK_PREFIX)
        row = {**q, **out, "refused": refused}
        if q["type"] == "out_of_scope":
            row["correct"] = refused
            row["retrieval_hit"] = None
        else:
            evidence = [s for s in out["sources"] if s["source"] == q["source"]
                        and any(contains(s["text"], k) for k in q["keywords"])]
            row["retrieval_hit"] = bool(evidence)
            row["correct"] = (not refused) and any(contains(out["answer"], k) for k in q["keywords"])
        row["cited"] = bool(re.search(r"\[S\d+\]", out["answer"]))
        rows.append(row)
        print(f"{q['id']:>2} {q['type']:<12} correct={row['correct']!s:<5} "
              f"hit={row['retrieval_hit']!s:<5} {out['total']:.1f}s  {out['answer'][:70]!r}")

    answerable = [r for r in rows if r["type"] != "out_of_scope"]
    oos = [r for r in rows if r["type"] == "out_of_scope"]
    answered = [r for r in answerable if not r["refused"]]
    totals = [r["total"] for r in rows]
    ttfts = [r["ttft"] for r in rows]

    def pct(n: int, d: int) -> str:
        return f"{n}/{d} ({100 * n / d:.0f}%)" if d else "n/a"

    summary = {
        "Retrieval hit rate (answerable)": pct(sum(r["retrieval_hit"] for r in answerable), len(answerable)),
        "Answer accuracy (answerable)": pct(sum(r["correct"] for r in answerable), len(answerable)),
        "  - positive": pct(sum(r["correct"] for r in answerable if r["type"] == "positive"),
                            sum(r["type"] == "positive" for r in answerable)),
        "  - paraphrase": pct(sum(r["correct"] for r in answerable if r["type"] == "paraphrase"),
                              sum(r["type"] == "paraphrase" for r in answerable)),
        "Correct refusals (out-of-scope)": pct(sum(r["correct"] for r in oos), len(oos)),
        "False refusals (answerable)": pct(sum(r["refused"] for r in answerable), len(answerable)),
        "Inline [S#] citation rate (answered)": pct(sum(r["cited"] for r in answered), len(answered)),
        "Time to first token, median / p95": f"{statistics.median(ttfts):.2f}s / {p95(ttfts):.2f}s",
        "Total latency, median / p95": f"{statistics.median(totals):.2f}s / {p95(totals):.2f}s",
    }

    lines = ["# AskThePaper — Evaluation Results", "",
             f"{len(rows)} questions over 3 PDFs (attention.pdf, my_document.pdf, sbert.pdf): "
             f"{sum(r['type'] == 'positive' for r in rows)} positive, "
             f"{sum(r['type'] == 'paraphrase' for r in rows)} paraphrase, {len(oos)} out-of-scope.", "",
             "## Summary", "", "| Metric | Result |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in summary.items()]
    lines += ["", "## Per-question results", "",
              "| # | Type | Question | Retrieval hit | Correct | Cited | Time | Answer (truncated) |",
              "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        ans = r["answer"].replace("\n", " ").replace("|", "/")[:110]
        hit = "—" if r["retrieval_hit"] is None else ("✅" if r["retrieval_hit"] else "❌")
        lines.append(f"| {r['id']} | {r['type']} | {r['question']} | {hit} | "
                     f"{'✅' if r['correct'] else '❌'} | {'✅' if r['cited'] else '—'} | "
                     f"{r['total']:.1f}s | {ans} |")
    lines += ["", "## Method", "",
              "- **Retrieval hit:** a returned source is from the expected PDF *and* contains an expected keyword.",
              "- **Correct (answerable):** the answer is not a refusal and contains an expected keyword.",
              "- **Correct (out-of-scope):** the answer is the refusal \"I don't know based on the provided documents.\"",
              "- Timed end-to-end through the API (`/chat`, streamed), on an RTX 4070 laptop with llama3.2:3b in Ollama.",
              "- Keyword matching is a rough proxy: it can't judge wording quality and could miss a correct answer phrased differently."]

    (HERE / "results.md").write_text("\n".join(lines), encoding="utf-8")
    (HERE / "results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print("\n" + "\n".join(f"{k}: {v}" for k, v in summary.items()))
    print(f"\nSaved {HERE / 'results.md'}")


if __name__ == "__main__":
    main()