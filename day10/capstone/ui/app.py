"""Streamlit UI for AskThePaper: upload PDFs, chat with streaming answers, see sources."""
import json
import os

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv("day10/capstone/.env")
API = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="AskThePaper", page_icon="📄", layout="wide")
st.title("📄 AskThePaper")
st.caption("Upload a PDF and ask questions. Answers come only from your documents, with page citations.")

if "history" not in st.session_state:
    st.session_state.history = []


def show_sources(sources: list[dict] | None) -> None:
    """Collapsible list of the chunks the answer was based on."""
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})"):
        for i, s in enumerate(sources, start=1):
            st.markdown(f"**[S{i}] {s['source']} — page {s['page']}** · relevance {s['score']:.2f}")
            text = s["text"]
            st.caption(text[:400] + ("…" if len(text) > 400 else ""))


# ---------- Sidebar: status + upload ----------
with st.sidebar:
    st.header("Documents")
    if "flash" in st.session_state:
        st.success(st.session_state.pop("flash"))
    try:
        health = requests.get(f"{API}/health", timeout=5).json()
    except requests.RequestException:
        st.error(f"API not reachable at {API}.\n\nStart it with:\n`uvicorn day10.capstone.api.main:app --port 8000`")
        st.stop()

    if not health["ollama"]:
        st.warning("Ollama is not reachable. Start the Ollama app, then refresh.")
    if health["documents"]:
        st.markdown("**Indexed:**")
        for d in health["documents"]:
            st.markdown(f"- {d}")
    else:
        st.info("No documents yet. Upload a PDF below.")

    up = st.file_uploader("Upload a PDF", type=["pdf"])
    if up and st.button("Index document", type="primary"):
        with st.spinner(f"Indexing {up.name}..."):
            try:
                r = requests.post(f"{API}/ingest",
                                  files={"file": (up.name, up.getvalue(), "application/pdf")},
                                  timeout=600)
            except requests.RequestException as e:
                st.error(f"Upload failed: {e}")
                st.stop()
        if r.ok:
            st.session_state.flash = f"Indexed {r.json()['chunks']} chunks from {up.name}"
            st.rerun()
        else:
            st.error(r.json().get("detail", "Indexing failed."))

    if st.button("Clear chat"):
        st.session_state.history = []
        st.rerun()

# ---------- Chat ----------
for m in st.session_state.history:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        show_sources(m.get("sources"))

if q := st.chat_input("Ask about your documents..."):
    st.session_state.history.append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.markdown(q)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        answer, sources = "", []
        try:
            with requests.post(f"{API}/chat", json={"question": q}, stream=True, timeout=300) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    if not line:
                        continue
                    msg = json.loads(line)
                    if "sources" in msg:
                        sources = msg["sources"]
                    elif "token" in msg:
                        answer += msg["token"]
                        placeholder.markdown(answer + "▌")   # typing cursor while streaming
                    elif "error" in msg:
                        answer = f"⚠️ {msg['error']}"
        except requests.RequestException as e:
            answer = f"⚠️ Request failed: {e}"
        placeholder.markdown(answer)
        show_sources(sources)

    st.session_state.history.append({"role": "assistant", "content": answer, "sources": sources})