"""RAG Question Answering (Streamlit)

Features (exactly as requested):
- Left sidebar Knowledge Base with three tabs: PDF, URL (incl. YouTube/website), Text
- Center: big question box with Get Answer
- Top-right: New + button to reset session
- History section: upload and query history with clear buttons

Back end uses FAISS vector store via LangChain. API key is read from .env (OPENAI_API_KEY) or sidebar.
"""
import os
import json
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv

from backend import loaders, rag
from backend.config import METADATA_PATH
from backend.logger import root_logger

# ---------- Bootstrap ----------
load_dotenv(override=True)
if os.getenv("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")  # ensure for langchain-openai

st.set_page_config(page_title="RAG Question Answering", layout="wide")
st.markdown(
    """
    <style>
      .block-container{padding-top:1rem;}
      .stTextArea textarea{font-size:1.05rem;}
      .hint{color:#c8c8c8;font-size:0.92rem;}
      .ok{background:#16321f;color:#b9f5d0;padding:10px 12px;border-radius:6px;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Helpers (history) ----------
HISTORY_FILE = METADATA_PATH

def _read_history():
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"uploads": [], "queries": []}


def _write_history(data):
    os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _add_upload_record(kind: str, title: str):
    h = _read_history()
    h["uploads"].insert(0, {"time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "type": kind, "title": title})
    _write_history(h)


def _add_query_record(question: str):
    h = _read_history()
    h["queries"].insert(0, {"time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "question": question})
    _write_history(h)


# ---------- Sidebar (Knowledge Base) ----------
st.sidebar.header("Knowledge Base")
kb_tab = st.sidebar.tabs(["PDF", "URL", "Text"])

# PDF tab
with kb_tab[0]:
    st.markdown("#### Upload PDF")
    pdf = st.file_uploader("Drag and drop file here", type=["pdf"], label_visibility="collapsed")
    if pdf is not None:
        if st.button("Ingest PDF", use_container_width=True):
            try:
                doc = loaders.load_pdf_bytes(pdf.read(), source=pdf.name)
                chunks = rag.ingest_text(title=pdf.name, text=doc.page_content, source="pdf", openai_api_key=os.getenv("OPENAI_API_KEY"))
                st.markdown('<div class="ok">PDF ingested successfully!</div>', unsafe_allow_html=True)
                _add_upload_record("pdf", pdf.name)
            except Exception as e:
                st.error(f"Failed to ingest PDF: {e}")
    st.caption("Limit 200MB per file • PDF")

# URL tab (website or YouTube)
with kb_tab[1]:
    st.markdown("#### Add from URL (website or YouTube)")
    url = st.text_input("Enter URL", placeholder="https://example.com/article or YouTube link")
    if st.button("Ingest URL", use_container_width=True):
        if not url.strip():
            st.warning("Enter a URL to ingest.")
        else:
            try:
                if "youtube.com" in url or "youtu.be" in url:
                    doc = loaders.load_youtube_transcript(url)
                    title = "YouTube"
                    kind = "youtube"
                else:
                    doc = loaders.load_url_to_document(url)
                    title = url
                    kind = "url"
                rag.ingest_text(title=title, text=doc.page_content, source=kind, openai_api_key=os.getenv("OPENAI_API_KEY"))
                st.markdown('<div class="ok">URL ingested successfully!</div>', unsafe_allow_html=True)
                _add_upload_record(kind, title)
            except Exception as e:
                st.error(f"Failed to ingest URL: {e}")

# Text tab
with kb_tab[2]:
    st.markdown("#### Paste Text")
    raw_text = st.text_area("Paste your text here...", height=180)
    if st.button("Ingest Text", use_container_width=True):
        if not raw_text.strip():
            st.warning("Enter some text to ingest.")
        else:
            try:
                rag.ingest_text(title="text", text=raw_text, source="text", openai_api_key=os.getenv("OPENAI_API_KEY"))
                st.markdown('<div class="ok">Text ingested successfully!</div>', unsafe_allow_html=True)
                _add_upload_record("text", "pasted text")
            except Exception as e:
                st.error(f"Failed to ingest text: {e}")

# API key in sidebar
st.sidebar.markdown("---")
sidebar_key = st.sidebar.text_input("OpenAI API Key (optional)", type="password")
if sidebar_key:
    os.environ["OPENAI_API_KEY"] = sidebar_key

# ---------- Header with New + button ----------
left, right = st.columns([0.9, 0.1])
with left:
    st.markdown("## RAG Question Answering")
with right:
    if st.button("New +"):
        st.session_state.pop("user_question", None)
        st.session_state.pop("last_answer", None)

# ---------- Ask box ----------
q = st.text_area("Ask your question here", key="user_question", height=140, placeholder="What would you like to know?")
if st.button("Get Answer", type="primary"):
    if not q.strip():
        st.info("Enter your question above to get started")
    else:
        try:
            res = rag.answer_query(q, openai_api_key=os.getenv("OPENAI_API_KEY"))
            st.session_state["last_answer"] = res
            _add_query_record(q)
        except Exception as e:
            st.error(f"Retrieval failed: {e}")

# Show answer/sources if present
res = st.session_state.get("last_answer")
if res:
    st.markdown("### Answer")
    st.write(res.get("answer", ""))
    st.markdown("### Sources")
    srcs = res.get("sources", [])
    if srcs:
        for s in srcs:
            st.markdown(f"- {s}")
    else:
        st.write("No sources.")
else:
    st.write("\n")

st.markdown("---")

# ---------- History ----------
st.markdown("### History")
htab1, htab2 = st.tabs(["Upload History", "Search History"])
with htab1:
    h = _read_history()
    uploads = h.get("uploads", [])
    if st.button("Clear Upload History"):
        h["uploads"] = []
        _write_history(h)
        uploads = []
    if uploads:
        for item in uploads:
            st.markdown(f"- {item['time']}: {item['type'].upper()} · {item['title']}")
    else:
        st.info("No uploads yet.")

with htab2:
    h = _read_history()
    queries = h.get("queries", [])
    if st.button("Clear Search History"):
        h["queries"] = []
        _write_history(h)
        queries = []
    if queries:
        for qh in queries:
            st.markdown(f"- {qh['time']}: {qh['question']}")
    else:
        st.info("No searches yet.")

st.caption("Vector DB: FAISS • Chunk size 800 • Overlap 150 • Top-K 4")
