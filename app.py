"""RAG Question Answering (Streamlit)

Features (exactly as requested):
- Left sidebar Knowledge Base with three tabs: PDF, URL (incl. YouTube/website), Text
- Center: big question box with Get Answer
- Top-right: New + button to reset session
- History section: upload and query history with clear buttons

Back end uses FAISS vector store via LangChain with Groq LLM.
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
if os.getenv("GROQ_API_KEY"):
    os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")  # ensure for langchain-groq

st.set_page_config(page_title="RAG Question Answering", layout="wide")
st.markdown(
    """
    <style>
      .block-container{padding-top:2.5rem;}
      .stTextArea textarea{font-size:1.05rem;}
      .hint{color:#c8c8c8;font-size:0.92rem;}
      .ok{background:#16321f;color:#b9f5d0;padding:10px 12px;border-radius:6px;}
      .stButton > button { z-index: 10; }
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


def _add_upload_record(kind: str, title: str, doc_id: str):
    h = _read_history()
    h["uploads"].insert(0, {"time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "type": kind, "title": title, "doc_id": doc_id})
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
                did = f"pdf:{pdf.name}"
                chunks = rag.ingest_text(title=pdf.name, text=doc.page_content, source="pdf", doc_id=did)
                st.markdown('<div class="ok">PDF ingested successfully!</div>', unsafe_allow_html=True)
                _add_upload_record("pdf", pdf.name, did)
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
                    doc = loaders.load_webpage(url)
                    title = doc.metadata.get("title", url)
                    kind = "webpage"
                
                # Ingest the content
                did = f"{kind}:{url}"
                chunks = rag.ingest_text(title=title, text=doc.page_content, source=kind, doc_id=did)
                st.markdown(f'<div class="ok">URL ingested successfully! ({chunks} chunks)</div>', unsafe_allow_html=True)
                _add_upload_record(kind, title, did)
                
            except Exception as e:
                st.error(f"Failed to ingest URL: {e}")
    st.caption("Supports most webpages and YouTube videos")

# Text tab
with kb_tab[2]:
    st.markdown("#### Paste Text")
    raw_text = st.text_area("Paste your text here...", height=180)
    if st.button("Ingest Text", use_container_width=True):
        if not raw_text.strip():
            st.warning("Enter some text to ingest.")
        else:
            try:
                did = f"text:{datetime.now().strftime('%Y%m%d%H%M%S')}"
                rag.ingest_text(title="text", text=raw_text, source="text", doc_id=did)
                st.markdown('<div class="ok">Text ingested successfully!</div>', unsafe_allow_html=True)
                _add_upload_record("text", "pasted text", did)
            except Exception as e:
                st.error(f"Failed to ingest text: {e}")

# API Key
with st.sidebar.expander("API Settings"):
    groq_key = st.text_input("Groq API Key", type="password", value=os.getenv("GROQ_API_KEY", ""))
    if groq_key:
        os.environ["GROQ_API_KEY"] = groq_key

# ---------- Header with New + button ----------
left, right = st.columns([0.8, 0.2])
with left:
    st.markdown("## RAG Question Answering")
with right:
    if st.button("New", type="secondary", use_container_width=True):
        st.session_state.pop("user_question", None)
        st.session_state.pop("last_answer", None)
        st.session_state.pop("chat_history", None)

# ---------- Ask box ----------
hstate = st.session_state.get("chat_history")
if hstate is None:
    st.session_state["chat_history"] = []

scope_choice = st.radio("Answer scope", ["All sources", "Selected sources"], index=0)
h = _read_history()
uploads = h.get("uploads", [])
options = [f"{u['type'].upper()} · {u['title']}" for u in uploads]
id_map = {f"{u['type'].upper()} · {u['title']}": u.get('doc_id') for u in uploads}
selected_labels = []
if scope_choice == "Selected sources":
    selected_labels = st.multiselect("Choose sources", options, help="Answers will use only selected sources")
selected_doc_ids = [id_map[l] for l in selected_labels]

question = st.text_area(
    "Ask a question about the documents in your knowledge base:",
    placeholder="Type your question here...",
    label_visibility="collapsed",
    height=100,
)

if st.button("Get Answer", type="primary", use_container_width=True):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Thinking..."):
            try:
                result = rag.answer_query(
                    question,
                    groq_api_key=os.getenv("GROQ_API_KEY"),
                    top_k=3,
                    doc_ids=(selected_doc_ids if scope_choice == "Selected sources" else None),
                    history=st.session_state.get("chat_history")
                )
                st.markdown("### Answer")
                st.markdown(result["answer"])
                if result["sources"]:
                    with st.expander("Sources"):
                        for src in result["sources"]:
                            st.markdown(f"- {src}")
                _add_query_record(question)
                st.session_state["chat_history"].append({"question": question, "answer": result.get("answer", "")})
            except Exception as e:
                st.error(f"Error getting answer: {e}")

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

st.caption("Vector DB: FAISS • Chunk size 500 • Overlap 50 • Top-K 4")
