"""Streamlit app using LangChain-based RAG pipeline.

This app uses the `backend` package which exposes LangChain wrappers for
loading, chunking, embedding, and retrieval. It provides multiple tabs
for ingestion, chat, URL/YouTube ingestion, resume analysis, and job
recommendations.
"""

import os
from pathlib import Path
import json

import streamlit as st

from backend import loaders, rag, resume_analyzer, job_recommender
from backend.config import UPLOAD_DIR, DEFAULT_TOP_K
from backend.utils import ensure_dir
from backend.logger import root_logger
from backend.memory import get_memory


ensure_dir(UPLOAD_DIR)

st.set_page_config(page_title="LangChain RAG App", layout="wide")
logger = root_logger

# Sidebar
st.sidebar.title("Settings")
st.sidebar.markdown("Provide `OPENAI_API_KEY` (optional). If provided, the app will use OpenAI for LLMs.")
OPENAI_KEY = st.sidebar.text_input("OpenAI API Key", type="password")
if OPENAI_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_KEY

top_k = st.sidebar.slider("Retrieval Top-K", min_value=1, max_value=10, value=DEFAULT_TOP_K)
st.sidebar.markdown("---")

st.title("📚 LangChain RAG — Streamlit")
st.markdown("Upload sources, ingest text, ask questions (RAG), analyze resumes, and get job recommendations.")

tabs = st.tabs(["Ingest", "Chat (RAG)", "URL & YouTube", "Resume Analyzer", "Job Recommender", "Manage Files"]) 

# Ingest tab
with tabs[0]:
    st.header("Ingest: PDF / Text")
    st.write("Upload PDFs or paste text to ingest into the FAISS vector store.")

    uploaded_files = st.file_uploader("Upload PDF file(s)", type=["pdf"], accept_multiple_files=True)
    if uploaded_files:
        for f in uploaded_files:
            raw = f.read()
            try:
                doc = loaders.load_pdf_bytes(raw, source=f.name)
                n = rag.ingest_text(title=f.name, text=doc.page_content, source="pdf", openai_api_key=OPENAI_KEY or os.environ.get("OPENAI_API_KEY"))
                Path(UPLOAD_DIR).joinpath(f.name).write_bytes(raw)
                st.success(f"Ingested {n} chunks from {f.name}")
            except Exception as exc:
                st.error(f"Failed to ingest {f.name}: {exc}")

    st.write("---")
    st.subheader("Paste text to ingest")
    txt_title = st.text_input("Title for text", value="pasted_text")
    txt_area = st.text_area("Paste text here", height=200)
    if st.button("Ingest pasted text"):
        if not txt_area.strip():
            st.warning("Paste text to ingest.")
        else:
            n = rag.ingest_text(title=txt_title or "pasted_text", text=txt_area, source="text", openai_api_key=OPENAI_KEY or os.environ.get("OPENAI_API_KEY"))
            st.success(f"Ingested {n} chunks from pasted text.")

# Chat (RAG)
with tabs[1]:
    st.header("Ask questions (RAG)")
    memory = get_memory()
    query = st.text_input("Type your question here")
    if st.button("Retrieve & Answer"):
        if not query.strip():
            st.warning("Enter a question.")
        else:
            try:
                resp = rag.answer_query(query, openai_api_key=OPENAI_KEY or os.environ.get("OPENAI_API_KEY"), top_k=top_k)
                st.subheader("Answer")
                st.write(resp.get("answer"))
                st.subheader("Citations")
                for s in resp.get("sources", []):
                    st.markdown(f"- `{s}`")
            except Exception as exc:
                logger.exception("Chat failed: %s", exc)
                st.error(f"Retrieval or LLM failed: {exc}")

# URL & YouTube
with tabs[2]:
    st.header("URL scraping & YouTube transcript ingestion")
    url = st.text_input("Paste webpage URL (http/https)")
    if st.button("Fetch & Ingest URL"):
        if not url.strip():
            st.warning("Enter a URL.")
        else:
            try:
                doc = loaders.load_url_to_document(url.strip())
                n = rag.ingest_text(title=url.strip(), text=doc.page_content, source="url", openai_api_key=OPENAI_KEY or os.environ.get("OPENAI_API_KEY"))
                st.success(f"Ingested {n} chunks from URL.")
                st.write(doc.page_content[:1500] + ("..." if len(doc.page_content) > 1500 else ""))
            except Exception as exc:
                st.error(f"Failed to fetch/ingest URL: {exc}")

    st.write("---")
    st.subheader("Ingest YouTube transcript")
    yt_url = st.text_input("Paste YouTube URL")
    if st.button("Fetch Transcript & Ingest"):
        if not yt_url.strip():
            st.warning("Paste a YouTube URL.")
        else:
            try:
                doc = loaders.load_youtube_transcript(yt_url.strip())
                n = rag.ingest_text(title=yt_url.strip(), text=doc.page_content, source="youtube", openai_api_key=OPENAI_KEY or os.environ.get("OPENAI_API_KEY"))
                st.success(f"Ingested {n} chunks from YouTube transcript.")
                st.write(doc.page_content[:1500] + ("..." if len(doc.page_content) > 1500 else ""))
            except Exception as exc:
                st.error(f"Failed to fetch transcript: {exc}")

# Resume Analyzer
with tabs[3]:
    st.header("Resume Analyzer")
    uploaded_resume = st.file_uploader("Upload Resume PDF", type=["pdf"]) 
    skills_input = st.text_area("Enter skills to check (comma-separated)", placeholder="Python, SQL, Machine Learning")
    if st.button("Analyze Resume"):
        if not uploaded_resume:
            st.warning("Upload a resume PDF first.")
        else:
            try:
                raw = uploaded_resume.read()
                doc = loaders.load_pdf_bytes(raw, source=uploaded_resume.name)
                skills = [s.strip() for s in skills_input.split(",") if s.strip()]
                pct, details = resume_analyzer.skill_match(doc.page_content, skills)
                st.metric("Skill match %", f"{pct:.1f}%")
                st.json(details)
                st.write("---")
                summary = resume_analyzer.summarize_resume(doc.page_content, openai_api_key=OPENAI_KEY or os.environ.get("OPENAI_API_KEY"))
                st.subheader("Resume summary")
                st.write(summary)
                if st.button("Ingest this resume into vector store"):
                    n = rag.ingest_text(title=uploaded_resume.name, text=doc.page_content, source="resume", openai_api_key=OPENAI_KEY or os.environ.get("OPENAI_API_KEY"))
                    st.success(f"Ingested {n} chunks from resume.")
            except Exception as exc:
                st.error(f"Resume analysis failed: {exc}")

# Job Recommender
with tabs[4]:
    st.header("Job Role Recommendation")
    skills_text = st.text_area("Enter skills or paste resume text", height=160)
    num_roles = st.number_input("Max roles to recommend", min_value=1, max_value=12, value=6)
    if st.button("Recommend Job Roles"):
        if not skills_text.strip():
            st.warning("Enter skills or resume text.")
        else:
            try:
                recs = job_recommender.recommend(skills_text, openai_api_key=OPENAI_KEY or os.environ.get("OPENAI_API_KEY"), max_roles=num_roles)
                st.subheader("Recommendations")
                st.write(recs)
            except Exception as exc:
                st.error(f"Recommendation failed: {exc}")

# Manage Files
with tabs[5]:
    st.header("Uploaded files")
    upload_dir = Path(UPLOAD_DIR)
    files = list(upload_dir.glob("*"))
    if files:
        st.write("Files saved to `data/uploads/`:")
        for f in files:
            st.markdown(f"- `{f.name}` ({f.stat().st_size} bytes)")
            if st.button(f"Delete {f.name}"):
                try:
                    f.unlink()
                    st.success(f"Deleted {f.name}")
                    st.experimental_rerun()
                except Exception as exc:
                    st.error(f"Failed to delete: {exc}")
    else:
        st.info("No uploaded files saved yet.")

st.markdown("---")
st.caption("LangChain RAG — embeddings + FAISS, optional OpenAI. See README for setup and notes.")
