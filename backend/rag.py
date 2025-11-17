"""RAG (Retrieval-Augmented Generation) utilities using LangChain.

Provides ingestion and query helpers that build on the LangChain
FAISS vectorstore and RetrievalQA chains. Functions return answers with
simple source citations extracted from document metadata.
"""
from typing import List, Dict, Optional, Any
import os

from langchain_openai import ChatOpenAI
from langchain_classic.chains import RetrievalQA
from langchain_core.documents import Document

from backend.embeddings import get_or_create_vectorstore, add_documents_to_vectorstore
from backend.chunker import split_text
from backend.loaders import document_from_text
from backend.config import DEFAULT_TOP_K, VECTOR_STORE_PATH
from backend.logger import root_logger


def ingest_text(title: str, text: str, source: str = "text", openai_api_key: Optional[str] = None) -> int:
    """Split `text` into chunks and add them to the FAISS vectorstore.

    Args:
        title: Title for the source (e.g., filename or URL).
        text: Raw text to ingest.
        source: Source type (pdf, url, youtube, resume, etc.).
        openai_api_key: Optional OpenAI key (selects embedding model).

    Returns:
        Number of chunks ingested.
    """
    logger = root_logger
    try:
        chunks = split_text(text)
        if not chunks:
            logger.warning("No content extracted for %s; skipping ingestion.", title)
            return 0
        docs: List[Document] = []
        for i, c in enumerate(chunks):
            docs.append(Document(page_content=c, metadata={"title": title, "source": source, "chunk": i}))
        add_documents_to_vectorstore(docs, openai_api_key=openai_api_key)
        logger.info("Ingested %d chunks from %s", len(docs), title)
        return len(docs)
    except Exception as exc:
        logger.exception("Failed to ingest text: %s", exc)
        raise


def _get_retriever(openai_api_key: Optional[str] = None, k: int = DEFAULT_TOP_K):
    vs = get_or_create_vectorstore(openai_api_key=openai_api_key)
    return vs.as_retriever(search_kwargs={"k": k})


def answer_query(query: str, openai_api_key: Optional[str] = None, top_k: int = DEFAULT_TOP_K) -> Dict[str, Any]:
    """Answer a user query using a LangChain RetrievalQA chain.

    Returns a dict with keys: `answer` (str) and `sources` (list).
    """
    logger = root_logger
    try:
        retriever = _get_retriever(openai_api_key=openai_api_key, k=top_k)

        if openai_api_key:
            try:
                llm = ChatOpenAI(temperature=0.0)
                qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
                resp = qa.run(query)
                # When using RetrievalQA.run we lose the individual docs; re-run a retrieve
                docs: List[Document] = retriever.get_relevant_documents(query)
                sources = []
                for d in docs:
                    md = d.metadata or {}
                    title = md.get("title") or md.get("source") or "unknown"
                    chunk = md.get("chunk")
                    sources.append(f"Source: {title} (chunk {chunk})")
                return {"answer": resp, "sources": sources}
            except Exception as e:
                # Fallback if OpenAI LLM fails (e.g., 429 insufficient_quota)
                logger.warning("LLM call failed; falling back to non-LLM retrieval: %s", e)
                docs: List[Document] = retriever.get_relevant_documents(query)
                if not docs:
                    return {"answer": "No documents found. Ingest data first.", "sources": []}
                concat = "\n\n".join([d.page_content for d in docs[:top_k]])
                sources = []
                for d in docs[:top_k]:
                    md = d.metadata or {}
                    title = md.get("title") or md.get("source") or "unknown"
                    chunk = md.get("chunk")
                    sources.append(f"Source: {title} (chunk {chunk})")
                return {"answer": concat, "sources": sources}
        else:
            # Fallback: perform retriever lookup and return concatenated chunks
            docs: List[Document] = retriever.get_relevant_documents(query)
            if not docs:
                return {"answer": "No documents found. Ingest data first.", "sources": []}
            concat = "\n\n".join([d.page_content for d in docs[:top_k]])
            sources = []
            for d in docs[:top_k]:
                md = d.metadata or {}
                title = md.get("title") or md.get("source") or "unknown"
                chunk = md.get("chunk")
                sources.append(f"Source: {title} (chunk {chunk})")
            return {"answer": concat, "sources": sources}
    except Exception as exc:
        logger.exception("RAG answer failed: %s", exc)
        raise
