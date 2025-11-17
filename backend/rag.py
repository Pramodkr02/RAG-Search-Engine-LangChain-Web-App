"""RAG (Retrieval-Augmented Generation) utilities using LangChain.

Provides ingestion and query helpers that build on the LangChain
FAISS vectorstore and RetrievalQA chains. Functions return answers with
simple source citations extracted from document metadata.
"""
from typing import List, Dict, Optional, Any
import os

try:
    from langchain_groq import ChatGroq
except Exception:
    ChatGroq = None
from langchain_classic.chains import RetrievalQA
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate

from backend.embeddings import get_or_create_vectorstore, add_documents_to_vectorstore
from backend.chunker import split_text
from backend.loaders import document_from_text
from backend.config import DEFAULT_TOP_K, VECTOR_STORE_PATH
from backend.logger import root_logger


def ingest_text(title: str, text: str, source: str = "text", doc_id: Optional[str] = None) -> int:
    """Split `text` into chunks and add them to the FAISS vectorstore.

    Args:
        title: Title for the source (e.g., filename or URL).
        text: Raw text to ingest.
        source: Source type (pdf, url, youtube, resume, etc.).

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
        did = doc_id or f"{source}:{title}"
        for i, c in enumerate(chunks):
            docs.append(Document(page_content=c, metadata={"title": title, "source": source, "doc_id": did, "chunk": i}))
        add_documents_to_vectorstore(docs)
        logger.info("Ingested %d chunks from %s", len(docs), title)
        return len(docs)
    except Exception as exc:
        logger.exception("Failed to ingest text: %s", exc)
        raise


def _get_retriever(k: int = DEFAULT_TOP_K):
    vs = get_or_create_vectorstore()
    return vs.as_retriever(search_type="mmr", search_kwargs={"k": k, "fetch_k": max(10, k * 5)})


def _extractive_answer(query: str, docs: List[Document], max_sentences: int = 2) -> str:
    import re
    q = query.lower()
    tokens = [t for t in re.split(r"[^a-zA-Z0-9]+", q) if t]
    sentences_scored = []
    # Special case: questions like "capital of X"
    m = re.search(r"capital\s+of\s+([a-zA-Z]+)", q)
    target = m.group(1).lower() if m else None
    for d in docs:
        for s in re.split(r"(?<=[\.!?])\s+", d.page_content):
            s_clean = s.strip()
            if not s_clean:
                continue
            s_low = s_clean.lower()
            score = sum(1 for t in tokens if t and t in s_low)
            # Boost sentences that contain both the key relation and entity
            if target and ("capital" in s_low) and (target in s_low):
                score += 5
            sentences_scored.append((score, len(s_clean), s_clean))
    sentences_scored = [t for t in sentences_scored if t[0] > 0]
    # If we detected a specific entity in the question, prefer sentences mentioning it
    if target:
        filtered = [t for t in sentences_scored if target in t[2].lower()]
        if filtered:
            sentences_scored = filtered
    if not sentences_scored:
        return "No exact match found in context."
    sentences_scored.sort(key=lambda x: (-x[0], x[1]))
    picked = [s for _, _, s in sentences_scored[:max_sentences]]
    return " ".join(picked)


def answer_query(query: str, groq_api_key: Optional[str] = None, top_k: int = DEFAULT_TOP_K, doc_ids: Optional[List[str]] = None) -> Dict[str, Any]:
    """Answer a user query using a LangChain RetrievalQA chain with Groq LLM.

    Returns a dict with keys: `answer` (str) and `sources` (list).
    """
    logger = root_logger
    try:
        retriever = _get_retriever(k=top_k)

        if groq_api_key and ChatGroq is not None:
            try:
                llm = ChatGroq(temperature=0.0, model_name="mixtral-8x7b-32768")
                prompt = PromptTemplate(
                    template=(
                        "Use the context to answer the question concisely. "
                        "If the answer is not in the context, say you don't know.\n\n"
                        "Context:\n{context}\n\nQuestion: {question}\n\nAnswer (1-2 sentences):"
                    ),
                    input_variables=["context", "question"],
                )
                if hasattr(retriever, 'invoke'):
                    docs = retriever.invoke(query)
                else:
                    docs = retriever.get_relevant_documents(query)
                if doc_ids:
                    docs = [d for d in docs if (d.metadata or {}).get("doc_id") in doc_ids]
                else:
                    if docs:
                        primary_id = (docs[0].metadata or {}).get("doc_id")
                        if primary_id:
                            docs = [d for d in docs if (d.metadata or {}).get("doc_id") == primary_id]
                if not docs:
                    return {"answer": "No documents found. Ingest data first.", "sources": []}
                context = "\n\n".join([d.page_content for d in docs[:top_k]])
                prompt_text = prompt.format(context=context, question=query)
                resp = llm.invoke(prompt_text)
                answer_text = getattr(resp, 'content', str(resp))
                sources = []
                for d in docs[:top_k]:
                    md = d.metadata or {}
                    title = md.get("title") or md.get("source") or "unknown"
                    chunk = md.get("chunk")
                    sources.append(f"Source: {title} (chunk {chunk if chunk is not None else 'N/A'})")
                return {"answer": answer_text, "sources": sources}
            except Exception as e:
                # Fallback if Groq LLM fails
                logger.warning("Groq LLM call failed; falling back to non-LLM retrieval: %s", e)
                try:
                    if hasattr(retriever, 'invoke'):
                        docs = retriever.invoke(query)
                    else:
                        docs = retriever.get_relevant_documents(query)
                    if doc_ids:
                        docs = [d for d in docs if (d.metadata or {}).get("doc_id") in doc_ids]
                    else:
                        if docs:
                            primary_id = (docs[0].metadata or {}).get("doc_id")
                            if primary_id:
                                docs = [d for d in docs if (d.metadata or {}).get("doc_id") == primary_id]
                    if not docs:
                        return {"answer": "No documents found. Ingest data first.", "sources": []}
                    answer_text = _extractive_answer(query, docs[:top_k])
                    sources = []
                    for d in docs[:top_k]:
                        md = d.metadata or {}
                        title = md.get("title") or md.get("source") or "unknown"
                        chunk = md.get("chunk")
                        sources.append(f"Source: {title} (chunk {chunk if chunk is not None else 'N/A'})")
                    return {"answer": answer_text, "sources": sources}
                except Exception as fallback_error:
                    logger.error("Fallback retrieval failed: %s", fallback_error)
                    return {"answer": "Error retrieving relevant documents. Please try again.", "sources": []}
        else:
            # Fallback: perform retriever lookup and return concatenated chunks
            try:
                if hasattr(retriever, 'invoke'):
                    docs = retriever.invoke(query)
                else:
                    docs = retriever.get_relevant_documents(query)
                if doc_ids:
                    docs = [d for d in docs if (d.metadata or {}).get("doc_id") in doc_ids]
                if not docs:
                    return {"answer": "No documents found. Ingest data first.", "sources": []}
                answer_text = _extractive_answer(query, docs[:top_k])
                sources = []
                for d in docs[:top_k]:
                    md = d.metadata or {}
                    title = md.get("title") or md.get("source") or "unknown"
                    chunk = md.get("chunk")
                    sources.append(f"Source: {title} (chunk {chunk if chunk is not None else 'N/A'})")
                return {"answer": answer_text, "sources": sources}
            except Exception as e:
                logger.exception("Error in non-LLM retrieval")
                return {"answer": "An error occurred while retrieving documents.", "sources": []}
                
    except Exception as e:
        logger.exception("Unexpected error in answer_query")
        return {"answer": f"An unexpected error occurred: {str(e)}", "sources": []}
