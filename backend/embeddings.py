"""Embeddings and FAISS vectorstore utilities using LangChain.

This module provides helper functions to create or load a FAISS
vectorstore that is compatible with LangChain workflows using
Sentence-Transformers via `HuggingFaceEmbeddings`.
"""
from typing import List, Optional, Tuple
import os

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore import InMemoryDocstore
from langchain_core.documents import Document

import faiss  # faiss-cpu

from backend.config import EMBEDDING_MODEL_NAME, VECTOR_STORE_PATH
from backend.utils import ensure_dir


def get_embedding_model():
    """Return a HuggingFace embeddings object.
    """
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)


def create_faiss_from_documents(docs: List[Document], embedding_model) -> FAISS:
    """Create a FAISS vectorstore from LangChain Documents.

    Args:
        docs: List of Documents to index.
        embedding_model: an embeddings object from LangChain.

    Returns:
        FAISS vectorstore instance.
    """
    texts = [d.page_content for d in docs]
    metadatas = [d.metadata or {} for d in docs]
    vs = FAISS.from_texts(texts, embedding_model, metadatas=metadatas)
    return vs


def load_vectorstore(path: str) -> Optional[FAISS]:
    """Load a persisted FAISS store if present.

    Returns None if not found.
    """
    if not os.path.exists(path):
        return None
    return FAISS.load_local(path, embeddings=get_embedding_model())


def persist_vectorstore(vs: FAISS, path: str) -> None:
    """Persist FAISS vectorstore to `path` (a directory).

    Args:
        vs: FAISS object
        path: directory path where files will be saved
    """
    ensure_dir(os.path.dirname(path) or "./")
    vs.save_local(path)


# Module-level convenience: in-memory current vectorstore (may be None)
CURRENT_VS: Optional[FAISS] = None


def _empty_faiss(emb) -> FAISS:
    """Create an empty FAISS vector store safely.

    We cannot call FAISS.from_texts([]) because it raises an index error.
    Instead, build an IndexFlatL2 using the embedding dimensionality.
    """
    dim = len(emb.embed_query(""))  # get embedding dimension
    index = faiss.IndexFlatL2(dim)
    return FAISS(embedding_function=emb, index=index, docstore=InMemoryDocstore({}), index_to_docstore_id={})


def get_or_create_vectorstore(docs: Optional[List[Document]] = None) -> FAISS:
    """Return a FAISS vectorstore. Load from disk if possible, otherwise
    create from `docs`.

    Args:
        docs: Optional list of Documents to initialize the store with.
    """
    global CURRENT_VS
    emb = get_embedding_model()
    if CURRENT_VS is None:
        # attempt load from disk
        try:
            vs = load_vectorstore(VECTOR_STORE_PATH)
            if vs:
                CURRENT_VS = vs
        except Exception:
            CURRENT_VS = None

    if CURRENT_VS is None:
        if not docs:
            CURRENT_VS = _empty_faiss(emb)
        else:
            CURRENT_VS = create_faiss_from_documents(docs, emb)
    return CURRENT_VS


def add_documents_to_vectorstore(docs: List[Document]):
    """Add documents to the global vectorstore.
    """
    global CURRENT_VS
    if not docs:
        return

    emb = get_embedding_model()
    if CURRENT_VS is None:
        CURRENT_VS = _empty_faiss(emb)
    CURRENT_VS.add_documents(docs)

    try:
        persist_vectorstore(CURRENT_VS, VECTOR_STORE_PATH)
    except Exception:
        # best-effort persistence; do not raise for UI
        pass
