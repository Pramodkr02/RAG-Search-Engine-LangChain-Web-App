# LangChain RAG — Mini Project

This repository refactors an existing Retrieval-Augmented Generation
(RAG) project into a clean, modular implementation built on
LangChain and Streamlit.

**Features**

- LangChain-based loaders, text-splitting, embeddings, and FAISS vector
  store
- RetrievalQA chain using OpenAI Chat models (optional)
- Streamlit UI with tabs for ingestion, chat, URL/YouTube ingestion,
  resume analysis, job recommendations, and file management
- Logging, type hints, and docstrings throughout the backend
- Simple persistence for FAISS vectorstore

**Quick start**

1. Create a virtual environment and activate it.

On Windows (PowerShell):

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. (Optional) Copy `.env.example` to `.env` and set `OPENAI_API_KEY`.

3. Run the Streamlit app:

```powershell
streamlit run app.py
```

**Project structure**

- `app.py` — Streamlit UI
- `backend/` — modular LangChain wrappers
  - `loaders.py` — webpage, PDF, YouTube loaders returning `Document`
  - `chunker.py` — LangChain `RecursiveCharacterTextSplitter` wrapper
  - `embeddings.py` — embeddings + FAISS helpers
  - `rag.py` — ingestion + RetrievalQA helpers
  - `resume_analyzer.py` — resume matching and summarization
  - `job_recommender.py` — simple role recommender
  - `logger.py` — logging setup
  - `memory.py` — conversation memory helper
- `data/` — uploads, vector store, logs
- `requirements.txt` — Python dependencies
- `.env.example` — example environment variables
- `APPROACH_NOTE.md` — design and rationale

**How the LangChain pipeline works**

1. Load raw input into LangChain `Document` objects with `backend.loaders`.
2. Chunk long documents with `backend.chunker.split_text`.
3. Encode chunks into vectors via `backend.embeddings.get_embedding_model`.
4. Store embeddings in a FAISS vectorstore and persist.
5. Query with `backend.rag.answer_query` which uses a retriever and
   `RetrievalQA` (optional LLM) to produce answers with citations.

**Assignment expectations**

This repo implements the required LangChain components: loaders,
text-splitter, embeddings, FAISS vector store, retrieval chain, and an
LLM wrapper. The Streamlit UI lets you ingest sources and ask
retrieval-augmented questions. Citations are shown next to answers.

**Future improvements**

- Better persistence, reindexing, and large-scale vector DB support
- Unit tests and CI
- Advanced UI with highlighted source passages and chat history

Screenshots: (placeholders)

![screenshot-1](screenshots/screen1.png)
![screenshot-2](screenshots/screen2.png)
