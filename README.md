# RAG Question Answering (Streamlit)

An end‑to‑end Retrieval‑Augmented Generation (RAG) app built with Streamlit and LangChain. Add knowledge via PDF, URL (websites or YouTube), or free Text, then ask questions and get concise answers with source citations. Includes History, a visible New button, and precise source scoping.

- Vector store: FAISS (local, persisted)
- Embeddings: Sentence‑Transformers (`all‑MiniLM‑L6‑v2`)
- LLM answering: Groq (optional) via `GROQ_API_KEY`; otherwise concise extractive retrieval

---

## Features
- PDF/URL/Text ingestion with chunking and stored embeddings
- Concise answers with citations
- History: upload and search logs with clear actions
- New button: resets session state and UI
- Answer scope: choose “All sources” or select specific uploads to avoid mixing

---

## Architecture
- `backend/loaders.py` — PDF (PyPDF2), Web (requests + BeautifulSoup), YouTube (youtube‑transcript‑api) → returns LangChain `Document`s
- `backend/chunker.py` — `RecursiveCharacterTextSplitter` (default `chunk_size=500`, `chunk_overlap=50`)
- `backend/embeddings.py` — HuggingFace embeddings + FAISS creation/persistence with safe empty‑index handling
- `backend/rag.py` — ingestion (`ingest_text`) tags each chunk with a unique `doc_id`; query (`answer_query`) uses MMR retrieval and returns concise answers; respects selected `doc_ids`
- `app.py` — Streamlit UI with PDF/URL/Text tabs, Answer scope selector, Ask box, New button, History
- `backend/config.py` — central constants (chunk sizes, paths)

FAISS lives at `data/vector_store.faiss`. History is stored at `data/metadata.json`.

---

## Prerequisites
- Python 3.10+
- Optional: Groq account and `GROQ_API_KEY` for LLM answers

---

## Setup (Windows PowerShell)
```powershell
# 1) Create and activate venv
python -m venv .venv
\.\.venv\Scripts\Activate.ps1

# 2) Install dependencies
pip install -r requirements.txt

# 3) (Optional) Create a .env file with your Groq key
New-Item -Path . -Name ".env" -ItemType "file" -Force | Out-Null
Add-Content .env "GROQ_API_KEY=sk-REPLACE_ME"

# 4) Run the app
streamlit run app.py
```

On macOS/Linux, activate with `source .venv/bin/activate`.

---

## Usage
1. Open the app URL printed by Streamlit.
2. In the sidebar, choose an ingestion method:
   - PDF: upload and click “Ingest PDF”
   - URL: paste website or YouTube link and click “Ingest URL”
   - Text: paste and click “Ingest Text”
3. Choose Answer scope above the question box:
   - All sources: auto‑focus on the top matching upload
   - Selected sources: pick specific uploads; answers use only those
4. Ask your question and click “Get Answer”. View concise answer and Sources.
5. Use the New button to reset session state.
6. Review the History tabs to see/correct your ingest/search activity.

---

## Configuration
- Defaults in `backend/config.py`:
  - `CHUNK_SIZE=500`, `CHUNK_OVERLAP=50`
  - `VECTOR_STORE_PATH='data/vector_store.faiss'`
  - `METADATA_PATH='data/metadata.json'`
  - `DEFAULT_TOP_K=4`

---

## Troubleshooting
- Groq not available
  - Leave `GROQ_API_KEY` empty; the app returns concise extractive answers from retrieved chunks.
- YouTube transcript issues
  - Ensure `youtube-transcript-api` is installed and the video has transcripts; the loader falls back to transcript listing when direct fetch fails.
- No text extracted from PDF
  - Some PDFs are image‑only. Use OCR‑ed PDFs or ingest via URL/Text.
- FAISS index missing/corrupted
  - The app initializes a safe empty FAISS index; if you delete `data/` during runtime, restart the app.

---

## Project Structure
```
├── app.py
├── backend
│   ├── __init__.py
│   ├── chunker.py
│   ├── config.py
│   ├── embeddings.py
│   ├── loaders.py
│   ├── memory.py
│   ├── rag.py
│   ├── logger.py
├── data
│   ├── vector_store.faiss  (created at runtime)
│   └── metadata.json       (history)
├── requirements.txt
└── .env                    (optional; GROQ_API_KEY)
```

---

## Tech Stack
- Streamlit
- LangChain core + community integrations
- FAISS (faiss‑cpu) for vector search
- Sentence‑Transformers via `langchain-huggingface`
- Groq (optional) via `langchain-groq`
- BeautifulSoup4, requests, PyPDF2, youtube‑transcript‑api, python‑dotenv

---

## License
Add a license (e.g., MIT) if you plan to distribute.
