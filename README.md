# RAG Question Answering (Streamlit)

A clean, end‑to‑end Retrieval‑Augmented Generation (RAG) web app built with Streamlit and LangChain. It matches the UI shown in the screenshots and focuses on exactly three ways to add knowledge: PDF, URL (websites or YouTube), and free Text. Includes History and a New session button.

- Vector store: FAISS (local, persisted)
- Embeddings: OpenAI if an API key is available; otherwise automatic fallback to local Sentence‑Transformers (`all‑MiniLM‑L6‑v2`)
- LLM answering: OpenAI Chat if available; otherwise retrieval‑only fallback (returns best matching context snippets)

---

## Features
- **PDF ingestion**: Upload a PDF; text is extracted, chunked, embedded, and stored in FAISS.
- **URL ingestion**: Provide a website URL or a YouTube link; the page text or transcript is ingested.
- **Text ingestion**: Paste raw text and ingest it.
- **Ask & Answer**: Large question box with a “Get Answer” button and source citations.
- **History**: Upload and Search history with Clear buttons.
- **New +**: Reset the current session quickly.

---

## Screenshots
> Replace with your actual images if needed
- sidebar, ingest states, and history sections are aligned to the screenshots you shared.

---

## Architecture
- `backend/loaders.py` — PDF (PyPDF2), Web (requests + BeautifulSoup), YouTube (YouTubeTranscriptApi) → returns LangChain `Document`s
- `backend/chunker.py` — `RecursiveCharacterTextSplitter` with configured chunk size/overlap
- `backend/embeddings.py` — selects embeddings provider (OpenAI or Sentence‑Transformers), manages FAISS creation/persistence, robust empty‑index handling, and automatic HF fallback on OpenAI 429
- `backend/rag.py` — ingestion helper `ingest_text(...)` and query helper `answer_query(...)` with LLM fallback to retrieval‑only
- `app.py` — Streamlit UI (PDF/URL/Text tabs, Ask box, New +, History)
- `backend/config.py` — central constants (chunk sizes, paths)

FAISS is persisted under `data/`. History is stored as JSON at `data/metadata.json`.

---

## Prerequisites
- Python 3.10+ (Windows/macOS/Linux)
- Optional: An OpenAI API key for LLM answers or OpenAI embeddings

---

## Setup (Windows PowerShell)
```powershell
# 1) Create and activate venv
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2) Install dependencies
pip install -r requirements.txt

# 3) (Optional) Create a .env file with your OpenAI key
#    If missing or quota is exceeded, the app automatically falls back to local embeddings/answers
New-Item -Path . -Name ".env" -ItemType "file" -Force | Out-Null
Add-Content .env "OPENAI_API_KEY=sk-REPLACE_ME"

# 4) Run the app
streamlit run app.py
```

> On macOS/Linux, use `source .venv/bin/activate` to activate the venv.

---

## Usage
1. Open the app in your browser (URL printed by Streamlit).
2. In the left sidebar under **Knowledge Base**, pick one of the tabs:
   - **PDF**: Drag & drop a PDF → click “Ingest PDF”.
   - **URL**: Paste a website or YouTube link → click “Ingest URL”.
   - **Text**: Paste text → click “Ingest Text”.
3. Ask a question in the large center box → click “Get Answer”.
4. See the **Answer** and **Sources** below.
5. Scroll to the **History** section to view and clear Upload/Search history.
6. Use **New +** to reset the current session state.

---

## Troubleshooting
- **429 insufficient_quota (OpenAI)**
  - The app automatically falls back to local Sentence‑Transformers embeddings for ingestion and to retrieval‑only answers for queries. You can also remove/leave the API key blank to force local mode.
- **PDF ingested but no chunks**
  - Some PDFs contain only images. Try OCR‑ed PDFs or use the URL/Text methods. The app will skip ingestion gracefully when no text is found.
- **FAISS index errors**
  - The app initializes an empty FAISS index safely; these should no longer occur. If you delete `data/` while the app is running, restart the app.
- **Networking blocked for URL ingestion**
  - Ensure the machine can reach the target website and YouTube transcript API.

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
└── .env                    (optional; OPENAI_API_KEY)
```

---

## Tech Stack
- Streamlit UI
- LangChain core + community integrations
- FAISS (faiss‑cpu) for vector search
- OpenAI (optional) and Sentence‑Transformers via `langchain‑huggingface`
- BeautifulSoup4, requests, PyPDF2, youtube‑transcript‑api

---

## Development Notes
- Style tweaks are inlined in `app.py` to match the screenshots (spacing, text sizes, success banner, etc.).
- The code purposely avoids extra features to keep the repo clean and the UX simple.

---

## License
Choose a license (e.g., MIT) and add it here.
