"""Loaders that produce LangChain Document objects (simple wrappers).

These functions keep simple dependencies (BeautifulSoup, PyPDF2,
YouTubeTranscriptApi) but return `langchain.schema.Document` objects so
they integrate smoothly with the rest of the LangChain pipeline.
"""
from typing import List
import re
import io
import requests
from bs4 import BeautifulSoup
try:
    from youtube_transcript_api import YouTubeTranscriptApi
except Exception:
    YouTubeTranscriptApi = None
from PyPDF2 import PdfReader

from langchain_core.documents import Document


def _extract_main_text(soup: BeautifulSoup) -> str:
    for tag in soup(["script", "style", "noscript", "svg"]):
        tag.decompose()
    for tag in soup(["header", "footer", "nav", "aside", "form", "iframe"]):
        tag.decompose()
    candidates = []
    for el in soup.find_all(["article", "main"]):
        txt = el.get_text("\n", strip=True)
        ps = len(el.find_all("p"))
        candidates.append((len(txt) + ps * 200, txt))
    if not candidates:
        for el in soup.find_all("div"):
            ps = el.find_all("p")
            if len(ps) >= 3:
                txt = el.get_text("\n", strip=True)
                candidates.append((len(txt) + len(ps) * 200, txt))
    text = ""
    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        text = candidates[0][1]
    else:
        text = soup.get_text("\n", strip=True)
    lines = [l.strip() for l in text.split("\n")]
    blacklist = {"home", "search", "login", "signup", "related articles", "copyright", "terms", "privacy"}
    filtered = []
    for l in lines:
        ll = l.lower()
        if any(b in ll for b in blacklist):
            continue
        if len(l) < 25:
            continue
        filtered.append(l)
    out = "\n".join(filtered) if filtered else text
    if len(out) > 12000:
        out = out[:12000]
    return out


def load_webpage(url: str) -> Document:
    """Fetch a webpage and return a LangChain Document with the page text.

    Args:
        url: The webpage URL to fetch.

    Returns:
        Document containing scraped text and metadata.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        text = _extract_main_text(soup)
        
        return Document(
            page_content=text,
            metadata={"source": url, "title": soup.title.string if soup.title else "Webpage", "type": "webpage"}
        )
    except Exception as e:
        raise Exception(f"Failed to load webpage {url}: {str(e)}")

# Update the __all__ list at the top of the file if it exists, or add it
__all__ = ['load_pdf_bytes', 'load_url_to_document', 'load_webpage', 'load_youtube_transcript', 'document_from_text']


def load_pdf_bytes(pdf_bytes: bytes, source: str = "pdf") -> Document:
    """Extract text from PDF bytes and return a LangChain Document.

    Args:
        pdf_bytes: Raw bytes of the PDF file.
        source: Optional source identifier (filename or URL).

    Returns:
        langchain Document with extracted text and metadata.
    """
    reader = PdfReader(io.BytesIO(pdf_bytes))
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return Document(page_content=text, metadata={"source": source, "type": "pdf"})


def load_url_to_document(url: str) -> Document:
    """Fetch a webpage and return a LangChain Document with the page text.

    Args:
        url: The webpage URL to fetch.

    Returns:
        Document containing scraped text and metadata.
    """
    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(url, headers=headers, timeout=15)
    resp.raise_for_status()
    html = resp.text
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = soup.get_text(" ")
    return Document(page_content=text, metadata={"source": url, "type": "webpage"})


def load_youtube_transcript(url: str) -> Document:
    """Get YouTube transcript text for a video URL and return Document.

    Args:
        url: YouTube video URL.

    Returns:
        Document with transcript text and metadata.
    """
    m = re.search(r"(?:v=|/)([0-9A-Za-z_-]{11})", url)
    if not m:
        raise ValueError("Invalid YouTube URL")
    vid = m.group(1)
    if YouTubeTranscriptApi is None:
        raise Exception("YouTubeTranscriptApi is not available")
    try:
        tr = YouTubeTranscriptApi.get_transcript(vid)
    except Exception:
        tr = YouTubeTranscriptApi.list_transcripts(vid).find_transcript(['en']).fetch()
    text = " ".join([t.get("text", "") for t in tr])
    return Document(page_content=text, metadata={"source": url, "type": "youtube"})


def document_from_text(text: str, source: str = "text") -> Document:
    """Wrap plain text into a LangChain Document.

    Args:
        text: Raw text.
        source: Optional source identifier.

    Returns:
        Document object.
    """
    return Document(page_content=text, metadata={"source": source, "type": "text"})
