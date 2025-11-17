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
from youtube_transcript_api import YouTubeTranscriptApi
from PyPDF2 import PdfReader

from langchain_core.documents import Document


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
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
            
        # Get text
        text = soup.get_text(separator='\n', strip=True)
        
        # Clean up multiple newlines
        text = '\n'.join([line for line in text.split('\n') if line.strip()])
        
        return Document(
            page_content=text,
            metadata={
                "source": url,
                "title": soup.title.string if soup.title else "Webpage",
                "type": "webpage"
            }
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
    video_id = m.group(1)
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    text = " ".join([t["text"] for t in transcript])
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
