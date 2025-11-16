"""Text splitting utilities using LangChain text splitters."""
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.config import CHUNK_SIZE, CHUNK_OVERLAP


def split_text(text: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split a raw text string into chunks using LangChain's
    RecursiveCharacterTextSplitter.

    Args:
        text: The input text to split.
        chunk_size: Desired chunk size in characters.
        chunk_overlap: Overlap between chunks in characters.

    Returns:
        A list of text chunks.
    """
    text = text.replace("\r\n", "\n")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " "]
    )
    return splitter.split_text(text)
