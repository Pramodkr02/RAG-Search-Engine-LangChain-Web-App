"""Resume analysis utilities with optional LangChain LLM enhancements."""
from typing import Dict, List, Tuple, Optional
from backend.logger import root_logger

from langchain_openai import ChatOpenAI


def skill_match(resume_text: str, skills: List[str]) -> Tuple[float, Dict[str, bool]]:
    """Simple keyword-based skill matching.

    Args:
        resume_text: Raw resume text.
        skills: List of skills to search for.

    Returns:
        Tuple of (percentage_match, mapping skill->present).
    """
    resume = resume_text.lower()
    found: Dict[str, bool] = {}
    matched = 0

    for skill in skills:
        sk = skill.lower().strip()
        present = sk in resume
        found[sk] = present
        if present:
            matched += 1

    percentage = (matched / len(skills)) * 100 if skills else 0.0
    return percentage, found


def summarize_resume(resume_text: str, openai_api_key: Optional[str] = None) -> str:
    """Return a short summary of the resume. Uses OpenAI if key provided.

    Args:
        resume_text: Raw resume text.
        openai_api_key: Optional OpenAI key.
    """
    logger = root_logger
    try:
        if openai_api_key:
            llm = ChatOpenAI(temperature=0.0)
            prompt = f"Summarize the resume and list 5 strengths and 5 improvement areas:\n\n{resume_text}"
            resp = llm.predict(prompt)
            return resp
        # fallback: naive extractive summary
        sentences = resume_text.split(". ")
        return ". ".join(sentences[:8]) + ("..." if len(sentences) > 8 else "")
    except Exception as exc:
        logger.exception("Failed to summarize resume: %s", exc)
        return "(resume summarization failed)"
