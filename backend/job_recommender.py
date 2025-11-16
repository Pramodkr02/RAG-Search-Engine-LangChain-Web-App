"""Job recommender utilities using LangChain LLMs.

Provides a simple wrapper that returns recommended job roles based on
given skills. Uses OpenAI (ChatOpenAI) when a key is available and a
fallback heuristic otherwise.
"""
from typing import List, Optional
from backend.logger import root_logger

from langchain_openai import ChatOpenAI


def recommend(skills_text: str, openai_api_key: Optional[str] = None, max_roles: int = 5) -> str:
    """Recommend job roles based on a skills string.

    Args:
        skills_text: Free-text skills or resume snippet.
        openai_api_key: Optional OpenAI key.
        max_roles: Maximum number of roles to recommend.
    """
    logger = root_logger
    try:
        if openai_api_key:
            llm = ChatOpenAI(temperature=0.2)
            prompt = f"Given these skills, recommend {max_roles} suitable job roles and a 1-line reason for each:\n\n{skills_text}"
            return llm.predict(prompt)

        # Fallback heuristic
        s = skills_text.lower()
        suggestions: List[str] = []
        if any(k in s for k in ["python", "pandas", "numpy", "machine learning", "ml", "sklearn"]):
            suggestions.append("Data Scientist / ML Engineer — Python + ML libraries detected")
        if any(k in s for k in ["react", "javascript", "typescript", "node"]):
            suggestions.append("Frontend / Fullstack Developer — JS/React stack detected")
        if any(k in s for k in ["sql", "postgres", "mysql", "mongodb"]):
            suggestions.append("Backend / Database Engineer — DB skills detected")
        if any(k in s for k in ["docker", "kubernetes", "aws", "gcp"]):
            suggestions.append("DevOps / SRE — cloud + infra tooling detected")
        if not suggestions:
            suggestions.append("Software Engineer — general software skills detected")

        return "\n".join(suggestions[:max_roles])
    except Exception as exc:
        logger.exception("Job recommendation failed: %s", exc)
        return "(recommendation failed)"
