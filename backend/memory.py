"""Conversation memory wrapper using LangChain's ConversationBufferMemory.

This module exports a factory function to get a memory instance that can
be reused by the Streamlit app to keep short-term chat history.
"""
from typing import Optional, Any, Dict, List


class SimpleBufferMemory:
    """Simple in-memory buffer for conversation history."""
    
    def __init__(self, memory_key: str = "chat_history", return_messages: bool = True):
        self.memory_key = memory_key
        self.return_messages = return_messages
        self.history: List[Dict[str, Any]] = []
    
    def save_context(self, inputs: Dict[str, str], outputs: Dict[str, str]) -> None:
        """Save context from this conversation to buffer."""
        self.history.append({"input": inputs, "output": outputs})
    
    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Return memory variables."""
        return {self.memory_key: self.history}
    
    def clear(self) -> None:
        """Clear memory."""
        self.history = []


def get_memory(key: str = "chat_history") -> SimpleBufferMemory:
    """Return a memory instance.

    Args:
        key: The memory key under which chat history is stored.
    """
    return SimpleBufferMemory(memory_key=key, return_messages=True)
