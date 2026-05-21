import os
from langchain_groq import ChatGroq

def get_llm() -> ChatGroq:
    """
    Initializes and returns the ChatGroq LLM.
    Requires GROQ_API_KEY to be set in environment variables.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY is not set.")
    return ChatGroq(
        model="llama-3.1-8b-instant", 
        temperature=0.0, 
        groq_api_key=api_key
    )

def extract_text_from_response(content) -> str:
    """
    Safely extracts string content from LLM response content.
    Handles string, list of dicts, or other formats returned by the Google GenAI SDK.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = []
        for part in content:
            if isinstance(part, str):
                text_parts.append(part)
            elif isinstance(part, dict) and "text" in part:
                text_parts.append(part["text"])
        return "".join(text_parts)
    return str(content)
