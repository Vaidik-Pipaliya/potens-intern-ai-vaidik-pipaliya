import os
from langchain_google_genai import ChatGoogleGenerativeAI

def get_llm() -> ChatGoogleGenerativeAI:
    """
    Initializes and returns the ChatGoogleGenerativeAI LLM.
    Requires GEMINI_API_KEY to be set in environment variables.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set.")
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash", 
        temperature=0.0, 
        google_api_key=api_key
    )
