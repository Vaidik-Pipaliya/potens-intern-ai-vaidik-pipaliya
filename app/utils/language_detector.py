import logging
from langchain_core.prompts import PromptTemplate
from app.rag.llm import get_llm

logger = logging.getLogger(__name__)

DETECT_LANGUAGE_PROMPT = """Analyze the following text and determine its primary language.
Respond with ONLY the name of the language (e.g., "English", "Spanish", "French", "German", "Chinese", "Hindi", "Gujarati").
Do not include any punctuation, explanation, or additional words.

Text: {text}
Language:"""

def detect_language(text: str) -> str:
    """
    Detects the primary language of the given text using Gemini.
    Returns the language name (e.g., "English", "Spanish", etc.).
    """
    if not text or len(text.strip()) < 3:
        return "English"
        
    try:
        llm = get_llm()
        prompt = PromptTemplate.from_template(DETECT_LANGUAGE_PROMPT)
        chain = prompt | llm
        
        response = chain.invoke({"text": text})
        lang = response.content.strip().title()
        
        # Clean up any unexpected punctuation or extra text
        lang = "".join(c for c in lang if c.isalnum() or c.isspace()).strip()
        
        logger.info(f"Detected language for query: '{lang}'")
        return lang
    except Exception as e:
        logger.error(f"Error detecting language: {str(e)}")
        return "English"  # Fallback to English
