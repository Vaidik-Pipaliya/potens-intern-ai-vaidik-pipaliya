import logging
from langdetect import detect

logger = logging.getLogger(__name__)

def detect_language(text: str) -> str:
    """
    Detects the primary language of the given text using langdetect and heuristics.
    Returns the language name (e.g., "English", "Spanish", etc.).
    """
    if not text or len(text.strip()) < 3:
        return "English"
        
    # Check if text is Hinglish or contains typical Hinglish words or uses Hindi script.
    # Hindi script (Devanagari) has character range \u0900-\u097f
    if any("\u0900" <= char <= "\u097f" for char in text):
        logger.info("Detected Devanagari script. Setting language to Hindi.")
        return "Hindi"
        
    try:
        lang_code = detect(text)
        logger.info(f"langdetect returned code: '{lang_code}' for text: '{text[:20]}...'")
        
        # Map lang_code to full language name
        mapping = {
            "en": "English",
            "es": "Spanish",
            "hi": "Hindi",
            "gu": "Gujarati",
            "fr": "French",
            "de": "German",
            "zh-cn": "Chinese",
            "zh-tw": "Chinese",
            "it": "Italian",
            "pt": "Portuguese",
            "ja": "Japanese",
            "ko": "Korean",
            "ru": "Russian",
        }
        
        detected = mapping.get(lang_code, "English")
        
        # Romanized Hindi (Hinglish) often scores as English in langdetect — route to Hindi translation.
        hinglish_words = {"ka", "ki", "ko", "ke", "hai", "hain", "aur", "kitna", "kab", "kya", "dur", "kabse", "hoga"}
        words = set(text.lower().split())
        if words.intersection(hinglish_words):
            logger.info("Detected Hinglish/Hindi transliteration words. Setting language to Hindi.")
            return "Hindi"
            
        logger.info(f"Detected language for query: '{detected}'")
        return detected
    except Exception as e:
        logger.error(f"Error detecting language with langdetect: {str(e)}")
        return "English"  # Fallback to English

