import logging
from langchain_core.prompts import PromptTemplate
from app.rag.llm import get_llm, extract_text_from_response

logger = logging.getLogger(__name__)

TRANSLATE_PROMPT = """Translate the following text into {target_language}.
Maintain the original meaning, tone, and formatting. Do not add any commentary, explanations, or extra text. Only provide the direct translation.

Text: {text}
Translation:"""

def translate_text(text: str, target_language: str) -> str:
    """
    Translates text into the target language using the shared Groq LLM.
    """
    if not text or not target_language:
        return text
        
    # Check if target language is English or matches the source (case-insensitive check)
    if target_language.lower() in ["english", "en"]:
        # If it's already English or we are translating to English, and text looks like English, we can skip.
        # But this function is general, so it will perform translation if called.
        pass
        
    try:
        llm = get_llm()
        prompt = PromptTemplate.from_template(TRANSLATE_PROMPT)
        chain = prompt | llm
        
        response = chain.invoke({
            "text": text,
            "target_language": target_language
        })
        
        translation = extract_text_from_response(response.content).strip()
        logger.info(f"Successfully translated text to {target_language}")
        return translation
    except Exception as e:
        logger.error(f"Error during translation: {str(e)}")
        return text  # On translation failure, return source text so the pipeline still responds.
