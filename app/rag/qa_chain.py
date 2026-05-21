import os
import logging
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document
from app.rag.vectorstore import get_vectorstore
from app.rag.prompts import QA_PROMPT
from app.rag.citations import extract_citations
from app.utils.language_detector import detect_language
from app.utils.translator import translate_text

logger = logging.getLogger(__name__)

def get_llm() -> ChatGoogleGenerativeAI:
    """
    Initializes and returns the ChatGoogleGenerativeAI LLM.
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

def query_rag(question: str, target_lang: str = None, k: int = 4) -> dict:
    """
    Core RAG pipeline with Multilingual support:
    1. Detect language of the incoming question.
    2. Translate query to English if not in English.
    3. Retrieve top-k chunks from ChromaDB using English query.
    4. Call Gemini LLM in English.
    5. Detect missing context and return fallback.
    6. Extract citations in English.
    7. Translate the final answer back to the requested or detected language.
    """
    # Detect query language
    detected_lang = detect_language(question)
    
    # Determine target response language
    response_lang = target_lang if target_lang else detected_lang
    
    # Translate question to English if needed
    is_multilingual = detected_lang.lower() not in ["english", "en"]
    if is_multilingual:
        eng_question = translate_text(question, "English")
        logger.info(f"Translated query to English: {eng_question}")
    else:
        eng_question = question

    try:
        db = get_vectorstore()
        # Retrieve relevant chunks using similarity search with English query
        retrieved_docs = db.similarity_search(eng_question, k=k)
    except Exception as e:
        logger.error(f"Failed to retrieve from vector store: {str(e)}")
        retrieved_docs = []
        
    if not retrieved_docs:
        return {
            "answer": "Not found in the provided documents.",
            "citations": [],
            "confidence": 0.0,
            "raw_docs": []
        }
        
    # Format context
    context_str = ""
    for idx, doc in enumerate(retrieved_docs):
        context_str += f"--- Document Piece {idx+1} ---\n{doc.page_content}\n\n"
        
    try:
        llm = get_llm()
        # Assemble chain
        chain = QA_PROMPT | llm
        response = chain.invoke({
            "context": context_str,
            "question": eng_question
        })
        
        answer = response.content.strip()
    except Exception as e:
        logger.error(f"LLM generation error: {str(e)}")
        answer = "Not found in the provided documents."
        
    # Clean answer: check if it matches the no-hallucination phrase
    if "not found in the provided documents" in answer.lower():
        return {
            "answer": "Not found in the provided documents.",
            "citations": [],
            "confidence": 0.0,
            "raw_docs": []
        }
        
    # Extract citations in English (matching English answer to English chunks)
    citations = extract_citations(answer, retrieved_docs)
    
    # Translate the English answer to the final response language if needed
    final_answer = answer
    if response_lang.lower() not in ["english", "en"]:
        final_answer = translate_text(answer, response_lang)
        logger.info(f"Translated final answer to {response_lang}")
        
    return {
        "answer": final_answer,
        "citations": citations,
        "confidence": 0.9,
        "raw_docs": retrieved_docs
    }

