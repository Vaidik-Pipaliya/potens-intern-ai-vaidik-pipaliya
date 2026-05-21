import os
import logging
from langchain_core.documents import Document
from app.rag.vectorstore import get_vectorstore
from app.rag.prompts import QA_PROMPT
from app.rag.citations import extract_citations
from app.utils.language_detector import detect_language
from app.utils.translator import translate_text
from app.rag.llm import get_llm, extract_text_from_response

logger = logging.getLogger(__name__)



def query_rag(question: str, target_lang: str = None, k: int = 4) -> dict:
    """
    Core RAG pipeline with Multilingual support:
    1. Detect language of the incoming question.
    2. Translate query to English if not in English.
    3. Retrieve top-k chunks from ChromaDB using English query.
    4. Call Groq LLM in English.
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
        
    # Piece indices in the prompt align 1:1 with retrieval order for [Piece N] citation tags.
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
        
        answer = extract_text_from_response(response.content).strip()
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
        
    # Citations must be resolved on the English answer before translation (chunks are English).
    citations = extract_citations(answer, retrieved_docs)
    
    # Tags are for attribution only; strip before user-facing text.
    import re
    cleaned_answer = re.sub(r'\[Piece\s*\d+\]', '', answer).strip()
    # Normalize spaces
    cleaned_answer = re.sub(r'\s+', ' ', cleaned_answer)
    # Remove space before punctuation (e.g. "focused day ." -> "focused day.")
    cleaned_answer = re.sub(r'\s+([.,!?])', r'\1', cleaned_answer)
    
    # Translate the English answer to the final response language if needed
    final_answer = cleaned_answer
    if response_lang.lower() not in ["english", "en"]:
        final_answer = translate_text(cleaned_answer, response_lang)
        logger.info(f"Translated final answer to {response_lang}")
        
    return {
        "answer": final_answer,
        "citations": citations,
        "confidence": 0.9,
        "raw_docs": retrieved_docs
    }

