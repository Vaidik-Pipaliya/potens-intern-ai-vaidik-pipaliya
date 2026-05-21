import os
import logging
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document
from app.rag.vectorstore import get_vectorstore
from app.rag.prompts import QA_PROMPT

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

def query_rag(question: str, k: int = 4) -> dict:
    """
    Core RAG pipeline:
    1. Retrieve top-k chunks from ChromaDB.
    2. Format context for prompt.
    3. Call Gemini LLM with strict no-hallucination constraint.
    4. Handle fallback if not found.
    """
    try:
        db = get_vectorstore()
        # Retrieve relevant chunks using similarity search
        retrieved_docs = db.similarity_search(question, k=k)
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
            "question": question
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
        
    return {
        "answer": answer,
        "citations": [], # Citation formatting will be implemented in Phase 6
        "confidence": 0.85, # Basic confidence for now
        "raw_docs": retrieved_docs
    }
