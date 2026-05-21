from fastapi import APIRouter, HTTPException
from app.api.schemas import AskRequest, AskResponse
from app.rag.qa_chain import query_rag

router = APIRouter()

@router.post("/ask", response_model=AskResponse)
def ask_question_endpoint(request: AskRequest):
    """
    Retrieves information from ingested documents and generates a citation-backed response.
    """
    try:
        # Run RAG query
        result = query_rag(request.question)
        
        # Format response to match API contract
        return AskResponse(
            answer=result["answer"],
            citations=result["citations"],
            confidence=result["confidence"]
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while processing the request: {str(e)}"
        )
