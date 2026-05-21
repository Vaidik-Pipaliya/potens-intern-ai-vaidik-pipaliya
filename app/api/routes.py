from fastapi import APIRouter, HTTPException
from app.api.schemas import AskRequest, AskResponse, ContradictRequest, ContradictResponse
from app.rag.qa_chain import query_rag
from app.rag.contradiction import analyze_contradiction

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

@router.post("/contradict", response_model=ContradictResponse)
def contradict_documents_endpoint(request: ContradictRequest):
    """
    Compares two ingested documents on a specific topic to analyze potential contradictions.
    """
    try:
        result = analyze_contradiction(
            doc1_id=request.doc1_id,
            doc2_id=request.doc2_id,
            topic=request.topic
        )
        return ContradictResponse(
            contradiction_found=result["contradiction_found"],
            reasoning=result["reasoning"],
            evidence_doc1=result["evidence_doc1"],
            evidence_doc2=result["evidence_doc2"]
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while analyzing contradictions: {str(e)}"
        )

