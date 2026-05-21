from pydantic import BaseModel, Field
from typing import List, Optional

class AskRequest(BaseModel):
    question: str = Field(..., description="The query to ask the document database.")
    language: Optional[str] = Field(None, description="Optional target language for multilingual querying.")

class Citation(BaseModel):
    file: str = Field(..., description="Source filename")
    page: int = Field(..., description="Page number of document (1-indexed)")
    chunk_id: int = Field(..., description="Unique chunk id")
    snippet: str = Field(..., description="Exact snippet from document source")

class AskResponse(BaseModel):
    answer: str = Field(..., description="Generated answer from documents")
    citations: List[Citation] = Field(default=[], description="List of source citations used to form the answer")
    confidence: float = Field(..., description="Confidence score between 0.0 and 1.0")
