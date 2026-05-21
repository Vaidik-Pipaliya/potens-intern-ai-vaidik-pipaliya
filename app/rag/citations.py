import re
from langchain_core.documents import Document

def split_into_sentences(text: str) -> list[str]:
    """
    Splits text into a list of sentences, ignoring very short fragments.
    """
    # Split using punctuation boundaries followed by whitespace
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if len(s.strip()) > 10]

def extract_citations(answer: str, retrieved_docs: list[Document]) -> list[dict]:
    """
    Analyzes the generated answer and maps its sentences back to the retrieved 
    document chunks to create accurate citations.
    
    For each sentence in the answer, it checks if it exists (case-insensitive) 
    in any of the retrieved document chunks. If a match is found, it adds 
    the citation.
    
    If no sentence matches but the answer is valid, it defaults to the 
    highest-ranking retrieved document.
    """
    if not answer or answer == "Not found in the provided documents.":
        return []
        
    citations = []
    seen_citations = set()
    
    sentences = split_into_sentences(answer)
    
    for sentence in sentences:
        for doc in retrieved_docs:
            content = doc.page_content
            
            # Normalize whitespace and case to ensure robust matching
            clean_sentence = re.sub(r'\s+', ' ', sentence.lower()).strip()
            clean_content = re.sub(r'\s+', ' ', content.lower()).strip()
            
            # Check if there's a strong overlap
            # 1. Exact sentence substring match inside source chunk
            # 2. Or source chunk content is a substring of the sentence
            if clean_sentence in clean_content or clean_content in clean_sentence:
                source = doc.metadata.get("source", "Unknown")
                page = doc.metadata.get("page", 1)
                chunk_id = doc.metadata.get("chunk_id", 0)
                
                citation_key = (source, page, chunk_id)
                if citation_key not in seen_citations:
                    seen_citations.add(citation_key)
                    # The snippet is the exact sentence used in the answer
                    citations.append({
                        "file": source,
                        "page": page,
                        "chunk_id": chunk_id,
                        "snippet": sentence
                    })
                    break  # Sentence matched, move to the next sentence
                    
    # Fallback: If no strict sentence-level matching succeeded,
    # associate with the first (most relevant) document retrieved
    if not citations and retrieved_docs:
        top_doc = retrieved_docs[0]
        source = top_doc.metadata.get("source", "Unknown")
        page = top_doc.metadata.get("page", 1)
        chunk_id = top_doc.metadata.get("chunk_id", 0)
        
        # Take a short snippet from the top document
        snippet = top_doc.page_content[:150].strip()
        if len(top_doc.page_content) > 150:
            snippet += "..."
            
        citations.append({
            "file": source,
            "page": page,
            "chunk_id": chunk_id,
            "snippet": snippet
        })
        
    return citations
