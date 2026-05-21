import re
from langchain_core.documents import Document

def split_into_sentences(text: str) -> list[str]:
    """
    Splits text into a list of sentences, ignoring very short fragments.
    """
    # Split using punctuation boundaries followed by whitespace
    sentences = re.split(r'(?<=[.!?])\s+', text)
    # Ignore tiny fragments ("OK.", "No.") that create false-positive substring matches.
    return [s.strip() for s in sentences if len(s.strip()) > 10]

def extract_citations(answer: str, retrieved_docs: list[Document]) -> list[dict]:
    """
    Analyzes the generated answer and maps its sentences/tags back to the retrieved 
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
    
    # Prefer [Piece N] tags from the QA prompt — maps directly to retrieval order, not fuzzy text overlap.
    has_tags = bool(re.search(r'\[Piece\s*\d+\]', answer))
    
    if has_tags:
        # Split into sentences or lines to process segments
        sentences = re.split(r'(?<=[.!?])\s+|\n+', answer)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        for sentence in sentences:
            # Find all [Piece X] tags in this sentence
            tags = re.findall(r'\[Piece\s*(\d+)\]', sentence)
            
            # Clean snippet is the sentence with tags stripped
            clean_snippet = re.sub(r'\[Piece\s*\d+\]', '', sentence).strip()
            # Normalize internal whitespaces and trailing punctuation issues
            clean_snippet = re.sub(r'\s+', ' ', clean_snippet)
            # Remove space before punctuation
            clean_snippet = re.sub(r'\s+([.,!?])', r'\1', clean_snippet)
            
            if not clean_snippet or len(clean_snippet) <= 5:
                continue
                
            for tag in tags:
                idx = int(tag) - 1
                if 0 <= idx < len(retrieved_docs):
                    doc = retrieved_docs[idx]
                    source = doc.metadata.get("source", "Unknown")
                    page = doc.metadata.get("page", 1)
                    chunk_id = doc.metadata.get("chunk_id", 0)
                    
                    citation_key = (source, page, chunk_id)
                    if citation_key not in seen_citations:
                        seen_citations.add(citation_key)
                        citations.append({
                            "file": source,
                            "page": page,
                            "chunk_id": chunk_id,
                            "snippet": clean_snippet
                        })
                        
    # 2. Legacy check: if no tags were present in the answer, run exact substring matching
    if not has_tags:
        sentences = split_into_sentences(answer)
        for sentence in sentences:
            for doc in retrieved_docs:
                content = doc.page_content
                
                # Normalize whitespace and case to ensure robust matching
                clean_sentence = re.sub(r'\s+', ' ', sentence.lower()).strip()
                clean_content = re.sub(r'\s+', ' ', content.lower()).strip()
                
                # Check if there's a strong overlap
                if clean_sentence in clean_content or clean_content in clean_sentence:
                    source = doc.metadata.get("source", "Unknown")
                    page = doc.metadata.get("page", 1)
                    chunk_id = doc.metadata.get("chunk_id", 0)
                    
                    citation_key = (source, page, chunk_id)
                    if citation_key not in seen_citations:
                        seen_citations.add(citation_key)
                        citations.append({
                            "file": source,
                            "page": page,
                            "chunk_id": chunk_id,
                            "snippet": sentence
                        })
                        break  # Sentence matched, move to the next sentence
                        
    # Last resort: paraphrased answers still get a citation, but snippet may not quote the answer text.
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
