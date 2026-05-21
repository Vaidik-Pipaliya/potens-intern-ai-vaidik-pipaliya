from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def split_documents(documents: list[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> list[Document]:
    """
    Splits a list of Documents into smaller, semantic chunks.
    Uses RecursiveCharacterTextSplitter to split by paragraphs, sentences, and words 
    in order to keep coherent contexts together.
    
    Crucially, it preserves the original metadata (source, page) and assigns a 
    unique chunk_id to each chunk for the citation engine.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks = splitter.split_documents(documents)
    
    # Enrich metadata with unique chunk identifiers
    for idx, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = idx
        
    return chunks
