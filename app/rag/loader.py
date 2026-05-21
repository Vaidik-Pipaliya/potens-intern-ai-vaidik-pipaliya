import os
import fitz  # PyMuPDF
from langchain_core.documents import Document

def load_pdf(file_path: str) -> list[Document]:
    """
    Loads a PDF file and extracts text page-by-page using PyMuPDF (fitz).
    Each page is loaded as a separate LangChain Document with metadata tracking 
    the file name and page number for citation generation.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF file not found at: {file_path}")
        
    documents = []
    file_name = os.path.basename(file_path)
    
    try:
        # Open PDF file using PyMuPDF
        doc = fitz.open(file_path)
        
        for page_idx in range(len(doc)):
            page = doc.load_page(page_idx)
            text = page.get_text()
            
            # Record source and page number (1-indexed for human readable citation)
            metadata = {
                "source": file_name,
                "page": page_idx + 1
            }
            
            documents.append(
                Document(
                    page_content=text,
                    metadata=metadata
                )
            )
            
        doc.close()
        
    except Exception as e:
        raise RuntimeError(f"Failed to parse PDF {file_name}: {str(e)}")
        
    return documents
