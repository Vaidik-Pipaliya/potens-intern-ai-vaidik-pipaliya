import os
import logging
from langchain_core.documents import Document
from app.rag.loader import load_pdf
from app.rag.splitter import split_documents
from app.rag.vectorstore import get_vectorstore, CHROMA_DB_PATH

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DOCUMENTS_DIR = "v:/PROJECTS/potens-intern-ai-vaidik-pipaliya/documents"

def load_txt_or_md(file_path: str) -> list[Document]:
    """
    Loads text/markdown file and returns a list containing one Document.
    """
    file_name = os.path.basename(file_path)
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    return [
        Document(
            page_content=text,
            metadata={"source": file_name, "page": 1}
        )
    ]

def load_documents_from_folder(docs_dir: str) -> list[Document]:
    """
    Scans directory and loads PDF and TXT/MD files.
    """
    documents = []
    if not os.path.exists(docs_dir):
        os.makedirs(docs_dir, exist_ok=True)
        return documents
        
    for file in os.listdir(docs_dir):
        file_path = os.path.join(docs_dir, file)
        if os.path.isdir(file_path):
            continue
            
        ext = os.path.splitext(file)[1].lower()
        try:
            if ext == ".pdf":
                docs = load_pdf(file_path)
                documents.extend(docs)
                logger.info(f"Successfully loaded PDF: {file}")
            elif ext in [".txt", ".md"]:
                docs = load_txt_or_md(file_path)
                documents.extend(docs)
                logger.info(f"Successfully loaded Text/MD: {file}")
            else:
                logger.warning(f"Skipping unsupported file: {file}")
        except Exception as e:
            logger.error(f"Error loading {file}: {str(e)}")
            
    return documents

def run_ingestion(docs_dir: str = DOCUMENTS_DIR, db_path: str = CHROMA_DB_PATH) -> tuple[int, str]:
    """
    Orchestrates the ingestion: loads documents, chunks them, embeds them,
    and updates/recreates the Chroma vector database.
    """
    logger.info(f"Scanning documents from {docs_dir}...")
    raw_docs = load_documents_from_folder(docs_dir)
    
    if not raw_docs:
        logger.warning("No documents found to ingest.")
        return 0, "No documents found to ingest. Place PDF or Text files in the 'documents' folder."
        
    logger.info(f"Splitting {len(raw_docs)} source pages into chunks...")
    chunks = split_documents(raw_docs)
    logger.info(f"Created {len(chunks)} chunks.")
    
    try:
        db = get_vectorstore(db_path)
        # Clear existing documents to ensure a clean build without deleting directory
        # which avoids WinError 32 file locking issues on Windows when server is running.
        try:
            existing_data = db.get()
            if existing_data and existing_data.get("ids"):
                db.delete(ids=existing_data["ids"])
                logger.info("Cleared existing vector store documents for a clean build.")
        except Exception as delete_err:
            logger.warning(f"Could not clear vector store documents: {delete_err}")
            
        db.add_documents(chunks)
        
        logger.info(f"Successfully ingested {len(chunks)} chunks to ChromaDB.")
        return len(chunks), f"Successfully indexed {len(chunks)} chunks into the vector database."
    except Exception as e:
        logger.error(f"Failed vector ingestion: {str(e)}")
        return 0, f"Failed vector ingestion: {str(e)}"

if __name__ == "__main__":
    run_ingestion()
