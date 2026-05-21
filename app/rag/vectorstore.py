import os
import time
import logging
from typing import List
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

logger = logging.getLogger(__name__)

CHROMA_DB_PATH = os.getenv(
    "CHROMA_DB_PATH", 
    "v:/PROJECTS/potens-intern-ai-vaidik-pipaliya/app/database/chroma_db"
)

class RateLimitedGoogleEmbeddings(GoogleGenerativeAIEmbeddings):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # Process in batches of 50, sleeping to avoid hitting rate limits
        batch_size = 50
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            logger.info(f"RateLimitedEmbeddings: Embedding batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1} (size {len(batch)})...")
            try:
                batch_embeddings = super().embed_documents(batch)
                all_embeddings.extend(batch_embeddings)
            except Exception as e:
                logger.warning(f"Error embedding batch: {e}. Sleeping 15s and retrying...")
                time.sleep(15)
                batch_embeddings = super().embed_documents(batch)
                all_embeddings.extend(batch_embeddings)
                
            if i + batch_size < len(texts):
                time.sleep(5) # Small sleep between successful batches
        return all_embeddings

def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    """
    Initializes Google Gemini Embeddings.
    Requires GEMINI_API_KEY in environment variables.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set in environment or .env file.")
    return RateLimitedGoogleEmbeddings(
        model="models/gemini-embedding-001", 
        google_api_key=api_key
    )

def get_vectorstore(db_path: str = CHROMA_DB_PATH) -> Chroma:
    """
    Initializes and returns the Chroma vector store using Gemini Embeddings.
    """
    embeddings = get_embeddings()
    return Chroma(
        persist_directory=db_path,
        embedding_function=embeddings
    )
