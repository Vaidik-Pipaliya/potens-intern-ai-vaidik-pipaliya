import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

CHROMA_DB_PATH = os.getenv(
    "CHROMA_DB_PATH", 
    "v:/PROJECTS/potens-intern-ai-vaidik-pipaliya/app/database/chroma_db"
)

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
    return GoogleGenerativeAIEmbeddings(
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
