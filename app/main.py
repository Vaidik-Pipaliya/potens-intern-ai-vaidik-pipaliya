from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router as api_router
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(
    title="Document Q&A System with Citations",
    description="RAG-based Question Answering system with source citations, built for internship evaluation.",
    version="1.0.0"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API endpoints
app.include_router(api_router, prefix="/api")

@app.get("/")
def read_root():
    return {"message": "RAG Document Q&A System API. Access /docs for API documentation."}

if __name__ == "__main__":
    import uvicorn
    import os
    host = os.getenv("API_HOST", "127.0.0.1")
    port = int(os.getenv("API_PORT", 8000))
    uvicorn.run("main:app", host=host, port=port, reload=True)
