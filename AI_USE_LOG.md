# AI Assistant Usage Log

This log documents the collaboration history, tool calls, and implementation decisions made during the construction of the RefineRAG application.

---

## 🛠️ Implementation Phases & Actions

### 1. Phase 1: Project Setup
- **Actions**: Cleared existing directory, initialized a clean Git repository, and constructed the project structure (`app/api/`, `app/rag/`, `app/utils/`, `documents/`).
- **Files Created**: `requirements.txt`, `.env` (template), `.gitignore`.

### 2. Phase 2: PDF Parsing & Metadata Extraction
- **Actions**: Built `app/rag/loader.py` using `PyMuPDF` (`fitz`). Implemented logic to traverse and extract pages, appending source metadata (`source`, `page`).
- **Files Created**: `app/rag/loader.py`.

### 3. Phase 3: Recursive Character Chunking
- **Actions**: Configured `app/rag/splitter.py` utilizing LangChain's `RecursiveCharacterTextSplitter`. Set chunk size to 1000 and overlap to 200, assigning unique sequential `chunk_id` attributes.
- **Files Created**: `app/rag/splitter.py`.

### 4. Phase 4: Embeddings & Vector DB Integration
- **Actions**: Configured local ChromaDB integration using `models/embedding-001` via `GoogleGenerativeAIEmbeddings`. Wrote `app/rag/ingest.py` to recursively load, chunk, and embed documents to ChromaDB.
- **Files Created**: `app/rag/vectorstore.py`, `app/rag/ingest.py`.

### 5. Phase 5: Core RAG Query Engine
- **Actions**: Wrote prompt instructions inside `app/rag/prompts.py` enforcing strict context-adherence. Initialized Gemini `gemini-1.5-flash` at temperature `0.0`. Set up FastAPI schema wrappers in `app/api/schemas.py` and router mappings in `app/api/routes.py`.
- **Files Created**: `app/rag/prompts.py`, `app/rag/qa_chain.py`, `app/api/schemas.py`, `app/api/routes.py`, `app/main.py`.

### 6. Phase 6: Deterministic Citation Parser
- **Actions**: Built `app/rag/citations.py` to parse LLM answer sentences and match them back lexically against the raw vector store outputs. This eliminated hallucinated page/chunk numbers.
- **Files Created**: `app/rag/citations.py`.

### 7. Phase 7: Document Contradiction Auditor
- **Actions**: Added `app/rag/contradiction.py` to compare specific sections of two different documents on a topic. Utilized Chroma metadata filtering to run isolated searches.
- **Files Created**: `app/rag/contradiction.py`.
- **Files Updated**: `app/api/schemas.py` (added Contradict schemas), `app/api/routes.py` (added /contradict route).

### 8. Phase 8: Multilingual Translation Layer
- **Actions**: Implemented `app/utils/language_detector.py` and `app/utils/translator.py` using Gemini. Integrated these helpers into `app/rag/qa_chain.py` to translate incoming queries to English, and translate final answers back to the target language while preserving the integrity of citation sources.
- **Files Created**: `app/utils/language_detector.py`, `app/utils/translator.py`.
- **Files Updated**: `app/rag/qa_chain.py` (bidirectional translation logic), `app/api/routes.py` (language mapping support).

### 9. Phase 9: Streamlit Interface Dashboard
- **Actions**: Built the UI frontend `streamlit_app.py` complete with file uploads, vector indexing controls, chat engine, language selector, and contradiction visual audit banners.
- **Files Created**: `streamlit_app.py`.

---

## 🔬 Validation & Code Verification

To ensure strict engineering discipline:
1. **API Validation**: Exposed FastAPI paths `/api/ask` and `/api/contradict` matching the strict Pydantic requirements.
2. **Deterministic Citations**: Answer sentences are cross-referenced with exact chunk text using string normalization to guarantee that metadata (filename and page number) corresponds to actual ingested documents.
3. **No Hallucinations**: Validated the template constraints; any search query lacking sufficient document context correctly results in the fallback sentence `"Not found in the provided documents."`
