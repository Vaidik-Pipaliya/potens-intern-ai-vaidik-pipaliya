import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# Ensure the root project directory is in the python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langchain_core.documents import Document
from app.rag.loader import load_pdf
from app.rag.splitter import split_documents
from app.rag.citations import split_into_sentences, extract_citations
from app.utils.language_detector import detect_language
from app.utils.translator import translate_text
from app.rag.qa_chain import query_rag
from app.rag.contradiction import analyze_contradiction
from app.api.schemas import AskResponse, ContradictResponse


class TestPDFLoader(unittest.TestCase):
    @patch("fitz.open")
    def test_load_pdf_success(self, mock_fitz_open):
        # Mock PyMuPDF objects
        mock_doc = MagicMock()
        mock_page_1 = MagicMock()
        mock_page_1.get_text.return_value = "Page 1 Content"
        mock_page_2 = MagicMock()
        mock_page_2.get_text.return_value = "Page 2 Content"
        
        mock_doc.__len__.return_value = 2
        mock_doc.load_page.side_effect = [mock_page_1, mock_page_2]
        mock_fitz_open.return_value = mock_doc

        # Run loader
        with patch("os.path.exists", return_value=True):
            docs = load_pdf("dummy.pdf")

        self.assertEqual(len(docs), 2)
        self.assertEqual(docs[0].page_content, "Page 1 Content")
        self.assertEqual(docs[0].metadata["source"], "dummy.pdf")
        self.assertEqual(docs[0].metadata["page"], 1)
        self.assertEqual(docs[1].page_content, "Page 2 Content")
        self.assertEqual(docs[1].metadata["source"], "dummy.pdf")
        self.assertEqual(docs[1].metadata["page"], 2)
        mock_doc.close.assert_called_once()

    def test_load_pdf_not_found(self):
        with self.assertRaises(FileNotFoundError):
            load_pdf("nonexistent.pdf")


class TestSplitter(unittest.TestCase):
    def test_split_documents(self):
        docs = [
            Document(page_content="Hello world. This is a sentence. " * 30, metadata={"source": "doc1.txt", "page": 1})
        ]
        chunks = split_documents(docs, chunk_size=100, chunk_overlap=20)
        self.assertTrue(len(chunks) > 1)
        for idx, chunk in enumerate(chunks):
            self.assertEqual(chunk.metadata["source"], "doc1.txt")
            self.assertEqual(chunk.metadata["page"], 1)
            self.assertEqual(chunk.metadata["chunk_id"], idx)


class TestCitationEngine(unittest.TestCase):
    def test_split_into_sentences(self):
        text = "This is a sentence. And this is another sentence! Short. Okay."
        sentences = split_into_sentences(text)
        # "Short." and "Okay." are <= 10 characters and should be skipped
        self.assertEqual(len(sentences), 2)
        self.assertEqual(sentences[0], "This is a sentence.")
        self.assertEqual(sentences[1], "And this is another sentence!")

    def test_extract_citations_exact_match(self):
        retrieved_docs = [
            Document(page_content="The internship duration is six months.", metadata={"source": "policy.pdf", "page": 1, "chunk_id": 1}),
            Document(page_content="Interns receive a stipend of $500 monthly.", metadata={"source": "policy.pdf", "page": 2, "chunk_id": 2})
        ]
        answer = "The internship duration is six months. Interns also receive a stipend."
        citations = extract_citations(answer, retrieved_docs)
        
        self.assertEqual(len(citations), 1)
        self.assertEqual(citations[0]["file"], "policy.pdf")
        self.assertEqual(citations[0]["page"], 1)
        self.assertEqual(citations[0]["chunk_id"], 1)
        self.assertEqual(citations[0]["snippet"], "The internship duration is six months.")

    def test_extract_citations_tag_based(self):
        retrieved_docs = [
            Document(page_content="The internship duration is six months.", metadata={"source": "policy.pdf", "page": 1, "chunk_id": 1}),
            Document(page_content="Interns receive a stipend of $500 monthly.", metadata={"source": "policy.pdf", "page": 2, "chunk_id": 2})
        ]
        answer = "The internship is short [Piece 1]. Interns receive pay [Piece 2]."
        citations = extract_citations(answer, retrieved_docs)
        
        self.assertEqual(len(citations), 2)
        self.assertEqual(citations[0]["file"], "policy.pdf")
        self.assertEqual(citations[0]["page"], 1)
        self.assertEqual(citations[0]["chunk_id"], 1)
        self.assertEqual(citations[0]["snippet"], "The internship is short.")
        
        self.assertEqual(citations[1]["file"], "policy.pdf")
        self.assertEqual(citations[1]["page"], 2)
        self.assertEqual(citations[1]["chunk_id"], 2)
        self.assertEqual(citations[1]["snippet"], "Interns receive pay.")

    def test_extract_citations_fallback(self):
        retrieved_docs = [
            Document(page_content="The internship duration is six months.", metadata={"source": "policy.pdf", "page": 1, "chunk_id": 1})
        ]
        # No overlap with "The internship duration is six months."
        answer = "We offer a wonderful training program for new graduates."
        citations = extract_citations(answer, retrieved_docs)
        
        self.assertEqual(len(citations), 1)
        self.assertEqual(citations[0]["file"], "policy.pdf")
        self.assertEqual(citations[0]["page"], 1)
        self.assertEqual(citations[0]["chunk_id"], 1)
        self.assertTrue(citations[0]["snippet"].startswith("The internship"))


class TestLanguageAndTranslation(unittest.TestCase):
    @patch("app.utils.language_detector.detect")
    def test_detect_language(self, mock_detect):
        mock_detect.return_value = "es"

        lang = detect_language("¿Cuál es la duración?")
        self.assertEqual(lang, "Spanish")

    @patch("app.utils.translator.get_llm")
    def test_translate_text(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "What is the duration?"
        mock_llm.return_value = mock_response
        mock_llm.invoke.return_value = mock_response
        mock_get_llm.return_value = mock_llm

        translation = translate_text("¿Cuál es la duración?", "English")
        self.assertEqual(translation, "What is the duration?")


class TestQAChain(unittest.TestCase):
    @patch("app.rag.qa_chain.get_vectorstore")
    @patch("app.rag.qa_chain.get_llm")
    @patch("app.rag.qa_chain.detect_language")
    def test_query_rag_found(self, mock_detect_lang, mock_get_llm, mock_get_vectorstore):
        mock_detect_lang.return_value = "English"
        
        # Mock vector store
        mock_db = MagicMock()
        mock_doc = Document(page_content="The duration is 6 months.", metadata={"source": "policy.pdf", "page": 1, "chunk_id": 0})
        mock_db.similarity_search.return_value = [mock_doc]
        mock_get_vectorstore.return_value = mock_db

        # Mock LLM
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "The duration is 6 months."
        mock_llm.return_value = mock_response
        mock_llm.invoke.return_value = mock_response
        mock_get_llm.return_value = mock_llm

        res = query_rag("What is the duration?")
        self.assertEqual(res["answer"], "The duration is 6 months.")
        self.assertEqual(len(res["citations"]), 1)
        self.assertEqual(res["citations"][0]["file"], "policy.pdf")
        self.assertEqual(res["citations"][0]["snippet"], "The duration is 6 months.")
        self.assertEqual(res["confidence"], 0.9)

    @patch("app.rag.qa_chain.get_vectorstore")
    @patch("app.rag.qa_chain.get_llm")
    @patch("app.rag.qa_chain.detect_language")
    def test_query_rag_not_found(self, mock_detect_lang, mock_get_llm, mock_get_vectorstore):
        mock_detect_lang.return_value = "English"
        
        mock_db = MagicMock()
        mock_db.similarity_search.return_value = []
        mock_get_vectorstore.return_value = mock_db

        res = query_rag("What is the secret formula?")
        self.assertEqual(res["answer"], "Not found in the provided documents.")
        self.assertEqual(res["citations"], [])
        self.assertEqual(res["confidence"], 0.0)


class TestContradictionAuditor(unittest.TestCase):
    @patch("app.rag.contradiction.get_vectorstore")
    @patch("app.rag.contradiction.get_llm")
    def test_analyze_contradiction_found(self, mock_get_llm, mock_get_vectorstore):
        # Mock VectorStore search for doc1 and doc2
        mock_db = MagicMock()
        mock_doc1 = Document(page_content="Stipend is $500 monthly.", metadata={"source": "v1.txt", "page": 1})
        mock_doc2 = Document(page_content="Stipend is $1000 monthly.", metadata={"source": "v2.txt", "page": 1})
        
        # First call for doc1_id, second call for doc2_id
        mock_db.similarity_search.side_effect = [[mock_doc1], [mock_doc2]]
        mock_get_vectorstore.return_value = mock_db

        # Mock LLM response with JSON structure
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = """
        {
            "contradiction_found": true,
            "reasoning": "Version 1 says stipend is $500 while Version 2 says it is $1000.",
            "evidence_doc1": "Stipend is $500 monthly.",
            "evidence_doc2": "Stipend is $1000 monthly."
        }
        """
        mock_llm.return_value = mock_response
        mock_llm.invoke.return_value = mock_response
        mock_get_llm.return_value = mock_llm

        res = analyze_contradiction("v1.txt", "v2.txt", "stipend")
        self.assertTrue(res["contradiction_found"])
        self.assertEqual(res["evidence_doc1"], "Stipend is $500 monthly.")
        self.assertEqual(res["evidence_doc2"], "Stipend is $1000 monthly.")
        self.assertIn("Version 1 says stipend is $500", res["reasoning"])


class TestFastAPIEndpoints(unittest.TestCase):
    def setUp(self):
        from fastapi.testclient import TestClient
        from app.main import app
        self.client = TestClient(app)

    @patch("app.api.routes.query_rag")
    def test_ask_endpoint(self, mock_query_rag):
        mock_query_rag.return_value = {
            "answer": "The duration is 6 months.",
            "citations": [
                {
                    "file": "policy.pdf",
                    "page": 1,
                    "chunk_id": 0,
                    "snippet": "The duration is 6 months."
                }
            ],
            "confidence": 0.9
        }

        response = self.client.post("/api/ask", json={"question": "What is the duration?"})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        # Verify schema mapping
        self.assertEqual(data["answer"], "The duration is 6 months.")
        self.assertEqual(len(data["citations"]), 1)
        self.assertEqual(data["citations"][0]["file"], "policy.pdf")
        self.assertEqual(data["confidence"], 0.9)

    @patch("app.api.routes.analyze_contradiction")
    def test_contradict_endpoint(self, mock_analyze_contradiction):
        mock_analyze_contradiction.return_value = {
            "contradiction_found": True,
            "reasoning": "Conflict detected.",
            "evidence_doc1": "Evidence A.",
            "evidence_doc2": "Evidence B."
        }

        response = self.client.post("/api/contradict", json={
            "doc1_id": "v1.txt",
            "doc2_id": "v2.txt",
            "topic": "stipend"
        })
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        self.assertTrue(data["contradiction_found"])
        self.assertEqual(data["reasoning"], "Conflict detected.")
        self.assertEqual(data["evidence_doc1"], "Evidence A.")
        self.assertEqual(data["evidence_doc2"], "Evidence B.")


if __name__ == "__main__":
    unittest.main()
