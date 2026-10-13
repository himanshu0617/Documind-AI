import os
import pytest
import pymupdf as fitz
from fastapi.testclient import TestClient
from langchain_core.documents import Document
from src.generation.llm import generate_answer
from src.config import settings
from src.api.main import app
from src.api.service import rag_service

# Integration test requires a real Gemini API Key
is_live_key_configured = bool(
    os.getenv("GEMINI_API_KEY") and os.getenv("GEMINI_API_KEY") != "your_gemini_api_key_here"
)
run_live_tests = os.getenv("DOCUMIND_RUN_LIVE_GEMINI_TESTS", "").lower() in {"1", "true", "yes"}

@pytest.mark.integration
@pytest.mark.skipif(
    not (is_live_key_configured and run_live_tests),
    reason="Set DOCUMIND_RUN_LIVE_GEMINI_TESTS=1 and configure a real GEMINI_API_KEY to run live tests."
)
def test_live_gemini_grounded_answer():
    """
    Manual live integration test: calls Gemini API with actual credentials.
    Verifies that answer generation accurately cites facts from provided context.
    """
    retrieved = [
        (
            Document(
                page_content="Project Antigravity uses high-dimensional vector representations for code indexing.",
                metadata={"source": "antigravity_spec.pdf", "page": 1, "chunk_id": "anti_p1_c1"}
            ),
            0.12
        )
    ]

    result = generate_answer(
        query="What kind of vector representations does Project Antigravity use?",
        retrieved_docs=retrieved
    )

    assert result.has_context is True
    assert "vector" in result.answer.lower()
    assert len(result.citations) == 1
    assert result.citations[0].source == "antigravity_spec.pdf"
    assert result.telemetry is not None
    assert result.telemetry.generation_duration_ms > 0

@pytest.mark.integration
@pytest.mark.skipif(
    not (is_live_key_configured and run_live_tests),
    reason="Set DOCUMIND_RUN_LIVE_GEMINI_TESTS=1 and configure a real GEMINI_API_KEY to run live tests."
)
def test_live_api_pdf_ingestion_and_question_answer_flow():
    """Exercise PDF upload, shared retrieval, and Gemini through the HTTP API."""
    pdf = fitz.open()
    page = pdf.new_page()
    page.insert_text(
        (50, 50),
        "The DocuMind sample policy says document retention is 30 days."
    )
    pdf_bytes = pdf.tobytes()
    pdf.close()

    rag_service.clear_all()
    client = TestClient(app)
    try:
        upload = client.post(
            "/documents/upload",
            files=[("files", ("sample_policy.pdf", pdf_bytes, "application/pdf"))]
        )
        assert upload.status_code == 201
        assert upload.json()["documents"][0]["total_chunks"] > 0

        answer = client.post(
            "/query",
            json={"query": "How long is document retention?"}
        )
        assert answer.status_code == 200
        assert answer.json()["citations"][0]["source"] == "sample_policy.pdf"
        assert "30" in answer.json()["answer"]
    finally:
        rag_service.clear_all()
