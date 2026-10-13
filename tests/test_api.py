import pytest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
import pymupdf as fitz

from src.api.main import app
from src.api.service import rag_service
from src.generation.llm import ProviderError

@pytest.fixture(scope="module")
def client():
    return TestClient(app)

def create_test_pdf_bytes(text: str = "DocuMind AI delivers high precision document RAG.") -> bytes:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_textbox(fitz.Rect(50, 50, 500, 700), text, fontsize=12)
    b = doc.write()
    doc.close()
    return b

def test_api_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "indexed_documents" in data
    assert "total_chunks" in data
    assert "embedding_model" in data
    assert client.get("/api/v1/health").status_code == 200

def test_api_collection_summary(client):
    response = client.get("/collections/default")
    assert response.status_code == 200
    assert response.json()["name"] == "default"
    assert "total_chunks" in response.json()
    cleared = client.delete("/collections/default")
    assert cleared.status_code == 200
    assert cleared.json()["remaining_documents"] == 0

def test_cors_allows_only_configured_origins(client):
    allowed = client.get("/health", headers={"Origin": "http://localhost:8501"})
    assert allowed.headers["access-control-allow-origin"] == "http://localhost:8501"

    rejected = client.get("/health", headers={"Origin": "https://unknown.example"})
    assert "access-control-allow-origin" not in rejected.headers

def test_api_query_rejects_blank_and_oversized_questions(client):
    blank = client.post("/query", json={"query": "   "})
    assert blank.status_code == 422
    assert blank.json()["error"]["code"] == "VALIDATION_ERROR"
    assert "input" not in str(blank.json()["error"]["details"])

    oversized = client.post("/query", json={"query": "x" * 1001})
    assert oversized.status_code == 422

    excessive_history = client.post(
        "/query",
        json={
            "query": "question",
            "history": [
                {"role": "user", "content": "old turn"}
                for _ in range(7)
            ]
        }
    )
    assert excessive_history.status_code == 422

def test_api_query_empty_collection_returns_400(client):
    rag_service.clear_all()
    response = client.post("/query", json={"query": "What is the revenue?"})
    assert response.status_code == 400
    data = response.json()
    assert "error" in data
    assert "No documents are currently indexed" in data["error"]["message"]

def test_api_upload_non_pdf_returns_400(client):
    files = [
        ("files", ("test.txt", b"plain text", "text/plain"))
    ]
    response = client.post("/documents/upload", files=files)
    assert response.status_code == 400
    data = response.json()
    assert "error" in data
    assert "Only PDF documents (.pdf) are supported" in data["error"]["message"]

def test_api_upload_corrupt_pdf_returns_400(client):
    response = client.post(
        "/documents/upload",
        files=[("files", ("corrupt.pdf", b"not a valid PDF", "application/pdf"))]
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "HTTP_400"
    assert rag_service.list_documents().total_documents == 0

def test_api_upload_oversized_file_returns_413(client, monkeypatch):
    from src.config import settings
    monkeypatch.setattr(settings, "max_file_size_mb", 1)  # 1MB limit
    oversized_data = b"%PDF-1.4\n" + b"A" * (2 * 1024 * 1024)  # 2MB
    files = [
        ("files", ("large.pdf", oversized_data, "application/pdf"))
    ]
    response = client.post("/documents/upload", files=files)
    assert response.status_code == 413
    data = response.json()
    assert "error" in data
    assert "exceeds maximum allowed size" in data["error"]["message"]

def test_api_upload_rejects_too_many_files(client, monkeypatch):
    from src.config import settings
    monkeypatch.setattr(settings, "max_upload_files", 1)
    response = client.post(
        "/documents/upload",
        files=[
            ("files", ("one.pdf", b"%PDF", "application/pdf")),
            ("files", ("two.pdf", b"%PDF", "application/pdf")),
        ]
    )
    assert response.status_code == 413

def test_api_upload_and_list_documents(client):
    rag_service.clear_all()
    pdf_bytes = create_test_pdf_bytes("DocuMind AI provides enterprise grade RAG intelligence.")
    files = [
        ("files", ("sample_spec.pdf", pdf_bytes, "application/pdf"))
    ]
    response = client.post("/documents/upload", files=files)
    assert response.status_code == 201
    data = response.json()
    assert data["total_uploaded"] == 1
    assert data["documents"][0]["filename"] == "sample_spec.pdf"
    assert data["documents"][0]["total_chunks"] >= 1

    # Verify listing
    list_resp = client.get("/documents")
    assert list_resp.status_code == 200
    list_data = list_resp.json()
    assert list_data["total_documents"] == 1
    assert list_data["documents"][0]["filename"] == "sample_spec.pdf"

def test_api_query_without_key_returns_service_unavailable(client, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "")
    response = client.post("/query", json={"query": "What does DocuMind AI provide?"})
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "HTTP_503"

def test_api_query_with_mocked_llm(client, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "mock_key_123")

    # Mock LLM generation to avoid burning live API quota
    mock_llm_instance = MagicMock()
    mock_llm_instance.invoke.return_value = AIMessage(
        content="DocuMind AI provides enterprise grade RAG intelligence."
    )
    monkeypatch.setattr(
        "src.generation.llm.ChatGoogleGenerativeAI",
        lambda **kwargs: mock_llm_instance
    )

    query_payload = {
        "query": "What does DocuMind AI provide?",
        "top_k": 2
    }
    response = client.post("/query", json=query_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["query"] == query_payload["query"]
    assert "enterprise grade RAG" in data["answer"]
    assert len(data["citations"]) >= 1
    assert data["citations"][0]["source"] == "sample_spec.pdf"
    assert data["has_context"] is True

def test_api_provider_rate_limit_has_safe_error_and_retry_header(client, monkeypatch):
    def fail_query(*args, **kwargs):
        raise ProviderError(
            "PROVIDER_RATE_LIMITED",
            "The language model provider is rate limiting requests. Please retry shortly.",
            429,
            retry_after=5
        )

    monkeypatch.setattr("src.api.service.RAGService.answer_query", fail_query)
    response = client.post("/query", json={"query": "What does DocuMind AI provide?"})
    assert response.status_code == 429
    assert response.headers["retry-after"] == "5"
    assert response.json()["error"]["message"].startswith(
        "The language model provider is rate limiting"
    )

def test_api_delete_document(client):
    # Delete existing
    del_resp = client.delete("/documents/sample_spec.pdf")
    assert del_resp.status_code == 200
    data = del_resp.json()
    assert data["removed_filename"] == "sample_spec.pdf"
    assert data["remaining_documents"] == 0

    # Delete non-existing returns 404
    del_resp_404 = client.delete("/documents/nonexistent.pdf")
    assert del_resp_404.status_code == 404
    err_data = del_resp_404.json()
    assert "not found" in err_data["error"]["message"].lower()

def test_api_clear_all_documents(client):
    # Ingest document then clear all
    pdf_bytes = create_test_pdf_bytes("Temporary document")
    client.post("/documents/upload", files=[("files", ("temp.pdf", pdf_bytes, "application/pdf"))])
    assert rag_service.vector_manager.total_chunks > 0

    clear_resp = client.delete("/documents")
    assert clear_resp.status_code == 200
    assert clear_resp.json()["remaining_documents"] == 0
    assert rag_service.vector_manager.total_chunks == 0
