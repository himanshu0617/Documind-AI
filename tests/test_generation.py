import pytest
from unittest.mock import MagicMock
from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from src.generation.llm import (
    ProviderError,
    format_context,
    generate_answer,
    SYSTEM_PROMPT,
)

def test_format_context_with_citations():
    retrieved = [
        (
            Document(
                page_content="The revenue grew by 25% in Q3.",
                metadata={"source": "report_q3.pdf", "page": 4, "chunk_id": "rep_p4_c1"}
            ),
            0.1234
        ),
        (
            Document(
                page_content="Operating margins expanded by 150 bps.",
                metadata={"source": "report_q3.pdf", "page": 5, "chunk_id": "rep_p5_c2"}
            ),
            0.2345
        )
    ]

    context_str, citations = format_context(retrieved)

    assert "report_q3.pdf" in context_str
    assert "Page: 4" in context_str
    assert "Page: 5" in context_str
    assert len(citations) == 2
    assert citations[0].source == "report_q3.pdf"
    assert citations[0].page == 4
    assert citations[0].score == 0.1234
    assert citations[0].text == "The revenue grew by 25% in Q3."

def test_empty_retrieved_documents():
    """Verify that when no chunks are found, the app gracefully explains lack of info."""
    result = generate_answer("What was the revenue?", retrieved_docs=[])
    assert result.has_context is False
    assert "do not provide enough information" in result.answer
    assert len(result.citations) == 0

def test_system_prompt_untrusted_data_rule():
    """Ensure system prompt explicitly dictates untrusted context handling and no hallucination."""
    assert "UNTRUSTED CONTENT DEFENSE" in SYSTEM_PROMPT
    assert "STRICT GROUNDING" in SYSTEM_PROMPT
    assert "do not provide enough information" in SYSTEM_PROMPT

def test_generate_answer_invokes_llm(monkeypatch):
    # Mock ChatGoogleGenerativeAI to avoid live API token consumption in automated tests
    mock_llm_instance = MagicMock()
    mock_llm_instance.invoke.return_value = AIMessage(content="Revenue grew by 25% according to report_q3.pdf p. 4.")
    
    monkeypatch.setattr(
        "src.generation.llm.ChatGoogleGenerativeAI",
        lambda **kwargs: mock_llm_instance
    )

    retrieved = [
        (
            Document(
                page_content="The revenue grew by 25% in Q3.",
                metadata={"source": "report_q3.pdf", "page": 4, "chunk_id": "rep_p4_c1"}
            ),
            0.15
        )
    ]

    result = generate_answer(
        query="How much did revenue grow?",
        retrieved_docs=retrieved,
        chat_history=[{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}],
        api_key="valid_mock_api_key_123"
    )

    assert result.has_context is True
    assert "25%" in result.answer
    assert len(result.citations) == 1
    assert result.citations[0].source == "report_q3.pdf"
    assert mock_llm_instance.invoke.called

def test_generate_answer_extracts_text_from_multimodal_content(monkeypatch):
    mock_llm_instance = MagicMock()
    mock_llm_instance.invoke.return_value = AIMessage(content=[
        {
            "type": "text",
            "text": "Here is the document summary.",
            "signature": "internal-signature-data",
        },
        {"type": "image", "data": "internal-image-data"},
    ])
    monkeypatch.setattr(
        "src.generation.llm.ChatGoogleGenerativeAI",
        lambda **kwargs: mock_llm_instance
    )

    doc = Document(
        page_content="The document describes Python functions.",
        metadata={"source": "lesson.pdf", "page": 1, "chunk_id": "lesson_p1_c1"}
    )
    result = generate_answer("Summarize the document.", [(doc, 0.1)], api_key="mock-key")

    assert result.answer == "Here is the document summary."
    assert "signature" not in result.answer
    assert "internal-image-data" not in result.answer

@pytest.mark.parametrize(
    ("error", "status_code", "error_code"),
    [
        (TimeoutError("synthetic timeout marker"), 504, "PROVIDER_TIMEOUT"),
        (RuntimeError("504 DEADLINE_EXCEEDED"), 504, "PROVIDER_TIMEOUT"),
        (RuntimeError("RESOURCE_EXHAUSTED"), 429, "PROVIDER_RATE_LIMITED"),
        (RuntimeError("503 UNAVAILABLE: temporarily unavailable"), 503, "PROVIDER_OVERLOADED"),
        (RuntimeError("404 NOT_FOUND: model not found"), 502, "PROVIDER_MODEL_UNAVAILABLE"),
        (RuntimeError("401 API key not valid"), 502, "PROVIDER_AUTHENTICATION_FAILED"),
        (RuntimeError("upstream failure"), 502, "PROVIDER_UNAVAILABLE"),
    ],
)
def test_provider_failures_are_classified(monkeypatch, error, status_code, error_code):
    mock_llm_instance = MagicMock()
    mock_llm_instance.invoke.side_effect = error
    monkeypatch.setattr(
        "src.generation.llm.ChatGoogleGenerativeAI",
        lambda **kwargs: mock_llm_instance
    )
    doc = Document(
        page_content="Relevant passage.",
        metadata={"source": "sample.pdf", "page": 1, "chunk_id": "c1"}
    )

    with pytest.raises(ProviderError) as raised:
        generate_answer("Question?", [(doc, 0.1)], api_key="mock-key")

    assert raised.value.status_code == status_code
    assert raised.value.code == error_code
    assert error.args[0] not in raised.value.message
