import pytest
from src.observability.tracker import TelemetryRecord, sanitize_text, log_query_telemetry

def test_sanitize_text():
    short_text = "What is the revenue?"
    assert sanitize_text(short_text) == short_text

    long_text = "A" * 200
    sanitized = sanitize_text(long_text, max_chars=50)
    assert len(sanitized) == 53  # 50 + "..."
    assert sanitized.endswith("...")

    # Strips newlines
    newline_text = "First line\nSecond line\nThird line"
    assert "\n" not in sanitize_text(newline_text)

def test_telemetry_record_serialization():
    rec = TelemetryRecord(
        query_snippet="Test query",
        retrieved_count=3,
        retrieval_duration_ms=12.555,
        generation_duration_ms=45.678,
        total_duration_ms=58.233,
        input_tokens=100,
        output_tokens=50,
        total_tokens=150
    )

    data = rec.to_dict()
    assert data["query_snippet"] == "Test query"
    assert data["retrieved_count"] == 3
    assert data["retrieval_duration_ms"] == pytest.approx(12.55, abs=0.02)
    assert data["total_tokens"] == 150
    assert data["error"] is None

def test_log_query_telemetry_does_not_raise(caplog):
    rec = TelemetryRecord(
        query_snippet="Safe query",
        retrieved_count=2,
        retrieval_duration_ms=10.0,
        generation_duration_ms=20.0,
        total_duration_ms=30.0,
        total_tokens=50
    )
    # Should execute cleanly without error
    log_query_telemetry(rec)

    err_rec = TelemetryRecord(
        query_snippet="Failing query",
        error="Simulation error"
    )
    log_query_telemetry(err_rec)
