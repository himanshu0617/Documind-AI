import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

# Configure structured, secure logging
logger = logging.getLogger("documind.observability")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [DocuMind] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

def sanitize_text(text: str, max_chars: int = 120) -> str:
    """Truncate and clean text to prevent leaking sensitive document payloads in logs."""
    if not text:
        return ""
    clean = text.replace("\n", " ").strip()
    if len(clean) > max_chars:
        return clean[:max_chars] + "..."
    return clean

@dataclass
class TelemetryRecord:
    """Telemetry data capturing latency, retrieval stats, token usage, and errors."""
    request_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    query_snippet: str = ""
    retrieved_count: int = 0
    retrieval_duration_ms: float = 0.0
    generation_duration_ms: float = 0.0
    total_duration_ms: float = 0.0
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "query_snippet": self.query_snippet,
            "retrieved_count": self.retrieved_count,
            "retrieval_duration_ms": round(self.retrieval_duration_ms, 2),
            "generation_duration_ms": round(self.generation_duration_ms, 2),
            "total_duration_ms": round(self.total_duration_ms, 2),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "error": self.error,
        }

def log_query_telemetry(record: TelemetryRecord):
    """Safely log telemetry data without exposing full text or credentials."""
    if record.error:
        logger.error(
            f"RequestId={record.request_id} | FAILED | "
            f"Query='{record.query_snippet}' | "
            f"TotalDuration={record.total_duration_ms:.1f}ms | "
            f"Error={record.error}"
        )
    else:
        tokens_info = (
            f"Tokens(in={record.input_tokens}, out={record.output_tokens}, total={record.total_tokens})"
            if record.total_tokens is not None else "Tokens=N/A"
        )
        logger.info(
            f"RequestId={record.request_id} | SUCCESS | "
            f"Query='{record.query_snippet}' | "
            f"RetrievedChunks={record.retrieved_count} ({record.retrieval_duration_ms:.1f}ms) | "
            f"Generation=({record.generation_duration_ms:.1f}ms) | "
            f"Total={record.total_duration_ms:.1f}ms | "
            f"{tokens_info}"
        )
