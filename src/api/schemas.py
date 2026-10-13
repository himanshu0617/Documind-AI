from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel, Field, field_validator
from src.config import settings

# 1. Health Schemas
class HealthResponse(BaseModel):
    status: Literal["healthy", "degraded", "unhealthy"] = "healthy"
    app_title: str
    version: str = "1.0.0"
    indexed_documents: int
    total_chunks: int
    embedding_model: str
    inference_model: str

# 2. Document Schemas
class DocumentStatsResponse(BaseModel):
    filename: str
    file_size_kb: float
    total_pages: int
    total_chunks: int
    is_scanned: bool = False
    warnings: List[str] = Field(default_factory=list)
    error: Optional[str] = None

class DocumentListResponse(BaseModel):
    total_documents: int
    total_chunks: int
    documents: List[DocumentStatsResponse]

class CollectionResponse(BaseModel):
    name: str
    total_documents: int
    total_chunks: int

class UploadResponse(BaseModel):
    message: str
    total_uploaded: int
    documents: List[DocumentStatsResponse]

class DeleteResponse(BaseModel):
    message: str
    removed_filename: Optional[str] = None
    remaining_documents: int
    remaining_chunks: int

# 3. Chat & Retrieval Schemas
class ChatMessageSchema(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1, max_length=10000)

class CitationResponse(BaseModel):
    index: int
    source: str
    page: int
    chunk_id: str
    score: float
    text: str

class TelemetryResponse(BaseModel):
    request_id: str
    retrieval_duration_ms: float
    generation_duration_ms: float
    total_duration_ms: float
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None

class QueryRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        max_length=settings.max_question_length,
        description="User question to be answered from the indexed documents."
    )
    top_k: Optional[int] = Field(
        default=None,
        ge=1,
        le=10,
        description="Optional override for maximum number of chunks to retrieve."
    )
    history: Optional[List[ChatMessageSchema]] = Field(
        default=None,
        max_length=6,
        description="Optional recent chat history for multi-turn conversation."
    )

    @field_validator("query")
    @classmethod
    def query_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Question must not be blank.")
        return value

class QueryResponse(BaseModel):
    query: str
    answer: str
    citations: List[CitationResponse]
    has_context: bool
    telemetry: Optional[TelemetryResponse] = None

# 4. Error Schema
class ErrorDetail(BaseModel):
    code: str
    message: str
    details: Optional[Any] = None

class ErrorResponse(BaseModel):
    error: ErrorDetail
