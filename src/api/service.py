import time
import logging
from typing import List, Dict, Optional, Tuple, Any

from src.config import settings, ConfigError
from src.document_processing.loader import process_pdf, DocumentStats, PDFProcessingError
from src.retrieval.vectorstore import VectorStoreManager
from src.generation.llm import generate_answer, GenerationResult
from src.api.schemas import (
    HealthResponse,
    DocumentStatsResponse,
    DocumentListResponse,
    UploadResponse,
    DeleteResponse,
    QueryResponse,
    CitationResponse,
    TelemetryResponse
)

logger = logging.getLogger(__name__)

class RAGService:
    """
    Core business logic service orchestrating document processing, vector storage,
    and question answering for both API endpoints and UI consumers.
    """
    def __init__(self, embedding_model_name: Optional[str] = None):
        self.model_name = embedding_model_name or settings.embedding_model_name
        self.vector_manager = VectorStoreManager(embedding_model_name=self.model_name)
        self.processed_documents: Dict[str, DocumentStats] = {}

    def get_health(self) -> HealthResponse:
        """Return operational status and metadata."""
        return HealthResponse(
            status="healthy",
            app_title=settings.app_title,
            version="1.0.0",
            indexed_documents=len(self.processed_documents),
            total_chunks=self.vector_manager.total_chunks,
            embedding_model=self.model_name,
            inference_model=settings.gemini_model_name
        )

    def process_and_add_document(self, filename: str, file_bytes: bytes) -> DocumentStatsResponse:
        """Process an uploaded PDF and add it to the FAISS vector index."""
        result = process_pdf(
            file_bytes=file_bytes,
            filename=filename,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            max_size_mb=settings.max_file_size_mb
        )

        stats = result.stats
        if not stats:
            raise RuntimeError(f"Failed to generate statistics for '{filename}'.")
        if stats.error:
            raise PDFProcessingError(stats.error)

        self.processed_documents[filename] = stats

        if result.chunks and not stats.error:
            self.vector_manager.add_documents(
                documents=result.chunks,
                doc_hash=stats.doc_hash,
                filename=filename
            )

        return DocumentStatsResponse(
            filename=stats.filename,
            file_size_kb=round(stats.file_size_bytes / 1024, 2),
            total_pages=stats.total_pages,
            total_chunks=stats.total_chunks,
            is_scanned=stats.is_scanned,
            warnings=stats.warnings,
            error=stats.error
        )

    def list_documents(self) -> DocumentListResponse:
        """Return all active documents and their processing metrics."""
        docs_list = [
            DocumentStatsResponse(
                filename=s.filename,
                file_size_kb=round(s.file_size_bytes / 1024, 2),
                total_pages=s.total_pages,
                total_chunks=s.total_chunks,
                is_scanned=s.is_scanned,
                warnings=s.warnings,
                error=s.error
            )
            for s in self.processed_documents.values()
        ]
        return DocumentListResponse(
            total_documents=len(self.processed_documents),
            total_chunks=self.vector_manager.total_chunks,
            documents=docs_list
        )

    def delete_document(self, filename: str) -> Optional[DeleteResponse]:
        """Remove a specific document from vector store and registry."""
        if filename not in self.processed_documents:
            return None

        del self.processed_documents[filename]
        self.vector_manager.remove_document(filename)

        return DeleteResponse(
            message=f"Document '{filename}' successfully removed.",
            removed_filename=filename,
            remaining_documents=len(self.processed_documents),
            remaining_chunks=self.vector_manager.total_chunks
        )

    def clear_all(self) -> DeleteResponse:
        """Clear all active documents and vector stores."""
        self.processed_documents.clear()
        self.vector_manager.clear()
        return DeleteResponse(
            message="All documents and vector indices successfully cleared.",
            removed_filename=None,
            remaining_documents=0,
            remaining_chunks=0
        )

    def answer_query(
        self,
        query: str,
        top_k: Optional[int] = None,
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> QueryResponse:
        """Retrieve relevant context and generate grounded answer with citations."""
        effective_top_k = top_k or settings.retrieval_top_k

        # Check if vector store contains documents
        if self.vector_manager.total_chunks == 0:
            return QueryResponse(
                query=query,
                answer="No documents are currently indexed. Please upload at least one PDF before submitting queries.",
                citations=[],
                has_context=False,
                telemetry=None
            )

        # 1. Timed similarity search
        t_start = time.perf_counter()
        retrieved_docs = self.vector_manager.similarity_search(query, top_k=effective_top_k)
        retrieval_duration_ms = (time.perf_counter() - t_start) * 1000.0

        # 2. Generation with Gemini LLM
        gen_result: GenerationResult = generate_answer(
            query=query,
            retrieved_docs=retrieved_docs,
            chat_history=chat_history,
            retrieval_duration_ms=retrieval_duration_ms
        )

        citation_responses = [
            CitationResponse(
                index=c.index,
                source=c.source,
                page=c.page,
                chunk_id=c.chunk_id,
                score=c.score,
                text=c.text
            )
            for c in gen_result.citations
        ]

        telemetry_response = None
        if gen_result.telemetry:
            telemetry_response = TelemetryResponse(
                request_id=gen_result.telemetry.request_id,
                retrieval_duration_ms=round(gen_result.telemetry.retrieval_duration_ms, 2),
                generation_duration_ms=round(gen_result.telemetry.generation_duration_ms, 2),
                total_duration_ms=round(gen_result.telemetry.total_duration_ms, 2),
                input_tokens=gen_result.telemetry.input_tokens,
                output_tokens=gen_result.telemetry.output_tokens,
                total_tokens=gen_result.telemetry.total_tokens
            )

        return QueryResponse(
            query=query,
            answer=gen_result.answer,
            citations=citation_responses,
            has_context=gen_result.has_context,
            telemetry=telemetry_response
        )

# Global default service instance for API
rag_service = RAGService()
