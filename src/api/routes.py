import logging
from typing import List, Optional
from fastapi import APIRouter, UploadFile, File, HTTPException, status, Depends
from src.config import settings, ConfigError
from src.api.schemas import (
    HealthResponse,
    DocumentListResponse,
    CollectionResponse,
    UploadResponse,
    DeleteResponse,
    QueryRequest,
    QueryResponse,
    DocumentStatsResponse,
)
from src.api.service import rag_service, RAGService
from src.document_processing.loader import PDFProcessingError
from src.generation.llm import ProviderError

logger = logging.getLogger(__name__)
router = APIRouter()

def get_service() -> RAGService:
    """Dependency provider for RAGService."""
    return rag_service

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health and Operational Status",
    description="Check application readiness, indexed documents, and active model versions."
)
def health_check(service: RAGService = Depends(get_service)) -> HealthResponse:
    return service.get_health()

@router.post(
    "/documents/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload PDF Documents",
    description="Upload and index one or multiple PDF documents into the FAISS vector database."
)
async def upload_documents(
    files: List[UploadFile] = File(..., description="PDF document files to upload and index"),
    service: RAGService = Depends(get_service)
) -> UploadResponse:
    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No files were provided for upload."
        )

    if len(files) > settings.max_upload_files:
        raise HTTPException(
            status_code=getattr(status, "HTTP_413_CONTENT_TOO_LARGE", 413),
            detail=f"A maximum of {settings.max_upload_files} PDF files may be uploaded at once."
        )

    processed_stats: List[DocumentStatsResponse] = []
    max_bytes = settings.max_file_size_mb * 1024 * 1024

    for file in files:
        filename = file.filename
        if not filename:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Every uploaded file must have a filename."
            )

        # Validate file extension
        if not filename.lower().endswith(".pdf"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid file type for '{filename}'. Only PDF documents (.pdf) are supported."
            )

        content = await file.read(max_bytes + 1)
        file_size = len(content)

        # Validate file size
        if file_size > max_bytes:
            size_mb = round(file_size / (1024 * 1024), 2)
            raise HTTPException(
                status_code=getattr(status, "HTTP_413_CONTENT_TOO_LARGE", 413),
                detail=(
                    f"File '{filename}' ({size_mb} MB) exceeds maximum allowed size "
                    f"of {settings.max_file_size_mb} MB."
                )
            )

        if file_size == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File '{filename}' is empty (0 bytes)."
            )

        try:
            stat_resp = service.process_and_add_document(filename, content)
            processed_stats.append(stat_resp)
        except PDFProcessingError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Could not process '{filename}': {e}"
            ) from e
        except HTTPException:
            raise
        except Exception as e:
            logger.error("Document ingestion failed (%s).", type(e).__name__)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Document processing is temporarily unavailable."
            ) from e

    return UploadResponse(
        message=f"Successfully processed {len(processed_stats)} document(s).",
        total_uploaded=len(processed_stats),
        documents=processed_stats
    )

@router.get(
    "/documents",
    response_model=DocumentListResponse,
    summary="List Ingested Documents",
    description="Retrieve the list of active documents and chunk statistics."
)
def list_documents(service: RAGService = Depends(get_service)) -> DocumentListResponse:
    return service.list_documents()

@router.get(
    "/collections/default",
    response_model=CollectionResponse,
    summary="Get the default collection",
    description="Return the process-local, shared default collection summary."
)
def get_collection(service: RAGService = Depends(get_service)) -> CollectionResponse:
    documents = service.list_documents()
    return CollectionResponse(
        name="default",
        total_documents=documents.total_documents,
        total_chunks=documents.total_chunks
    )

@router.delete(
    "/collections/default",
    response_model=DeleteResponse,
    summary="Clear the default collection",
    description="Remove all documents from the process-local default collection."
)
def clear_collection(service: RAGService = Depends(get_service)) -> DeleteResponse:
    return service.clear_all()

@router.delete(
    "/documents/{filename}",
    response_model=DeleteResponse,
    summary="Remove a Document",
    description="Remove a specific document from the index and rebuild the vector store."
)
def delete_document(
    filename: str,
    service: RAGService = Depends(get_service)
) -> DeleteResponse:
    result = service.delete_document(filename)
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{filename}' not found in active collection."
        )
    return result

@router.delete(
    "/documents",
    response_model=DeleteResponse,
    summary="Clear All Documents",
    description="Clear all documents and reset the FAISS vector database."
)
def clear_all_documents(service: RAGService = Depends(get_service)) -> DeleteResponse:
    return service.clear_all()

@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Query Document Intelligence Pipeline",
    description="Retrieve relevant document passages and generate a grounded answer with citations."
)
def query_documents(
    payload: QueryRequest,
    service: RAGService = Depends(get_service)
) -> QueryResponse:
    # Check if collection is empty
    if service.vector_manager.total_chunks == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No documents are currently indexed. Please upload at least one PDF before querying."
        )

    # Convert chat history if supplied
    chat_history = None
    if payload.history:
        chat_history = [
            {"role": msg.role, "content": msg.content}
            for msg in payload.history
        ]

    try:
        response = service.answer_query(
            query=payload.query,
            top_k=payload.top_k,
            chat_history=chat_history
        )
        return response
    except ConfigError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The language model API key is not configured."
        )
    except ProviderError as e:
        raise HTTPException(
            status_code=e.status_code,
            detail=e.message,
            headers={"Retry-After": str(e.retry_after)} if e.retry_after else None
        ) from e
