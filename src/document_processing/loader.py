import hashlib
import io
from dataclasses import dataclass, field
from typing import List, Optional
import pymupdf as fitz
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

@dataclass
class DocumentStats:
    """Statistics for an ingested document."""
    filename: str
    doc_hash: str
    file_size_bytes: int
    total_pages: int
    total_text_length: int
    total_chunks: int
    is_scanned: bool = False
    warnings: List[str] = field(default_factory=list)
    error: Optional[str] = None

@dataclass
class PDFProcessingResult:
    """Result of PDF ingestion and chunking."""
    chunks: List[Document] = field(default_factory=list)
    stats: Optional[DocumentStats] = None

class PDFProcessingError(Exception):
    """Exception raised when PDF processing fails."""
    pass

def compute_content_hash(data: bytes) -> str:
    """Generate SHA-256 hash for document bytes to detect duplicates/unchanged content."""
    return hashlib.sha256(data).hexdigest()

def process_pdf(
    file_bytes: bytes,
    filename: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    max_size_mb: int = 25
) -> PDFProcessingResult:
    """
    Ingest, validate, extract, and chunk a PDF document.
    
    Preserves page numbers and metadata on all chunks.
    Detects scanned/image-only PDFs and flags OCR requirements.
    """
    file_size_bytes = len(file_bytes)
    doc_hash = compute_content_hash(file_bytes)
    warnings: List[str] = []

    # 1. Validation: file size
    max_bytes = max_size_mb * 1024 * 1024
    if file_size_bytes > max_bytes:
        size_mb_actual = round(file_size_bytes / (1024 * 1024), 2)
        err_msg = (
            f"File '{filename}' exceeds maximum allowed size "
            f"({size_mb_actual}MB > {max_size_mb}MB limit)."
        )
        stats = DocumentStats(
            filename=filename,
            doc_hash=doc_hash,
            file_size_bytes=file_size_bytes,
            total_pages=0,
            total_text_length=0,
            total_chunks=0,
            error=err_msg
        )
        return PDFProcessingResult(chunks=[], stats=stats)

    # 2. Validation: file extension
    if not filename.lower().endswith(".pdf"):
        err_msg = f"Invalid file type for '{filename}'. Only PDF documents are supported."
        stats = DocumentStats(
            filename=filename,
            doc_hash=doc_hash,
            file_size_bytes=file_size_bytes,
            total_pages=0,
            total_text_length=0,
            total_chunks=0,
            error=err_msg
        )
        return PDFProcessingResult(chunks=[], stats=stats)

    # 3. Open PDF using PyMuPDF
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
    except fitz.FileDataError as e:
        err_msg = f"File '{filename}' is corrupted or not a valid PDF: {str(e)}"
        stats = DocumentStats(
            filename=filename,
            doc_hash=doc_hash,
            file_size_bytes=file_size_bytes,
            total_pages=0,
            total_text_length=0,
            total_chunks=0,
            error=err_msg
        )
        return PDFProcessingResult(chunks=[], stats=stats)
    except Exception as e:
        err_msg = f"Failed to open '{filename}': {str(e)}"
        stats = DocumentStats(
            filename=filename,
            doc_hash=doc_hash,
            file_size_bytes=file_size_bytes,
            total_pages=0,
            total_text_length=0,
            total_chunks=0,
            error=err_msg
        )
        return PDFProcessingResult(chunks=[], stats=stats)

    # 4. Handle encrypted / password-protected files
    if doc.is_encrypted:
        # Check if it opens with empty password
        if not doc.authenticate(""):
            doc.close()
            err_msg = f"File '{filename}' is password-protected/encrypted and cannot be processed."
            stats = DocumentStats(
                filename=filename,
                doc_hash=doc_hash,
                file_size_bytes=file_size_bytes,
                total_pages=0,
                total_text_length=0,
                total_chunks=0,
                error=err_msg
            )
            return PDFProcessingResult(chunks=[], stats=stats)

    total_pages = doc.page_count
    if total_pages == 0:
        doc.close()
        err_msg = f"File '{filename}' contains 0 pages."
        stats = DocumentStats(
            filename=filename,
            doc_hash=doc_hash,
            file_size_bytes=file_size_bytes,
            total_pages=0,
            total_text_length=0,
            total_chunks=0,
            error=err_msg
        )
        return PDFProcessingResult(chunks=[], stats=stats)

    # 5. Extract text page by page
    pages_text = []
    total_text_length = 0
    pages_with_images_only = 0

    for page_idx in range(total_pages):
        page = doc[page_idx]
        text = page.get_text("text").strip()
        page_num = page_idx + 1  # 1-indexed for human readability
        
        if text:
            pages_text.append((page_num, text))
            total_text_length += len(text)
        else:
            # Check if page contains images/drawings
            images = page.get_images()
            if images or len(page.get_drawings()) > 0:
                pages_with_images_only += 1

    doc.close()

    # 6. Scanned / OCR detection
    is_scanned = False
    if total_text_length == 0:
        is_scanned = True
        warnings.append(
            f"No extractable text was found in '{filename}'. "
            "The document appears to be scanned or image-based. "
            "Standard text parsing cannot extract content from image-only PDFs; "
            "Optical Character Recognition (OCR) is required to process this document."
        )
    elif pages_with_images_only > 0:
        warnings.append(
            f"Page(s) without extractable text were detected in '{filename}'. "
            f"Scanned or image-only content on these pages requires OCR to be indexed."
        )

    # If completely empty/scanned with no text, return without chunks but with clear warning
    if total_text_length == 0:
        stats = DocumentStats(
            filename=filename,
            doc_hash=doc_hash,
            file_size_bytes=file_size_bytes,
            total_pages=total_pages,
            total_text_length=0,
            total_chunks=0,
            is_scanned=True,
            warnings=warnings
        )
        return PDFProcessingResult(chunks=[], stats=stats)

    # 7. Chunking: split text while preserving page number and source metadata
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )

    all_chunks: List[Document] = []
    for page_num, page_content in pages_text:
        page_chunks = text_splitter.split_text(page_content)
        for chunk_idx, chunk_text in enumerate(page_chunks):
            chunk_id = f"{filename}_p{page_num}_c{chunk_idx + 1}"
            doc_chunk = Document(
                page_content=chunk_text,
                metadata={
                    "source": filename,
                    "page": page_num,
                    "chunk_id": chunk_id,
                    "char_count": len(chunk_text),
                    "doc_hash": doc_hash
                }
            )
            all_chunks.append(doc_chunk)

    stats = DocumentStats(
        filename=filename,
        doc_hash=doc_hash,
        file_size_bytes=file_size_bytes,
        total_pages=total_pages,
        total_text_length=total_text_length,
        total_chunks=len(all_chunks),
        is_scanned=is_scanned,
        warnings=warnings
    )
    return PDFProcessingResult(chunks=all_chunks, stats=stats)
