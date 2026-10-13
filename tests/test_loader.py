import pymupdf as fitz
from src.document_processing.loader import process_pdf, compute_content_hash

def create_sample_pdf_bytes(pages_text: list) -> bytes:
    """Helper to generate in-memory PDF bytes with specific text on each page."""
    doc = fitz.open()
    for text in pages_text:
        page = doc.new_page()
        page.insert_text((50, 50), text)
    pdf_bytes = doc.write()
    doc.close()
    return pdf_bytes

def test_process_valid_multipage_pdf():
    pages = [
        "First page content with crucial facts about DocuMind AI.",
        "Second page content explaining retrieval-augmented generation in detail."
    ]
    pdf_bytes = create_sample_pdf_bytes(pages)
    result = process_pdf(
        file_bytes=pdf_bytes,
        filename="test_doc.pdf",
        chunk_size=100,
        chunk_overlap=20
    )

    assert result.stats is not None
    assert result.stats.error is None
    assert result.stats.total_pages == 2
    assert result.stats.is_scanned is False
    assert len(result.chunks) >= 2

    # Verify metadata preservation on each chunk
    for chunk in result.chunks:
        assert chunk.metadata["source"] == "test_doc.pdf"
        assert chunk.metadata["page"] in [1, 2]
        assert "chunk_id" in chunk.metadata
        assert chunk.metadata["doc_hash"] == compute_content_hash(pdf_bytes)

def test_process_scanned_pdf_ocr_warning():
    """Verify that an image-only/empty PDF is flagged as requiring OCR."""
    doc = fitz.open()
    page = doc.new_page()
    # Draw a rectangle without inserting text
    page.draw_rect(fitz.Rect(10, 10, 100, 100))
    pdf_bytes = doc.write()
    doc.close()

    result = process_pdf(
        file_bytes=pdf_bytes,
        filename="scanned_document.pdf"
    )

    assert result.stats is not None
    assert result.stats.is_scanned is True
    assert len(result.chunks) == 0
    assert any("OCR" in w for w in result.stats.warnings)

def test_process_non_pdf_rejected():
    fake_bytes = b"This is plain text, not a pdf."
    result = process_pdf(
        file_bytes=fake_bytes,
        filename="document.txt"
    )
    assert result.stats is not None
    assert result.stats.error is not None
    assert "Only PDF documents are supported" in result.stats.error

def test_process_corrupted_pdf():
    corrupted_bytes = b"%PDF-1.4\ncorrupted content that cannot be parsed by fitz"
    result = process_pdf(
        file_bytes=corrupted_bytes,
        filename="corrupted.pdf"
    )
    assert result.stats is not None
    assert result.stats.error is not None
    assert "corrupted" in result.stats.error.lower()

def test_process_oversized_file():
    sample_bytes = b"A" * (2 * 1024 * 1024)  # 2MB
    result = process_pdf(
        file_bytes=sample_bytes,
        filename="large.pdf",
        max_size_mb=1  # limit 1MB
    )
    assert result.stats is not None
    assert result.stats.error is not None
    assert "exceeds maximum allowed size" in result.stats.error
