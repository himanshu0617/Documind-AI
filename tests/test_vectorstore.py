import pytest
from langchain_core.documents import Document
from src.retrieval.vectorstore import VectorStoreManager

@pytest.fixture(scope="module")
def vector_manager():
    """Shared VectorStoreManager for tests using cached model."""
    return VectorStoreManager()

def test_add_and_search_documents(vector_manager):
    vector_manager.clear()
    
    doc1_chunks = [
        Document(
            page_content="Quantum computing leverages superposition and entanglement to solve certain problems.",
            metadata={"source": "quantum.pdf", "page": 1, "chunk_id": "q_1"}
        ),
        Document(
            page_content="Qubits are the basic unit of quantum information, unlike classical binary bits.",
            metadata={"source": "quantum.pdf", "page": 2, "chunk_id": "q_2"}
        )
    ]
    doc2_chunks = [
        Document(
            page_content="Photosynthesis is the process by which green plants convert light energy into chemical energy.",
            metadata={"source": "biology.pdf", "page": 1, "chunk_id": "b_1"}
        )
    ]

    # Add first doc
    added1 = vector_manager.add_documents(doc1_chunks, doc_hash="hash_quantum", filename="quantum.pdf")
    assert added1 is True
    assert vector_manager.total_chunks == 2

    # Add second doc
    added2 = vector_manager.add_documents(doc2_chunks, doc_hash="hash_bio", filename="biology.pdf")
    assert added2 is True
    assert vector_manager.total_chunks == 3

    # Test deduplication / unchanged doc skipping
    added_duplicate = vector_manager.add_documents(doc1_chunks, doc_hash="hash_quantum", filename="quantum.pdf")
    assert added_duplicate is False
    assert vector_manager.total_chunks == 3

    # Search for quantum concept
    results = vector_manager.similarity_search("How do qubits work?", top_k=2)
    assert len(results) == 2
    top_doc, score = results[0]
    assert top_doc.metadata["source"] == "quantum.pdf"
    assert "qubit" in top_doc.page_content.lower() or "quantum" in top_doc.page_content.lower()

    # Search across multiple documents
    results_bio = vector_manager.similarity_search("How do plants make energy?", top_k=1)
    assert len(results_bio) == 1
    assert results_bio[0][0].metadata["source"] == "biology.pdf"

def test_remove_document(vector_manager):
    assert "quantum.pdf" in vector_manager.indexed_files
    assert "biology.pdf" in vector_manager.indexed_files

    # Remove quantum.pdf
    removed = vector_manager.remove_document("quantum.pdf")
    assert removed is True
    assert "quantum.pdf" not in vector_manager.indexed_files
    assert vector_manager.total_chunks == 1  # only biology remains

    # Search for quantum should now only retrieve from remaining documents (biology)
    results = vector_manager.similarity_search("What is a qubit?", top_k=1)
    assert len(results) == 1
    assert results[0][0].metadata["source"] == "biology.pdf"

    # Clean up
    vector_manager.clear()
    assert vector_manager.total_chunks == 0
    assert len(vector_manager.indexed_files) == 0
