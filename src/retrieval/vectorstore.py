import logging
from typing import List, Tuple, Optional, Dict, Any
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

# Prefer langchain_huggingface, fallback to langchain_community if needed
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)

# Cache embedding instance in module so we don't re-instantiate multiple models
_EMBEDDINGS_CACHE: Dict[str, HuggingFaceEmbeddings] = {}

def get_embeddings_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> HuggingFaceEmbeddings:
    """Load or retrieve cached embedding model."""
    if model_name not in _EMBEDDINGS_CACHE:
        logger.info(f"Loading embedding model: {model_name}")
        _EMBEDDINGS_CACHE[model_name] = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
    return _EMBEDDINGS_CACHE[model_name]

class VectorStoreManager:
    """
    Manages vector storage and similarity retrieval using FAISS.
    
    Maintains a per-instance index, avoids re-indexing unchanged documents,
    and supports multi-document collections.
    """
    def __init__(self, embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.embedding_model_name = embedding_model_name
        self._embeddings = get_embeddings_model(embedding_model_name)
        self._vector_store: Optional[FAISS] = None
        # Maps doc_hash -> List of Documents
        self._indexed_documents: Dict[str, List[Document]] = {}
        # Maps filename -> doc_hash
        self._filename_to_hash: Dict[str, str] = {}

    @property
    def total_chunks(self) -> int:
        """Return total number of chunks currently indexed."""
        return sum(len(chunks) for chunks in self._indexed_documents.values())

    @property
    def indexed_files(self) -> List[str]:
        """Return list of filenames currently indexed."""
        return list(self._filename_to_hash.keys())

    def is_document_indexed(self, doc_hash: str) -> bool:
        """Check if a document with this content hash is already indexed."""
        return doc_hash in self._indexed_documents

    def add_documents(self, documents: List[Document], doc_hash: str, filename: str) -> bool:
        """
        Add document chunks to vector index.
        
        Skips indexing if doc_hash is already indexed to avoid redundant computations.
        Returns True if newly added, False if skipped.
        """
        if not documents:
            logger.warning(f"No chunks provided to index for '{filename}'.")
            return False

        if self.is_document_indexed(doc_hash):
            logger.info(f"Document '{filename}' (hash: {doc_hash[:8]}...) already indexed. Skipping.")
            return False

        try:
            if self._vector_store is None:
                self._vector_store = FAISS.from_documents(documents, self._embeddings)
            else:
                self._vector_store.add_documents(documents)

            self._indexed_documents[doc_hash] = documents
            self._filename_to_hash[filename] = doc_hash
            logger.info(f"Successfully indexed {len(documents)} chunks for '{filename}'.")
            return True
        except Exception as e:
            logger.error(f"Failed to index document '{filename}': {str(e)}")
            raise RuntimeError(f"Embedding/VectorStore error for '{filename}': {str(e)}") from e

    def remove_document(self, filename: str) -> bool:
        """
        Remove a document and its chunks from index, rebuilding vector store
        with remaining documents.
        """
        if filename not in self._filename_to_hash:
            return False

        doc_hash = self._filename_to_hash.pop(filename)
        self._indexed_documents.pop(doc_hash, None)

        # Rebuild vector store from remaining chunks
        remaining_chunks = [
            chunk
            for chunks in self._indexed_documents.values()
            for chunk in chunks
        ]

        if remaining_chunks:
            self._vector_store = FAISS.from_documents(remaining_chunks, self._embeddings)
        else:
            self._vector_store = None

        return True

    def clear(self) -> None:
        """Reset and clear all indexed vectors and document collections."""
        self._vector_store = None
        self._indexed_documents.clear()
        self._filename_to_hash.clear()

    def similarity_search(
        self,
        query: str,
        top_k: int = 4,
        score_threshold: Optional[float] = None
    ) -> List[Tuple[Document, float]]:
        """
        Retrieve the most relevant chunks for a question.
        
        Returns a list of (Document, score) tuples.
        When embeddings are normalized, score represents cosine distance (lower is closer).
        """
        if not self._vector_store or not query.strip():
            return []

        # Retrieve documents with distance scores
        results: List[Tuple[Document, float]] = self._vector_store.similarity_search_with_score(
            query, k=top_k
        )

        if score_threshold is not None:
            # Filter results if threshold is specified
            results = [
                (doc, score) for doc, score in results
                if score <= score_threshold
            ]

        return results
