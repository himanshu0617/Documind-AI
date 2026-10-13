import os
import sys
import time
import logging
import streamlit as st

# Ensure the root directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.config import settings, ConfigError
from src.document_processing.loader import process_pdf, DocumentStats
from src.retrieval.vectorstore import VectorStoreManager
from src.generation.llm import ProviderError, generate_answer
from src.ui.layout import (
    setup_page,
    render_header,
    render_metrics_bar,
    render_citations,
    render_document_card,
    render_empty_state,
    render_diagnostics
)

logger = logging.getLogger(__name__)

def initialize_session_state():
    """Initialize UI state scoped to the current Streamlit session."""
    if "vector_manager" not in st.session_state:
        st.session_state.vector_manager = VectorStoreManager(
            embedding_model_name=settings.embedding_model_name
        )
    if "processed_documents" not in st.session_state:
        st.session_state.processed_documents = {}  # Dict[str, DocumentStats]
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "debug_mode" not in st.session_state:
        st.session_state.debug_mode = settings.debug

def remove_document(filename: str):
    """Remove a specific document from vector store and session state."""
    if filename in st.session_state.processed_documents:
        del st.session_state.processed_documents[filename]
        st.session_state.vector_manager.remove_document(filename)

def clear_all_documents():
    """Reset the entire vector store and uploaded documents."""
    st.session_state.vector_manager.clear()
    st.session_state.processed_documents.clear()
    st.session_state.messages.clear()

def clear_chat_history():
    """Clear chat conversation while retaining indexed documents."""
    st.session_state.messages.clear()

def main():
    settings.validate_runtime()
    setup_page(settings.app_title)
    initialize_session_state()

    # 1. Sidebar Configuration & Document Management
    with st.sidebar:
        st.title("🧠 DocuMind AI")
        st.caption("Precision Document Intelligence & RAG")
        st.divider()

        # API Key Management
        if not settings.gemini_api_key or settings.gemini_api_key == "your_gemini_api_key_here":
            st.warning("GEMINI_API_KEY is not configured. Set it in your environment or .env file.")
        else:
            st.success("API Key Active (Environment)", icon="🔑")

        # RAG Parameters & Diagnostics
        with st.expander("⚙️ Pipeline Settings & Diagnostics", expanded=False):
            top_k = st.slider(
                "Max chunks to retrieve (Top-K):",
                min_value=1,
                max_value=10,
                value=settings.retrieval_top_k,
                help="Number of most relevant passages sent to the LLM."
            )
            chunk_size = st.number_input(
                "Chunk Size (chars):",
                min_value=200,
                max_value=3000,
                value=settings.chunk_size,
                step=100
            )
            chunk_overlap = st.number_input(
                "Chunk Overlap (chars):",
                min_value=0,
                max_value=500,
                value=settings.chunk_overlap,
                step=50
            )
            st.session_state.debug_mode = st.toggle(
                "Developer Diagnostics",
                value=st.session_state.debug_mode,
                help="Show execution latency breakdown, token consumption, and retrieval scores."
            )

        st.subheader("📁 Ingest Documents")
        uploaded_files = st.file_uploader(
            "Upload PDF files",
            type=["pdf"],
            accept_multiple_files=True,
            max_upload_size=settings.max_file_size_mb,
            help=f"Drag and drop PDFs here. Maximum file size: {settings.max_file_size_mb} MB per document"
        )

        # Ingestion handling with progress
        if uploaded_files:
            if len(uploaded_files) > settings.max_upload_files:
                st.error(
                    f"Upload no more than {settings.max_upload_files} PDFs at once."
                )
            else:
                newly_added = 0
                for uploaded_file in uploaded_files:
                    filename = uploaded_file.name
                    # Avoid re-processing if already in session
                    if filename not in st.session_state.processed_documents:
                        with st.spinner(f"Processing and chunking '{filename}'..."):
                            file_bytes = uploaded_file.read()
                            result = process_pdf(
                                file_bytes=file_bytes,
                                filename=filename,
                                chunk_size=chunk_size,
                                chunk_overlap=chunk_overlap,
                                max_size_mb=settings.max_file_size_mb
                            )

                            st.session_state.processed_documents[filename] = result.stats
                            if result.chunks and result.stats:
                                st.session_state.vector_manager.add_documents(
                                    documents=result.chunks,
                                    doc_hash=result.stats.doc_hash,
                                    filename=filename
                                )
                                newly_added += 1

                if newly_added > 0:
                    st.toast(f"Successfully indexed {newly_added} new document(s)!", icon="✅")

        # Display list of active documents
        if st.session_state.processed_documents:
            st.markdown(f"**Active Documents ({len(st.session_state.processed_documents)}):**")
            for fname, stats in list(st.session_state.processed_documents.items()):
                render_document_card(stats, on_remove_callback=remove_document)

            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                if st.button("Clear Chat", use_container_width=True):
                    clear_chat_history()
                    st.rerun()
            with col_btn2:
                if st.button("Remove All", use_container_width=True):
                    clear_all_documents()
                    st.rerun()
        else:
            st.info("No documents currently loaded.")

    # 2. Main Interface Header
    render_header(settings.app_title)

    # 3. Top Metrics Row
    total_docs = len(st.session_state.processed_documents)
    total_chunks = st.session_state.vector_manager.total_chunks
    pipeline_status = "Online" if total_chunks > 0 else "Awaiting Docs"
    render_metrics_bar(
        doc_count=total_docs,
        chunk_count=total_chunks,
        model_name=settings.gemini_model_name,
        status=pipeline_status
    )

    # 4. Main Area Empty State or History
    if total_docs == 0:
        render_empty_state()

    # 5. Render Conversation History
    for msg in st.session_state.messages:
        role = msg["role"]
        content = msg["content"]
        citations = msg.get("citations", [])
        telemetry = msg.get("telemetry", None)

        with st.chat_message(role):
            st.markdown(content)
            if citations:
                render_citations(citations)
            if st.session_state.debug_mode and telemetry:
                render_diagnostics(telemetry)

    # 6. Chat Input & Retrieval Pipeline
    if prompt := st.chat_input(
        "Ask a question about the uploaded documents...",
        max_chars=settings.max_question_length
    ):
        # Display user message immediately
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Process user query
        with st.chat_message("assistant"):
            if total_chunks == 0:
                warning_msg = (
                    "No document content is currently indexed. "
                    "Please upload at least one PDF with extractable text using the sidebar."
                )
                st.warning(warning_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": warning_msg,
                    "citations": []
                })
            else:
                with st.spinner("Searching document context and generating grounded answer..."):
                    try:
                        # 1. Timed vector similarity search
                        retrieval_start = time.perf_counter()
                        retrieved_docs = st.session_state.vector_manager.similarity_search(
                            query=prompt,
                            top_k=top_k
                        )
                        retrieval_ms = (time.perf_counter() - retrieval_start) * 1000.0

                        # 2. Grounded generation with Gemini
                        gen_result = generate_answer(
                            query=prompt,
                            retrieved_docs=retrieved_docs,
                            chat_history=st.session_state.messages[:-1],
                            retrieval_duration_ms=retrieval_ms
                        )

                        st.markdown(gen_result.answer)
                        if gen_result.citations:
                            render_citations(gen_result.citations)
                        if st.session_state.debug_mode and gen_result.telemetry:
                            render_diagnostics(gen_result.telemetry)

                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": gen_result.answer,
                            "citations": gen_result.citations,
                            "telemetry": gen_result.telemetry
                        })

                    except ConfigError as ce:
                        st.error(f"Configuration Error: {str(ce)}")
                    except ProviderError as ex:
                        st.error(ex.message)
                    except Exception as ex:
                        logger.error("Question answering failed (%s).", type(ex).__name__)
                        st.error("Question answering failed unexpectedly. Please retry.")

if __name__ == "__main__":
    main()
