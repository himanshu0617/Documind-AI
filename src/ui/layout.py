import streamlit as st
import html
from typing import List, Optional
from src.generation.llm import Citation
from src.document_processing.loader import DocumentStats
from src.observability.tracker import TelemetryRecord

def setup_page(title: str):
    """Configure Streamlit page layout, metadata, and custom styles."""
    st.set_page_config(
        page_title=title,
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    apply_custom_styles()

def apply_custom_styles():
    """Apply modern CSS styles suitable for an enterprise portfolio."""
    st.markdown(
        """
        <style>
        /* General layout refinements */
        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1200px;
        }
        /* Top metrics container */
        .metric-card {
            background: #ffffff;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 12px 16px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
            text-align: center;
        }
        .metric-title {
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: #5f6368;
            font-weight: 600;
        }
        .metric-value {
            font-size: 20px;
            font-weight: 700;
            color: #202124;
            margin-top: 4px;
        }
        /* Citation Cards */
        .citation-box {
            background-color: #f8f9fa;
            border-left: 4px solid #1a73e8;
            padding: 10px 14px;
            margin-top: 8px;
            margin-bottom: 8px;
            border-radius: 4px;
            border-top: 1px solid #e8eaed;
            border-right: 1px solid #e8eaed;
            border-bottom: 1px solid #e8eaed;
        }
        .citation-header {
            font-size: 13px;
            font-weight: 600;
            color: #1a73e8;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .citation-body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            font-size: 13px;
            color: #3c4043;
            margin-top: 6px;
            line-height: 1.5;
            background: #ffffff;
            padding: 8px;
            border-radius: 4px;
            border: 1px solid #f1f3f4;
        }
        /* Telemetry badge */
        .telemetry-badge {
            font-size: 11px;
            font-family: monospace;
            background: #f1f3f4;
            color: #3c4043;
            padding: 2px 6px;
            border-radius: 4px;
            margin-right: 6px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

def render_header(title: str):
    """Render top application banner with status badge."""
    col1, col2 = st.columns([0.8, 0.2])
    with col1:
        st.title(f"🧠 {title}")
        st.caption("Document Q&A • Grounded retrieval with FAISS & Gemini")
    with col2:
        st.markdown(
            """
            <div style="text-align: right; padding-top: 25px;">
                <span style="background-color: #e6f4ea; color: #137333; padding: 6px 12px; border-radius: 16px; font-weight: 600; font-size: 12px; border: 1px solid #ceead6;">
                    ● Active Session
                </span>
            </div>
            """,
            unsafe_allow_html=True
        )

def render_metrics_bar(doc_count: int, chunk_count: int, model_name: str, status: str = "Ready"):
    """Render top metrics row for quick operational status."""
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f"""<div class="metric-card">
                <div class="metric-title">Active Documents</div>
                <div class="metric-value">📄 {doc_count}</div>
            </div>""",
            unsafe_allow_html=True
        )
    with c2:
        st.markdown(
            f"""<div class="metric-card">
                <div class="metric-title">Indexed Chunks</div>
                <div class="metric-value">🧩 {chunk_count}</div>
            </div>""",
            unsafe_allow_html=True
        )
    with c3:
        st.markdown(
            f"""<div class="metric-card">
                <div class="metric-title">Inference Model</div>
                <div class="metric-value" style="font-size: 16px; padding-top: 4px;">⚡ {model_name}</div>
            </div>""",
            unsafe_allow_html=True
        )
    with c4:
        st.markdown(
            f"""<div class="metric-card">
                <div class="metric-title">Pipeline Status</div>
                <div class="metric-value" style="color: #137333;">{status}</div>
            </div>""",
            unsafe_allow_html=True
        )
    st.write("")

def render_citations(citations: List[Citation]):
    """Render interactive, expandable citations for retrieved passages."""
    if not citations:
        return

    num_sources = len(citations)
    with st.expander(f"📚 Retrieved Source Citations ({num_sources} chunk{'s' if num_sources > 1 else ''})", expanded=False):
        for cit in citations:
            score_display = f"Distance Score: {cit.score:.4f}"
            source = html.escape(cit.source)
            chunk_id = html.escape(cit.chunk_id)
            text = html.escape(cit.text)
            st.markdown(
                f"""
                <div class="citation-box">
                    <div class="citation-header">
                        <span>📄 <b>{source}</b> — Page {cit.page} (Chunk {chunk_id})</span>
                        <span style="color: #5f6368; font-size: 12px; font-weight: normal;">{score_display}</span>
                    </div>
                    <div class="citation-body">
                        {text}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

def render_diagnostics(telemetry: Optional[TelemetryRecord]):
    """Render diagnostic observability details when in debug mode."""
    if not telemetry:
        return

    with st.expander("🔍 Observability & Telemetry (Debug Mode)", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"**Request ID:** `{telemetry.request_id}`")
            st.markdown(f"**Total Duration:** `{telemetry.total_duration_ms:.1f} ms`")
        with c2:
            st.markdown(f"**Retrieval Time:** `{telemetry.retrieval_duration_ms:.1f} ms`")
            st.markdown(f"**Generation Time:** `{telemetry.generation_duration_ms:.1f} ms`")
        with c3:
            tok_in = telemetry.input_tokens if telemetry.input_tokens is not None else "N/A"
            tok_out = telemetry.output_tokens if telemetry.output_tokens is not None else "N/A"
            tok_tot = telemetry.total_tokens if telemetry.total_tokens is not None else "N/A"
            st.markdown(f"**Prompt Tokens:** `{tok_in}`")
            st.markdown(f"**Output Tokens:** `{tok_out}` (Total: `{tok_tot}`)")

def render_document_card(stats: DocumentStats, on_remove_callback=None):
    """Render status card for an uploaded document."""
    if stats.error:
        st.error(f"❌ **{stats.filename}**: {stats.error}")
        return

    size_kb = round(stats.file_size_bytes / 1024, 1)
    
    with st.container(border=True):
        col1, col2 = st.columns([0.8, 0.2])
        with col1:
            st.markdown(f"**📄 {stats.filename}**")
            st.caption(f"{stats.total_pages} page(s) • {stats.total_chunks} chunk(s) • {size_kb} KB")
        with col2:
            if on_remove_callback and st.button("🗑️", key=f"del_{stats.filename}", help=f"Remove {stats.filename}"):
                on_remove_callback(stats.filename)
                st.rerun()

        # Display warnings if present (e.g., scanned / OCR warning)
        if stats.warnings:
            for w in stats.warnings:
                if stats.is_scanned:
                    st.warning(f"⚠️ **OCR Required:** {w}")
                else:
                    st.info(f"ℹ️ {w}")

def render_empty_state():
    """Render welcoming empty state with instructions and sample workflow."""
    st.markdown(
        """
        <div style="background-color: #f8f9fa; border: 2px dashed #dadce0; border-radius: 10px; padding: 30px; text-align: center; margin-top: 10px; margin-bottom: 25px;">
            <div style="font-size: 40px; margin-bottom: 10px;">📂</div>
            <h3 style="margin-bottom: 6px; color: #202124;">No Documents Ingested Yet</h3>
            <p style="color: #5f6368; max-width: 580px; margin: 0 auto 18px auto; font-size: 14px;">
                Upload one or multiple PDF documents using the left sidebar to initialize the FAISS vector index and begin querying.
            </p>
            <div style="display: flex; justify-content: center; gap: 15px; font-size: 13px; color: #1a73e8;">
                <span>🧠 In-Memory Index</span>
                <span>•</span>
                <span>⚡ HuggingFace MiniLM Embeddings</span>
                <span>•</span>
                <span>🎯 Grounded Citations</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
