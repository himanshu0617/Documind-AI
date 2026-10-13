from contextlib import nullcontext
from unittest.mock import MagicMock

from src.generation.llm import Citation
from src.ui import layout


def test_citation_html_escapes_untrusted_document_text(monkeypatch):
    markdown = MagicMock()
    monkeypatch.setattr(layout.st, "expander", lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(layout.st, "markdown", markdown)

    layout.render_citations([
        Citation(
            index=1,
            source="<img src=x onerror=alert(1)>.pdf",
            page=1,
            chunk_id="chunk-1",
            score=0.1,
            text="<script>alert(1)</script>"
        )
    ])

    rendered = markdown.call_args.args[0]
    assert "<script>" not in rendered
    assert "&lt;script&gt;" in rendered
    assert "<img" not in rendered
    assert "&lt;img src=x onerror=alert(1)&gt;" in rendered
