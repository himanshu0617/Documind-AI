import os
import sys
import json
import time
import argparse
from typing import List, Dict, Any, Optional, Tuple
import pymupdf as fitz

# Ensure project root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.document_processing.loader import process_pdf
from src.retrieval.vectorstore import VectorStoreManager

def create_pdf_from_pages(pages: List[str]) -> bytes:
    """Create in-memory PDF from text strings for reproducible evaluation."""
    doc = fitz.open()
    for page_text in pages:
        page = doc.new_page()
        rect = fitz.Rect(50, 50, 550, 750)
        page.insert_textbox(rect, page_text, fontsize=11)
    pdf_bytes = doc.write()
    doc.close()
    return pdf_bytes

def evaluate_retrieval(
    vector_manager: VectorStoreManager,
    test_cases: List[Dict[str, Any]],
    k_values: List[int] = [1, 2, 4]
) -> Dict[str, Any]:
    """
    Evaluate retrieval quality across test cases:
    - Recall@k for each k
    - Mean Reciprocal Rank (MRR)
    """
    applicable_cases = [tc for tc in test_cases if tc["type"] in ("answerable", "ambiguous")]
    total_applicable = len(applicable_cases)

    recall_at_k: Dict[int, float] = {k: 0.0 for k in k_values}
    reciprocal_ranks: List[float] = []
    case_results: List[Dict[str, Any]] = []

    for tc in applicable_cases:
        query = tc["question"]
        expected_substrings = [s.lower() for s in tc.get("expected_source_substrings", [])]
        target_doc = tc.get("target_document")

        # Retrieve up to max k
        max_k = max(k_values)
        start_t = time.perf_counter()
        results = vector_manager.similarity_search(query, top_k=max_k)
        retrieval_time_ms = (time.perf_counter() - start_t) * 1000.0

        first_hit_rank: Optional[int] = None
        hits_at_k: Dict[int, bool] = {k: False for k in k_values}

        for rank_idx, (doc, score) in enumerate(results, start=1):
            content_lower = doc.page_content.lower()
            source = doc.metadata.get("source")

            is_match = False
            # Check if expected document matches and contains expected text
            if target_doc and source == target_doc:
                if any(sub in content_lower for sub in expected_substrings):
                    is_match = True
            elif not target_doc and any(sub in content_lower for sub in expected_substrings):
                is_match = True

            if is_match and first_hit_rank is None:
                first_hit_rank = rank_idx

            for k in k_values:
                if rank_idx <= k and is_match:
                    hits_at_k[k] = True

        # Accumulate metrics
        rr = (1.0 / first_hit_rank) if first_hit_rank else 0.0
        reciprocal_ranks.append(rr)

        for k in k_values:
            if hits_at_k[k]:
                recall_at_k[k] += 1.0

        case_results.append({
            "id": tc["id"],
            "type": tc["type"],
            "question": query,
            "target_document": target_doc,
            "first_hit_rank": first_hit_rank,
            "reciprocal_rank": round(rr, 4),
            "hits_at_k": hits_at_k,
            "retrieval_time_ms": round(retrieval_time_ms, 2)
        })

    # Average metrics
    avg_recall_at_k = {
        f"Recall@{k}": round(recall_at_k[k] / total_applicable, 4) if total_applicable > 0 else 0.0
        for k in k_values
    }
    mrr = round(sum(reciprocal_ranks) / total_applicable, 4) if total_applicable > 0 else 0.0

    return {
        "total_evaluated_cases": total_applicable,
        "metrics": {
            **avg_recall_at_k,
            "MRR": mrr
        },
        "case_details": case_results
    }

def run_evaluation(data_path: str, output_path: Optional[str] = None):
    """Run full RAG benchmark against evaluation corpus and dataset."""
    print("=" * 65)
    print("DocuMind AI -- RAG Evaluation Suite")
    print("=" * 65)

    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    corpus = data.get("corpus", [])
    test_cases = data.get("test_cases", [])

    print(f"Loaded {len(corpus)} evaluation documents and {len(test_cases)} test cases.")

    # 1. Ingest evaluation corpus into VectorStoreManager
    print("\nIngesting evaluation documents into FAISS vector store...")
    vector_manager = VectorStoreManager()
    vector_manager.clear()

    for doc_spec in corpus:
        fname = doc_spec["filename"]
        pdf_bytes = create_pdf_from_pages(doc_spec["pages"])
        result = process_pdf(
            file_bytes=pdf_bytes,
            filename=fname,
            chunk_size=500,
            chunk_overlap=100
        )
        if result.chunks and result.stats:
            vector_manager.add_documents(result.chunks, result.stats.doc_hash, fname)
            print(f"  [+] Indexed '{fname}': {result.stats.total_chunks} chunks, {result.stats.total_pages} pages")

    # 2. Evaluate Retrieval Quality
    print("\nEvaluating Retrieval Quality (Recall@k, MRR)...")
    retrieval_report = evaluate_retrieval(vector_manager, test_cases, k_values=[1, 2, 4])

    print("\n--- RETRIEVAL METRICS ---")
    for metric_name, val in retrieval_report["metrics"].items():
        print(f"  * {metric_name:12}: {val:.4f} ({val * 100:.1f}%)")

    # This script measures retrieval only; generation is not evaluated here.
    full_report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_documents": len(corpus),
        "total_test_cases": len(test_cases),
        "retrieval": retrieval_report,
        "evaluation_scope": "synthetic-corpus retrieval only; no LLM answer-quality metrics"
    }

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(full_report, f, indent=2)
        print(f"\n[+] Detailed evaluation report saved to: {output_path}")

    print("=" * 65)
    return full_report

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DocuMind AI RAG Evaluator")
    parser.add_argument(
        "--data",
        default=os.path.join(os.path.dirname(__file__), "eval_dataset.json"),
        help="Path to evaluation dataset JSON"
    )
    parser.add_argument(
        "--output",
        default=os.path.join(os.path.dirname(__file__), "eval_report.json"),
        help="Path to output report JSON"
    )
    args = parser.parse_args()
    run_evaluation(args.data, args.output)
