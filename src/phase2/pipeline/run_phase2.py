"""
Phase 2 orchestration pipeline.

Flow:
  1. Generate synthetic content (if not already generated)
  2. Build / load RAG index
  3. For each query × engine:
     a. Retrieve relevant chunks
     b. Call LLM with RAG-augmented prompt
     c. Compute metrics
     d. Store raw response + metrics to GCS + BQ
  4. Aggregate run summary with before/after deltas
  5. Store run summary to BQ
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from src.common.config import config
from src.common.logger import get_logger
from src.common.models import RunSummary
from src.phase1.queries.loader import load_queries
from src.phase1.queries.engines import get_all_engines
from src.phase1.metrics.extractor import compute_metrics
from src.phase1.metrics.aggregator import aggregate_run
from src.phase1.storage.gcs_client import GCSClient
from src.phase1.storage.bigquery_client import BigQueryClient
from src.phase2.content_gen.generator import generate_all_content
from src.phase2.rag.indexer import RAGIndexer

logger = get_logger(__name__)

_CONTENT_DIR = Path(__file__).parents[4] / "data" / "synthetic_content"


def run_phase2(
    run_id: Optional[str] = None,
    baseline_run_id: Optional[str] = None,
    dry_run: bool = False,
    regenerate_content: bool = False,
    rebuild_index: bool = False,
) -> RunSummary:
    """
    Execute the full Phase 2 RAG-augmented pipeline.

    Parameters
    ----------
    run_id:
        Override the auto-generated run identifier.
    baseline_run_id:
        If provided, fetch this Phase 1 run's summary to compute deltas.
    dry_run:
        Skip GCS/BQ writes.
    regenerate_content:
        Force regeneration of synthetic content even if already on disk.
    rebuild_index:
        Force rebuilding the FAISS index even if already built.
    """
    run_id = run_id or (
        f"{config.run_id_prefix}-p2-"
        f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}-"
        f"{uuid.uuid4().hex[:6]}"
    )
    started_at = datetime.now(timezone.utc).isoformat()

    logger.info("=== Phase 2 pipeline START | run_id=%s | dry_run=%s ===", run_id, dry_run)

    # --- Step 1: Synthetic content ---
    if regenerate_content or not any(_CONTENT_DIR.glob("*.md")):
        logger.info("Generating synthetic content...")
        generate_all_content(output_dir=_CONTENT_DIR)
    else:
        logger.info("Synthetic content already exists, skipping generation.")

    # --- Step 2: RAG index ---
    indexer = RAGIndexer()
    index_path = Path(__file__).parents[4] / "data" / "results" / "rag_index" / "index.faiss"
    if rebuild_index or not index_path.exists():
        logger.info("Building RAG index from synthetic content...")
        indexer.build_from_files(_CONTENT_DIR)
    else:
        logger.info("Loading existing RAG index...")
        indexer.load()

    # --- Step 3: Query loop ---
    queries = load_queries()
    engines = get_all_engines()

    gcs = GCSClient() if not dry_run else None
    bq = BigQueryClient() if not dry_run else None

    # Optionally fetch baseline
    baseline: Optional[RunSummary] = None
    if baseline_run_id and not dry_run:
        bq_for_baseline = bq or BigQueryClient()
        rows = bq_for_baseline.get_run_summaries(phase="phase1", limit=100)
        for row in rows:
            if row.get("run_id") == baseline_run_id:
                # Normalize types coming back from BigQuery
                for ts_field in ("started_at", "finished_at"):
                    if row.get(ts_field) and not isinstance(row[ts_field], str):
                        row[ts_field] = row[ts_field].isoformat()
                if row.get("per_engine_metrics") and isinstance(row["per_engine_metrics"], str):
                    row["per_engine_metrics"] = json.loads(row["per_engine_metrics"])
                baseline = RunSummary(**row)
                break
        if baseline:
            logger.info("Baseline loaded: %s", baseline_run_id)
        else:
            logger.warning("Baseline run_id not found: %s", baseline_run_id)

    all_metrics = []

    for query in queries:
        for engine in engines:
            try:
                # Retrieve RAG context
                chunks = indexer.retrieve(query.text, top_k=config.rag_top_k)
                context = indexer.build_context_string(chunks)

                # Query LLM with context
                record = engine.query(
                    run_id=run_id,
                    phase="phase2",
                    query_id=query.query_id,
                    query_text=query.text,
                    context=context,
                )

                metrics = compute_metrics(record)

                if not dry_run:
                    gcs.upload_record(record)
                    bq.insert_raw_response(record)
                    bq.insert_metrics(metrics)

                all_metrics.append(metrics)
                logger.info(
                    "Done | engine=%s | query=%s | mention=%s | sov=%.3f",
                    engine.engine_name,
                    query.query_id,
                    metrics.mention_rate,
                    metrics.share_of_voice,
                )

            except Exception as exc:
                logger.error(
                    "Error | engine=%s | query=%s | %s",
                    engine.engine_name,
                    query.query_id,
                    exc,
                    exc_info=True,
                )

    # --- Step 4: Aggregate ---
    summary = aggregate_run(
        run_id=run_id,
        phase="phase2",
        started_at=started_at,
        metrics_list=all_metrics,
        baseline=baseline,
    )

    if not dry_run:
        bq.insert_run_summary(summary)

    logger.info(
        "=== Phase 2 COMPLETE | mention_rate=%.3f | sov=%.3f | Δsov=%s ===",
        summary.avg_mention_rate,
        summary.avg_share_of_voice,
        summary.delta_share_of_voice,
    )
    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Phase 2 RAG-augmented pipeline")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--baseline-run-id", type=str, default=None)
    parser.add_argument("--regenerate-content", action="store_true")
    parser.add_argument("--rebuild-index", action="store_true")
    args = parser.parse_args()

    result = run_phase2(
        run_id=args.run_id,
        baseline_run_id=args.baseline_run_id,
        dry_run=args.dry_run,
        regenerate_content=args.regenerate_content,
        rebuild_index=args.rebuild_index,
    )
    print(result.model_dump_json(indent=2))
