"""
Phase 1 orchestration pipeline.

Flow:
  load queries → for each query × engine:
    1. call LLM
    2. compute metrics
    3. store raw response to GCS + BQ
    4. store metrics to BQ
  → aggregate run summary → store to BQ
"""
from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from typing import Optional

from src.common.config import config
from src.common.logger import get_logger
from src.common.models import RunSummary
from src.phase1.queries.loader import load_queries
from src.phase1.queries.engines import get_all_engines, LLMEngine
from src.phase1.metrics.extractor import compute_metrics
from src.phase1.metrics.aggregator import aggregate_run
from src.phase1.storage.gcs_client import GCSClient
from src.phase1.storage.bigquery_client import BigQueryClient

logger = get_logger(__name__)


def run_phase1(
    run_id: Optional[str] = None,
    dry_run: bool = False,
) -> RunSummary:
    """
    Execute the full Phase 1 measurement pipeline.

    Parameters
    ----------
    run_id:
        Override the auto-generated run identifier.
    dry_run:
        If True, skip GCS/BQ writes (useful for local testing).
    """
    run_id = run_id or f"{config.run_id_prefix}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:6]}"
    started_at = datetime.now(timezone.utc).isoformat()

    logger.info("=== Phase 1 pipeline START | run_id=%s | dry_run=%s ===", run_id, dry_run)

    queries = load_queries()
    engines: list[LLMEngine] = get_all_engines()

    gcs = GCSClient() if not dry_run else None
    bq = BigQueryClient() if not dry_run else None

    all_metrics = []

    for query in queries:
        for engine in engines:
            try:
                # Pace Gemini calls to stay within free-tier rate limits
                if "gemini" in engine.engine_name.lower():
                    time.sleep(5)

                # 1. Query LLM
                record = engine.query(
                    run_id=run_id,
                    phase="phase1",
                    query_id=query.query_id,
                    query_text=query.text,
                )

                # 2. Compute metrics
                metrics = compute_metrics(record)

                # 3. Persist
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

    # 4. Aggregate
    summary = aggregate_run(
        run_id=run_id,
        phase="phase1",
        started_at=started_at,
        metrics_list=all_metrics,
    )

    if not dry_run:
        bq.insert_run_summary(summary)

    logger.info(
        "=== Phase 1 COMPLETE | mention_rate=%.3f | sov=%.3f | sentiment=%.3f ===",
        summary.avg_mention_rate,
        summary.avg_share_of_voice,
        summary.avg_sentiment_score,
    )
    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Phase 1 measurement pipeline")
    parser.add_argument("--dry-run", action="store_true", help="Skip GCP writes")
    parser.add_argument("--run-id", type=str, default=None)
    args = parser.parse_args()

    result = run_phase1(run_id=args.run_id, dry_run=args.dry_run)
    print(result.model_dump_json(indent=2))
