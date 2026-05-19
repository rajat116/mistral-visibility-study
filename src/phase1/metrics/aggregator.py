"""
Aggregate per-response metrics into a RunSummary.

Also computes Consistency Score — how consistently Mistral is described
across all responses in a run (inverse of sentiment standard deviation).
"""
from __future__ import annotations

import statistics
from datetime import datetime, timezone
from typing import Optional

from src.common.models import VisibilityMetrics, RunSummary
from src.common.logger import get_logger

logger = get_logger(__name__)


def _safe_mean(values: list[float]) -> float:
    return round(statistics.mean(values), 4) if values else 0.0


def _consistency_score(sentiment_scores: list[float]) -> float:
    """
    Consistency = 1 - stdev(sentiment_scores), clamped to [0, 1].
    A stdev of 0 means perfectly consistent (score = 1).
    A stdev of 1 means maximally inconsistent (score = 0).
    """
    if len(sentiment_scores) < 2:
        return 1.0
    stdev = statistics.stdev(sentiment_scores)
    return round(max(0.0, 1.0 - stdev), 4)


def aggregate_run(
    run_id: str,
    phase: str,
    started_at: str,
    metrics_list: list[VisibilityMetrics],
    baseline: Optional[RunSummary] = None,
) -> RunSummary:
    """Build a RunSummary from a list of per-response metrics."""
    if not metrics_list:
        raise ValueError("Cannot aggregate an empty metrics list.")

    # Update consistency score on each metric record
    sentiments = [m.sentiment_score for m in metrics_list]
    cs = _consistency_score(sentiments)
    for m in metrics_list:
        m.consistency_score = cs

    # Per-engine breakdown
    engines: dict[str, list[VisibilityMetrics]] = {}
    for m in metrics_list:
        engines.setdefault(m.llm_engine, []).append(m)

    per_engine: dict[str, dict] = {}
    for engine, items in engines.items():
        per_engine[engine] = {
            "mention_rate": _safe_mean([i.mention_rate for i in items]),
            "prominence_score": _safe_mean([i.prominence_score for i in items]),
            "sentiment_score": _safe_mean([i.sentiment_score for i in items]),
            "share_of_voice": _safe_mean([i.share_of_voice for i in items]),
            "recommendation_rate": _safe_mean([i.recommendation_rate for i in items]),
            "response_count": len(items),
        }

    summary = RunSummary(
        run_id=run_id,
        phase=phase,
        started_at=started_at,
        finished_at=datetime.now(timezone.utc).isoformat(),
        total_queries=len({m.query_id for m in metrics_list}),
        total_responses=len(metrics_list),
        avg_mention_rate=_safe_mean([m.mention_rate for m in metrics_list]),
        avg_prominence_score=_safe_mean([m.prominence_score for m in metrics_list]),
        avg_sentiment_score=_safe_mean([m.sentiment_score for m in metrics_list]),
        avg_share_of_voice=_safe_mean([m.share_of_voice for m in metrics_list]),
        avg_recommendation_rate=_safe_mean([m.recommendation_rate for m in metrics_list]),
        avg_consistency_score=cs,
        per_engine_metrics=per_engine,
    )

    # Compute deltas if baseline provided (Phase 2)
    if baseline:
        summary.delta_mention_rate = round(
            summary.avg_mention_rate - baseline.avg_mention_rate, 4
        )
        summary.delta_prominence_score = round(
            summary.avg_prominence_score - baseline.avg_prominence_score, 4
        )
        summary.delta_sentiment_score = round(
            summary.avg_sentiment_score - baseline.avg_sentiment_score, 4
        )
        summary.delta_share_of_voice = round(
            summary.avg_share_of_voice - baseline.avg_share_of_voice, 4
        )
        summary.delta_recommendation_rate = round(
            summary.avg_recommendation_rate - baseline.avg_recommendation_rate, 4
        )

    logger.info(
        "Run summary | run=%s | phase=%s | responses=%d | mention_rate=%.3f | sov=%.3f",
        run_id,
        phase,
        summary.total_responses,
        summary.avg_mention_rate,
        summary.avg_share_of_voice,
    )
    return summary
