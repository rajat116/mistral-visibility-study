"""Unit tests for the metrics aggregator."""
import pytest
from src.common.models import VisibilityMetrics
from src.phase1.metrics.aggregator import aggregate_run, _consistency_score


def _make_metric(**kwargs) -> VisibilityMetrics:
    defaults = dict(
        record_id="rec-001",
        run_id="run-001",
        phase="phase1",
        llm_engine="gpt-4o",
        query_id="q001",
        mention_rate=1.0,
        prominence_score=0.8,
        sentiment_score=0.5,
        share_of_voice=0.4,
        recommendation_rate=1.0,
    )
    defaults.update(kwargs)
    return VisibilityMetrics(**defaults)


class TestConsistencyScore:
    def test_identical_sentiments(self):
        assert _consistency_score([0.5, 0.5, 0.5]) == 1.0

    def test_varied_sentiments(self):
        score = _consistency_score([1.0, 0.0, -1.0, 0.5])
        assert 0.0 <= score <= 1.0

    def test_single_value(self):
        assert _consistency_score([0.7]) == 1.0

    def test_empty(self):
        assert _consistency_score([]) == 1.0


class TestAggregateRun:
    def test_basic_aggregation(self):
        metrics = [
            _make_metric(mention_rate=1.0, share_of_voice=0.5, sentiment_score=0.3),
            _make_metric(mention_rate=0.0, share_of_voice=0.0, sentiment_score=-0.2, query_id="q002"),
        ]
        summary = aggregate_run("run-test", "phase1", "2024-01-01T00:00:00+00:00", metrics)

        assert summary.total_responses == 2
        assert summary.avg_mention_rate == pytest.approx(0.5, abs=0.01)
        assert summary.avg_share_of_voice == pytest.approx(0.25, abs=0.01)

    def test_consistency_score_set(self):
        metrics = [
            _make_metric(sentiment_score=0.5, query_id=f"q{i:03d}")
            for i in range(5)
        ]
        summary = aggregate_run("run-test", "phase1", "2024-01-01T00:00:00+00:00", metrics)
        assert summary.avg_consistency_score == 1.0  # all same sentiment

    def test_per_engine_breakdown(self):
        metrics = [
            _make_metric(llm_engine="gpt-4o", mention_rate=1.0),
            _make_metric(llm_engine="gemini-1.5-pro", mention_rate=0.0, query_id="q002"),
        ]
        summary = aggregate_run("run-test", "phase1", "2024-01-01T00:00:00+00:00", metrics)
        assert "gpt-4o" in summary.per_engine_metrics
        assert "gemini-1.5-pro" in summary.per_engine_metrics

    def test_delta_computation(self):
        from src.common.models import RunSummary
        from datetime import datetime, timezone

        baseline = RunSummary(
            run_id="baseline",
            phase="phase1",
            started_at="2024-01-01T00:00:00+00:00",
            total_queries=5,
            total_responses=10,
            avg_mention_rate=0.4,
            avg_prominence_score=0.3,
            avg_sentiment_score=0.1,
            avg_share_of_voice=0.2,
            avg_recommendation_rate=0.2,
            avg_consistency_score=0.8,
        )
        metrics = [_make_metric(mention_rate=0.8, share_of_voice=0.5)]
        summary = aggregate_run(
            "run-p2", "phase2", "2024-01-02T00:00:00+00:00", metrics, baseline=baseline
        )
        assert summary.delta_mention_rate == pytest.approx(0.4, abs=0.01)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            aggregate_run("run-empty", "phase1", "2024-01-01T00:00:00+00:00", [])
