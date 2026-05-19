"""Shared Pydantic data models used across phases."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class QueryRecord(BaseModel):
    """Raw query + response captured from an LLM."""

    record_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    run_id: str
    phase: str  # "phase1" | "phase2"
    llm_engine: str  # "gpt-4o" | "gemini-1.5-pro"
    query_id: str
    query_text: str
    response_text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    rag_context_used: bool = False
    created_at: str = Field(default_factory=_now)


class VisibilityMetrics(BaseModel):
    """Computed visibility metrics for a single QueryRecord."""

    metric_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    record_id: str
    run_id: str
    phase: str
    llm_engine: str
    query_id: str

    # Core metrics
    mention_rate: float  # 0 or 1 per response; aggregated to % at run level
    prominence_score: float  # 0-1, 1 = mentioned first
    sentiment_score: float  # -1 to 1
    share_of_voice: float  # 0-1, mistral_mentions / total_model_mentions
    recommendation_rate: float  # 0 or 1; 1 if Mistral is top recommendation
    consistency_score: Optional[float] = None  # populated after full run aggregation

    # Supporting detail
    mistral_mention_count: int = 0
    total_model_mentions: int = 0
    first_mention_position: Optional[float] = None  # normalised 0-1 char position
    top_recommended_model: Optional[str] = None
    models_mentioned: list[str] = Field(default_factory=list)

    created_at: str = Field(default_factory=_now)


class RunSummary(BaseModel):
    """Aggregated metrics for an entire run (all queries + engines)."""

    run_id: str
    phase: str
    started_at: str
    finished_at: str = Field(default_factory=_now)

    total_queries: int
    total_responses: int

    # Aggregated across all responses
    avg_mention_rate: float
    avg_prominence_score: float
    avg_sentiment_score: float
    avg_share_of_voice: float
    avg_recommendation_rate: float
    avg_consistency_score: float

    # Per-engine breakdown (stored as JSON string in BQ)
    per_engine_metrics: dict = Field(default_factory=dict)

    # Delta vs baseline (populated for phase2 runs)
    delta_mention_rate: Optional[float] = None
    delta_prominence_score: Optional[float] = None
    delta_sentiment_score: Optional[float] = None
    delta_share_of_voice: Optional[float] = None
    delta_recommendation_rate: Optional[float] = None
