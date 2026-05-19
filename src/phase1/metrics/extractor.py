"""
Extract and compute all six visibility metrics from a raw LLM response.

Metrics
-------
1. Mention Rate        – Binary: did the response mention Mistral?
2. Prominence Score    – How early does Mistral appear (1 = first word, 0 = not present)?
3. Sentiment Score     – VADER compound sentiment of Mistral-adjacent sentences (-1..1).
4. Share of Voice      – Mistral mentions / total competitor model mentions.
5. Recommendation Rate – Binary: is Mistral the top / primary recommendation?
6. Consistency Score   – Computed at run level in aggregator.py (not here).
"""
from __future__ import annotations

import re
from typing import Optional

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from src.common.models import QueryRecord, VisibilityMetrics
from src.common.logger import get_logger

logger = get_logger(__name__)

_ANALYZER = SentimentIntensityAnalyzer()

# Models tracked for Share of Voice
TRACKED_MODELS: list[str] = [
    "mistral",
    "gpt-4",
    "gpt-3",
    "gpt4",
    "chatgpt",
    "openai",
    "claude",
    "anthropic",
    "gemini",
    "google",
    "llama",
    "meta",
    "cohere",
    "command",
    "falcon",
    "mixtral",
]

# Patterns that signal Mistral is the top recommendation
_TOP_REC_PATTERNS: list[re.Pattern] = [
    re.compile(r"\brecommend\b.{0,80}mistral", re.IGNORECASE | re.DOTALL),
    re.compile(r"\bmistral\b.{0,80}\brecommend", re.IGNORECASE | re.DOTALL),
    re.compile(r"\bbest\b.{0,60}mistral", re.IGNORECASE | re.DOTALL),
    re.compile(r"\bmistral\b.{0,60}\bbest\b", re.IGNORECASE | re.DOTALL),
    re.compile(r"^[^.]*mistral[^.]*\.", re.IGNORECASE | re.MULTILINE),  # 1st sentence
    re.compile(r"\btop\s+(?:pick|choice|recommendation)\b.{0,80}mistral", re.IGNORECASE | re.DOTALL),
    re.compile(r"\bmistral\b.{0,80}\btop\s+(?:pick|choice|recommendation)\b", re.IGNORECASE | re.DOTALL),
]


def _count_pattern(text: str, pattern: str) -> int:
    return len(re.findall(pattern, text, re.IGNORECASE))


def _mistral_mentions(text: str) -> int:
    return _count_pattern(text, r"\bmistral\b")


def _total_model_mentions(text: str) -> int:
    total = 0
    for model in TRACKED_MODELS:
        total += _count_pattern(text, rf"\b{re.escape(model)}\b")
    return total


def _prominence_score(text: str) -> tuple[float, Optional[float]]:
    """
    Returns (score, normalised_position).
    score = 1 - normalised_position if Mistral mentioned, else 0.
    Normalised position = first_match_char_index / len(text).
    """
    match = re.search(r"\bmistral\b", text, re.IGNORECASE)
    if not match:
        return 0.0, None
    pos = match.start() / max(len(text), 1)
    return round(1.0 - pos, 4), round(pos, 4)


def _sentiment_score(text: str) -> float:
    """
    Average VADER compound score of sentences that contain 'mistral'.
    Falls back to 0.0 if no sentences found.
    """
    sentences = re.split(r"(?<=[.!?])\s+", text)
    mistral_sentences = [s for s in sentences if re.search(r"\bmistral\b", s, re.IGNORECASE)]
    if not mistral_sentences:
        return 0.0
    scores = [_ANALYZER.polarity_scores(s)["compound"] for s in mistral_sentences]
    return round(sum(scores) / len(scores), 4)


def _share_of_voice(text: str) -> tuple[float, int, int]:
    """Returns (sov, mistral_count, total_count)."""
    mistral_count = _mistral_mentions(text)
    total_count = _total_model_mentions(text)
    sov = round(mistral_count / total_count, 4) if total_count > 0 else 0.0
    return sov, mistral_count, total_count


def _recommendation_rate(text: str) -> tuple[float, Optional[str]]:
    """
    Returns (score, top_recommended_model).
    Score = 1.0 if Mistral is top recommendation, else 0.0.
    Also extracts which model appears to be the top recommendation.
    """
    for pattern in _TOP_REC_PATTERNS:
        if pattern.search(text):
            return 1.0, "mistral"

    # Try to detect which model IS recommended if not Mistral
    for model in ["gpt-4", "claude", "gemini", "llama", "cohere"]:
        for pat in [
            re.compile(rf"\brecommend\b.{{0,80}}{re.escape(model)}", re.IGNORECASE | re.DOTALL),
            re.compile(rf"\bbest\b.{{0,60}}{re.escape(model)}", re.IGNORECASE | re.DOTALL),
        ]:
            if pat.search(text):
                return 0.0, model
    return 0.0, None


def _models_mentioned(text: str) -> list[str]:
    found = []
    for model in TRACKED_MODELS:
        if re.search(rf"\b{re.escape(model)}\b", text, re.IGNORECASE):
            found.append(model)
    return found


def compute_metrics(record: QueryRecord) -> VisibilityMetrics:
    """Compute all extractable metrics for a single QueryRecord."""
    text = record.response_text

    mention_rate = 1.0 if _mistral_mentions(text) > 0 else 0.0
    prominence, first_pos = _prominence_score(text)
    sentiment = _sentiment_score(text)
    sov, mistral_count, total_count = _share_of_voice(text)
    rec_rate, top_model = _recommendation_rate(text)
    models = _models_mentioned(text)

    logger.debug(
        "Metrics | engine=%s | query=%s | mention=%s | prominence=%.3f | sov=%.3f",
        record.llm_engine,
        record.query_id,
        mention_rate,
        prominence,
        sov,
    )

    return VisibilityMetrics(
        record_id=record.record_id,
        run_id=record.run_id,
        phase=record.phase,
        llm_engine=record.llm_engine,
        query_id=record.query_id,
        mention_rate=mention_rate,
        prominence_score=prominence,
        sentiment_score=sentiment,
        share_of_voice=sov,
        recommendation_rate=rec_rate,
        mistral_mention_count=mistral_count,
        total_model_mentions=total_count,
        first_mention_position=first_pos,
        top_recommended_model=top_model,
        models_mentioned=models,
    )
