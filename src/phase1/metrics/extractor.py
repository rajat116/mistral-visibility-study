"""
Extract and compute all six visibility metrics from a raw LLM response.

All brand references come from config — no hardcoded brand names.

Metrics
-------
1. Mention Rate        – Binary: did the response mention the brand?
2. Prominence Score    – How early the brand appears (1 = first word, 0 = absent).
3. Sentiment Score     – VADER compound sentiment of brand-adjacent sentences (-1..1).
4. Share of Voice      – brand_mentions / total_competitor_mentions.
5. Recommendation Rate – Binary: is the brand the top recommendation?
6. Consistency Score   – Computed at run level in aggregator.py.
"""
from __future__ import annotations

import re
from typing import Optional

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from src.common.config import config
from src.common.models import QueryRecord, VisibilityMetrics
from src.common.logger import get_logger

logger = get_logger(__name__)

_ANALYZER = SentimentIntensityAnalyzer()


def _build_brand_pattern() -> re.Pattern:
    """Build a regex that matches any brand alias."""
    aliases = config.brand_alias_list
    parts = sorted([re.escape(a) for a in aliases], key=len, reverse=True)
    return re.compile(r"\b(?:" + "|".join(parts) + r")\b", re.IGNORECASE)


def _build_competitor_patterns() -> list[re.Pattern]:
    return [
        re.compile(rf"\b{re.escape(c)}\b", re.IGNORECASE)
        for c in config.competitor_list
    ]


def _count_brand(text: str) -> int:
    return len(_build_brand_pattern().findall(text))


def _count_all_brands(text: str) -> int:
    """Count brand + all competitor mentions."""
    total = _count_brand(text)
    for pat in _build_competitor_patterns():
        total += len(pat.findall(text))
    return total


def _prominence_score(text: str) -> tuple[float, Optional[float]]:
    match = _build_brand_pattern().search(text)
    if not match:
        return 0.0, None
    pos = match.start() / max(len(text), 1)
    return round(1.0 - pos, 4), round(pos, 4)


def _sentiment_score(text: str) -> float:
    pat = _build_brand_pattern()
    sentences = re.split(r"(?<=[.!?])\s+", text)
    brand_sentences = [s for s in sentences if pat.search(s)]
    if not brand_sentences:
        return 0.0
    scores = [_ANALYZER.polarity_scores(s)["compound"] for s in brand_sentences]
    return round(sum(scores) / len(scores), 4)


def _share_of_voice(text: str) -> tuple[float, int, int]:
    brand_count = _count_brand(text)
    total_count = _count_all_brands(text)
    sov = round(brand_count / total_count, 4) if total_count > 0 else 0.0
    return sov, brand_count, total_count


def _recommendation_rate(text: str) -> tuple[float, Optional[str]]:
    """Returns (1.0, brand_slug) if brand is top recommendation, else (0.0, other_model)."""
    brand_pat = _build_brand_pattern()
    brand_slug = config.brand_slug

    top_rec_patterns = [
        re.compile(r"\brecommend\b.{0,80}" + brand_slug, re.IGNORECASE | re.DOTALL),
        re.compile(brand_slug + r".{0,80}\brecommend\b", re.IGNORECASE | re.DOTALL),
        re.compile(r"\bbest\b.{0,60}" + brand_slug, re.IGNORECASE | re.DOTALL),
        re.compile(brand_slug + r".{0,60}\bbest\b", re.IGNORECASE | re.DOTALL),
        re.compile(r"\btop\s+(?:pick|choice|recommendation)\b.{0,80}" + brand_slug, re.IGNORECASE | re.DOTALL),
    ]
    for pat in top_rec_patterns:
        if pat.search(text):
            return 1.0, brand_slug

    for competitor in config.competitor_list:
        for template in [
            re.compile(rf"\brecommend\b.{{0,80}}{re.escape(competitor)}", re.IGNORECASE | re.DOTALL),
            re.compile(rf"\bbest\b.{{0,60}}{re.escape(competitor)}", re.IGNORECASE | re.DOTALL),
        ]:
            if template.search(text):
                return 0.0, competitor
    return 0.0, None


def _models_mentioned(text: str) -> list[str]:
    found = []
    brand_pat = _build_brand_pattern()
    if brand_pat.search(text):
        found.append(config.brand_slug)
    for competitor in config.competitor_list:
        if re.search(rf"\b{re.escape(competitor)}\b", text, re.IGNORECASE):
            found.append(competitor)
    return list(dict.fromkeys(found))  # deduplicate, preserve order


def compute_metrics(record: QueryRecord) -> VisibilityMetrics:
    """Compute all extractable metrics for a single QueryRecord."""
    text = record.response_text

    mention_rate = 1.0 if _count_brand(text) > 0 else 0.0
    prominence, first_pos = _prominence_score(text)
    sentiment = _sentiment_score(text)
    sov, brand_count, total_count = _share_of_voice(text)
    rec_rate, top_model = _recommendation_rate(text)
    models = _models_mentioned(text)

    logger.debug(
        "Metrics | engine=%s | query=%s | mention=%s | prominence=%.3f | sov=%.3f",
        record.llm_engine, record.query_id, mention_rate, prominence, sov,
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
        mistral_mention_count=brand_count,
        total_model_mentions=total_count,
        first_mention_position=first_pos,
        top_recommended_model=top_model,
        models_mentioned=models,
    )
