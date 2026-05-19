"""Unit tests for the metrics extractor."""
import pytest
from src.common.models import QueryRecord
from src.phase1.metrics.extractor import (
    compute_metrics,
    _mistral_mentions,
    _prominence_score,
    _sentiment_score,
    _share_of_voice,
    _recommendation_rate,
)

DUMMY_RUN = "test-run-001"


def _make_record(response: str) -> QueryRecord:
    return QueryRecord(
        run_id=DUMMY_RUN,
        phase="phase1",
        llm_engine="gpt-4o",
        query_id="q001",
        query_text="What's the best LLM?",
        response_text=response,
    )


class TestMentionRate:
    def test_mentioned(self):
        r = _make_record("Mistral is a great model. You should try Mistral.")
        m = compute_metrics(r)
        assert m.mention_rate == 1.0

    def test_not_mentioned(self):
        r = _make_record("GPT-4 is excellent. Claude is also good.")
        m = compute_metrics(r)
        assert m.mention_rate == 0.0

    def test_case_insensitive(self):
        r = _make_record("MISTRAL offers competitive pricing.")
        m = compute_metrics(r)
        assert m.mention_rate == 1.0


class TestProminenceScore:
    def test_first_word(self):
        score, pos = _prominence_score("Mistral is the best LLM.")
        assert score > 0.95  # very early mention

    def test_last_word(self):
        long_prefix = "x " * 200
        score, pos = _prominence_score(long_prefix + "Mistral.")
        assert score < 0.1

    def test_not_mentioned(self):
        score, pos = _prominence_score("GPT-4 and Claude are great models.")
        assert score == 0.0
        assert pos is None


class TestSentimentScore:
    def test_positive_sentence(self):
        score = _sentiment_score("Mistral is an excellent, top-performing model that I love.")
        assert score > 0.0

    def test_negative_sentence(self):
        score = _sentiment_score("Mistral unfortunately performs poorly and is unreliable.")
        assert score < 0.0

    def test_neutral_no_mention(self):
        score = _sentiment_score("GPT-4 is good.")
        assert score == 0.0


class TestShareOfVoice:
    def test_only_mistral(self):
        sov, mc, tc = _share_of_voice("Mistral is great. Mistral is fast.")
        assert mc == 2
        assert sov == 1.0

    def test_mixed(self):
        text = "Mistral is good. GPT-4 is good. Claude is good."
        sov, mc, tc = _share_of_voice(text)
        assert mc == 1
        assert tc > mc
        assert 0.0 < sov < 1.0

    def test_no_models(self):
        sov, mc, tc = _share_of_voice("The sky is blue.")
        assert sov == 0.0
        assert tc == 0


class TestRecommendationRate:
    def test_recommend_mistral(self):
        score, model = _recommendation_rate("I recommend Mistral for most production use-cases.")
        assert score == 1.0
        assert model == "mistral"

    def test_best_mistral(self):
        score, model = _recommendation_rate("The best model for your needs is Mistral.")
        assert score == 1.0

    def test_recommend_other(self):
        score, model = _recommendation_rate("I recommend GPT-4 for most tasks.")
        assert score == 0.0

    def test_no_recommendation(self):
        score, model = _recommendation_rate("There are many models available today.")
        assert score == 0.0


class TestComputeMetrics:
    def test_full_response_integration(self):
        response = (
            "Mistral AI offers an excellent balance of speed and quality. "
            "I highly recommend Mistral for production use. "
            "While GPT-4 and Claude are also strong choices, "
            "Mistral's open-weight models and competitive pricing make it outstanding."
        )
        r = _make_record(response)
        m = compute_metrics(r)

        assert m.mention_rate == 1.0
        assert m.prominence_score > 0.5
        assert m.sentiment_score > 0.0
        assert 0.0 < m.share_of_voice < 1.0
        assert m.recommendation_rate == 1.0
        assert "mistral" in m.models_mentioned
