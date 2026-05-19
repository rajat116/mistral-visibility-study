"""LLM query engines for Phase 1 and Phase 2."""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Optional

import openai
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted

from src.common.config import config
from src.common.logger import get_logger
from src.common.models import QueryRecord

logger = get_logger(__name__)

SYSTEM_PROMPT = (
    "You are a helpful AI assistant with deep knowledge of large language models "
    "and AI tools. Answer the user's question thoroughly and objectively, mentioning "
    "relevant models and providers where appropriate."
)


class LLMEngine(ABC):
    """Abstract base for any LLM query engine."""

    @property
    @abstractmethod
    def engine_name(self) -> str: ...

    @abstractmethod
    def query(
        self,
        run_id: str,
        phase: str,
        query_id: str,
        query_text: str,
        context: Optional[str] = None,
    ) -> QueryRecord: ...


class OpenAIEngine(LLMEngine):
    """Queries OpenAI chat completions (GPT-4o by default)."""

    def __init__(self) -> None:
        self._client = openai.OpenAI(api_key=config.openai_api_key)
        self._model = config.openai_model

    @property
    def engine_name(self) -> str:
        return self._model

    def query(
        self,
        run_id: str,
        phase: str,
        query_id: str,
        query_text: str,
        context: Optional[str] = None,
    ) -> QueryRecord:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        user_content = query_text
        if context:
            user_content = (
                f"Use the following context to inform your answer:\n\n"
                f"{context}\n\n---\n\nQuestion: {query_text}"
            )
        messages.append({"role": "user", "content": user_content})

        logger.info("OpenAI query | run=%s | query=%s", run_id, query_id)
        t0 = time.monotonic()
        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=0.3,
            max_tokens=1024,
        )
        latency_ms = (time.monotonic() - t0) * 1000

        choice = response.choices[0]
        usage = response.usage

        return QueryRecord(
            run_id=run_id,
            phase=phase,
            llm_engine=self._model,
            query_id=query_id,
            query_text=query_text,
            response_text=choice.message.content or "",
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            latency_ms=latency_ms,
            rag_context_used=context is not None,
        )


class GeminiEngine(LLMEngine):
    """Queries Google Gemini via the generativeai SDK."""

    def __init__(self) -> None:
        genai.configure(api_key=config.gemini_api_key)
        self._model_name = config.gemini_model
        self._model = genai.GenerativeModel(
            model_name=self._model_name,
            system_instruction=SYSTEM_PROMPT,
        )

    @property
    def engine_name(self) -> str:
        return self._model_name

    def query(
        self,
        run_id: str,
        phase: str,
        query_id: str,
        query_text: str,
        context: Optional[str] = None,
    ) -> QueryRecord:
        user_content = query_text
        if context:
            user_content = (
                f"Use the following context to inform your answer:\n\n"
                f"{context}\n\n---\n\nQuestion: {query_text}"
            )

        logger.info("Gemini query | run=%s | query=%s", run_id, query_id)
        t0 = time.monotonic()

        # Retry up to 3 times on rate limit (free tier: 15 req/min)
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self._model.generate_content(
                    user_content,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.3,
                        max_output_tokens=1024,
                    ),
                )
                break
            except ResourceExhausted:
                if attempt < max_retries - 1:
                    wait = 60 * (attempt + 1)  # 60s, 120s
                    logger.warning(
                        "Gemini rate limited. Waiting %ds before retry %d/%d...",
                        wait, attempt + 1, max_retries - 1,
                    )
                    time.sleep(wait)
                else:
                    raise

        latency_ms = (time.monotonic() - t0) * 1000

        response_text = response.text if response.parts else ""
        usage = response.usage_metadata
        prompt_tokens = usage.prompt_token_count if usage else 0
        completion_tokens = usage.candidates_token_count if usage else 0

        return QueryRecord(
            run_id=run_id,
            phase=phase,
            llm_engine=self._model_name,
            query_id=query_id,
            query_text=query_text,
            response_text=response_text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
            rag_context_used=context is not None,
        )


def get_all_engines() -> list[LLMEngine]:
    """Instantiate and return all configured engines."""
    import os
    engines: list[LLMEngine] = [OpenAIEngine()]
    if os.getenv("ENABLE_GEMINI", "false").lower() == "true":
        engines.append(GeminiEngine())
    return engines
