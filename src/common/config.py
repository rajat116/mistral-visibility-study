"""Central configuration loaded from environment variables."""
import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    # ------------------------------------------------------------------ #
    # Brand being studied — the ONLY section companies need to change
    # ------------------------------------------------------------------ #
    brand_name: str = field(
        default_factory=lambda: os.getenv("BRAND_NAME", "Mistral AI")
    )
    # Short slug used in regex matching (lowercase, no spaces)
    brand_slug: str = field(
        default_factory=lambda: os.getenv("BRAND_SLUG", "mistral")
    )
    # Comma-separated aliases to catch (e.g. "mistral,mixtral,mistral ai")
    brand_aliases: str = field(
        default_factory=lambda: os.getenv("BRAND_ALIASES", "mistral,mixtral")
    )
    # One-line description of what the brand does
    brand_description: str = field(
        default_factory=lambda: os.getenv(
            "BRAND_DESCRIPTION",
            "a French AI company offering open and closed-weight large language models",
        )
    )
    # The product/market category (used to generate queries)
    brand_category: str = field(
        default_factory=lambda: os.getenv("BRAND_CATEGORY", "large language model provider")
    )
    # Comma-separated list of known competitors to track for Share of Voice
    competitor_names: str = field(
        default_factory=lambda: os.getenv(
            "COMPETITOR_NAMES",
            "openai,gpt-4,gpt4,chatgpt,claude,anthropic,gemini,google,llama,meta,cohere,falcon",
        )
    )

    # ------------------------------------------------------------------ #
    # API Keys
    # ------------------------------------------------------------------ #
    openai_api_key: str = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY", "")
    )
    gemini_api_key: str = field(
        default_factory=lambda: os.getenv("GEMINI_API_KEY", "")
    )

    # ------------------------------------------------------------------ #
    # GCP
    # ------------------------------------------------------------------ #
    gcp_project_id: str = field(
        default_factory=lambda: os.getenv("GCP_PROJECT_ID", "")
    )
    gcp_region: str = field(default_factory=lambda: os.getenv("GCP_REGION", "US"))
    gcs_bucket_name: str = field(
        default_factory=lambda: os.getenv("GCS_BUCKET_NAME", "ai-visibility-raw-responses")
    )
    bq_dataset: str = field(
        default_factory=lambda: os.getenv("BQ_DATASET", "ai_visibility")
    )
    google_application_credentials: str = field(
        default_factory=lambda: os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")
    )

    # ------------------------------------------------------------------ #
    # LLM Models
    # ------------------------------------------------------------------ #
    openai_model: str = field(
        default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4o")
    )
    gemini_model: str = field(
        default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    )

    # ------------------------------------------------------------------ #
    # Phase 2 RAG
    # ------------------------------------------------------------------ #
    rag_chunk_size: int = field(
        default_factory=lambda: int(os.getenv("RAG_CHUNK_SIZE", "512"))
    )
    rag_chunk_overlap: int = field(
        default_factory=lambda: int(os.getenv("RAG_CHUNK_OVERLAP", "64"))
    )
    rag_top_k: int = field(
        default_factory=lambda: int(os.getenv("RAG_TOP_K", "5"))
    )

    # ------------------------------------------------------------------ #
    # Scheduler & Misc
    # ------------------------------------------------------------------ #
    schedule_cron: str = field(
        default_factory=lambda: os.getenv("SCHEDULE_CRON", "0 9 * * 1")
    )
    log_level: str = field(
        default_factory=lambda: os.getenv("LOG_LEVEL", "INFO")
    )
    run_id_prefix: str = field(
        default_factory=lambda: os.getenv("RUN_ID_PREFIX", "visibility")
    )
    enable_gemini: bool = field(
        default_factory=lambda: os.getenv("ENABLE_GEMINI", "false").lower() == "true"
    )

    # ------------------------------------------------------------------ #
    # Computed helpers
    # ------------------------------------------------------------------ #
    @property
    def brand_alias_list(self) -> list[str]:
        return [a.strip().lower() for a in self.brand_aliases.split(",") if a.strip()]

    @property
    def competitor_list(self) -> list[str]:
        return [c.strip().lower() for c in self.competitor_names.split(",") if c.strip()]


# Singleton
config = Config()
