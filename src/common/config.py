"""Central configuration loaded from environment variables."""
import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    # --- API Keys ---
    openai_api_key: str = field(default_factory=lambda: os.environ["OPENAI_API_KEY"])
    gemini_api_key: str = field(default_factory=lambda: os.environ["GEMINI_API_KEY"])

    # --- GCP ---
    gcp_project_id: str = field(
        default_factory=lambda: os.getenv(
            "GCP_PROJECT_ID", "project-553727f2-fa51-4dd8-8e0"
        )
    )
    gcp_region: str = field(default_factory=lambda: os.getenv("GCP_REGION", "US"))
    gcs_bucket_name: str = field(
        default_factory=lambda: os.getenv(
            "GCS_BUCKET_NAME", "mistral-visibility-raw-responses"
        )
    )
    bq_dataset: str = field(
        default_factory=lambda: os.getenv("BQ_DATASET", "mistral_visibility")
    )
    google_application_credentials: str = field(
        default_factory=lambda: os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")
    )

    # --- Models ---
    openai_model: str = field(
        default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4o")
    )
    gemini_model: str = field(
        default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    )

    # --- Phase 2 RAG ---
    rag_chunk_size: int = field(
        default_factory=lambda: int(os.getenv("RAG_CHUNK_SIZE", "512"))
    )
    rag_chunk_overlap: int = field(
        default_factory=lambda: int(os.getenv("RAG_CHUNK_OVERLAP", "64"))
    )
    rag_top_k: int = field(
        default_factory=lambda: int(os.getenv("RAG_TOP_K", "5"))
    )

    # --- Scheduler ---
    schedule_cron: str = field(
        default_factory=lambda: os.getenv("SCHEDULE_CRON", "0 9 * * 1")
    )

    # --- Misc ---
    log_level: str = field(
        default_factory=lambda: os.getenv("LOG_LEVEL", "INFO")
    )
    run_id_prefix: str = field(
        default_factory=lambda: os.getenv("RUN_ID_PREFIX", "visibility")
    )


# Singleton
config = Config()
