"""BigQuery client — stores structured metrics and run summaries."""
from __future__ import annotations

import json
from typing import Any

from google.cloud import bigquery
from google.api_core.exceptions import NotFound

from src.common.config import config
from src.common.logger import get_logger
from src.common.models import QueryRecord, VisibilityMetrics, RunSummary

logger = get_logger(__name__)

# --------------------------------------------------------------------------- #
# Table schemas
# --------------------------------------------------------------------------- #

_RAW_RESPONSES_SCHEMA = [
    bigquery.SchemaField("record_id", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("run_id", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("phase", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("llm_engine", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("query_id", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("query_text", "STRING"),
    bigquery.SchemaField("response_text", "STRING"),
    bigquery.SchemaField("prompt_tokens", "INTEGER"),
    bigquery.SchemaField("completion_tokens", "INTEGER"),
    bigquery.SchemaField("latency_ms", "FLOAT"),
    bigquery.SchemaField("rag_context_used", "BOOLEAN"),
    bigquery.SchemaField("created_at", "TIMESTAMP"),
]

_METRICS_SCHEMA = [
    bigquery.SchemaField("metric_id", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("record_id", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("run_id", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("phase", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("llm_engine", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("query_id", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("mention_rate", "FLOAT"),
    bigquery.SchemaField("prominence_score", "FLOAT"),
    bigquery.SchemaField("sentiment_score", "FLOAT"),
    bigquery.SchemaField("share_of_voice", "FLOAT"),
    bigquery.SchemaField("recommendation_rate", "FLOAT"),
    bigquery.SchemaField("consistency_score", "FLOAT"),
    bigquery.SchemaField("mistral_mention_count", "INTEGER"),
    bigquery.SchemaField("total_model_mentions", "INTEGER"),
    bigquery.SchemaField("first_mention_position", "FLOAT"),
    bigquery.SchemaField("top_recommended_model", "STRING"),
    bigquery.SchemaField("models_mentioned", "STRING"),  # stored as JSON array string
    bigquery.SchemaField("created_at", "TIMESTAMP"),
]

_RUN_SUMMARY_SCHEMA = [
    bigquery.SchemaField("run_id", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("phase", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("started_at", "TIMESTAMP"),
    bigquery.SchemaField("finished_at", "TIMESTAMP"),
    bigquery.SchemaField("total_queries", "INTEGER"),
    bigquery.SchemaField("total_responses", "INTEGER"),
    bigquery.SchemaField("avg_mention_rate", "FLOAT"),
    bigquery.SchemaField("avg_prominence_score", "FLOAT"),
    bigquery.SchemaField("avg_sentiment_score", "FLOAT"),
    bigquery.SchemaField("avg_share_of_voice", "FLOAT"),
    bigquery.SchemaField("avg_recommendation_rate", "FLOAT"),
    bigquery.SchemaField("avg_consistency_score", "FLOAT"),
    bigquery.SchemaField("per_engine_metrics", "STRING"),  # JSON string
    bigquery.SchemaField("delta_mention_rate", "FLOAT"),
    bigquery.SchemaField("delta_prominence_score", "FLOAT"),
    bigquery.SchemaField("delta_sentiment_score", "FLOAT"),
    bigquery.SchemaField("delta_share_of_voice", "FLOAT"),
    bigquery.SchemaField("delta_recommendation_rate", "FLOAT"),
]

TABLE_CONFIGS: dict[str, list[bigquery.SchemaField]] = {
    "raw_responses": _RAW_RESPONSES_SCHEMA,
    "visibility_metrics": _METRICS_SCHEMA,
    "run_summaries": _RUN_SUMMARY_SCHEMA,
}


class BigQueryClient:
    def __init__(self) -> None:
        self._client = bigquery.Client(project=config.gcp_project_id)
        self._dataset_id = config.bq_dataset
        self._ensure_dataset()
        self._ensure_tables()

    # ---------------------------------------------------------------------- #
    # Setup
    # ---------------------------------------------------------------------- #

    def _ensure_dataset(self) -> None:
        dataset_ref = f"{config.gcp_project_id}.{self._dataset_id}"
        try:
            self._client.get_dataset(dataset_ref)
            logger.info("BigQuery dataset exists: %s", dataset_ref)
        except NotFound:
            dataset = bigquery.Dataset(dataset_ref)
            dataset.location = config.gcp_region
            self._client.create_dataset(dataset)
            logger.info("BigQuery dataset created: %s", dataset_ref)

    def _ensure_tables(self) -> None:
        for table_name, schema in TABLE_CONFIGS.items():
            table_ref = f"{config.gcp_project_id}.{self._dataset_id}.{table_name}"
            try:
                self._client.get_table(table_ref)
                logger.info("BigQuery table exists: %s", table_name)
            except NotFound:
                table = bigquery.Table(table_ref, schema=schema)
                self._client.create_table(table)
                logger.info("BigQuery table created: %s", table_name)

    # ---------------------------------------------------------------------- #
    # Inserts
    # ---------------------------------------------------------------------- #

    def insert_raw_response(self, record: QueryRecord) -> None:
        row = record.model_dump()
        row["created_at"] = row["created_at"]  # already ISO string; BQ accepts it
        self._stream_rows("raw_responses", [row])

    def insert_metrics(self, metrics: VisibilityMetrics) -> None:
        row = metrics.model_dump()
        row["models_mentioned"] = json.dumps(row["models_mentioned"])
        self._stream_rows("visibility_metrics", [row])

    def insert_run_summary(self, summary: RunSummary) -> None:
        row = summary.model_dump()
        row["per_engine_metrics"] = json.dumps(row["per_engine_metrics"])
        self._stream_rows("run_summaries", [row])

    def _stream_rows(self, table_name: str, rows: list[dict[str, Any]]) -> None:
        table_ref = f"{config.gcp_project_id}.{self._dataset_id}.{table_name}"
        errors = self._client.insert_rows_json(table_ref, rows)
        if errors:
            logger.error("BigQuery insert errors for %s: %s", table_name, errors)
            raise RuntimeError(f"BigQuery insert failed: {errors}")
        logger.debug("Inserted %d row(s) into %s", len(rows), table_name)

    # ---------------------------------------------------------------------- #
    # Queries
    # ---------------------------------------------------------------------- #

    def get_latest_phase1_summary(self) -> dict | None:
        """Fetch the most recent Phase 1 run summary."""
        query = f"""
            SELECT *
            FROM `{config.gcp_project_id}.{self._dataset_id}.run_summaries`
            WHERE phase = 'phase1'
            ORDER BY finished_at DESC
            LIMIT 1
        """
        rows = list(self._client.query(query).result())
        return dict(rows[0]) if rows else None

    def get_run_summaries(self, phase: str | None = None, limit: int = 50) -> list[dict]:
        """Fetch recent run summaries, optionally filtered by phase."""
        where = f"WHERE phase = '{phase}'" if phase else ""
        query = f"""
            SELECT *
            FROM `{config.gcp_project_id}.{self._dataset_id}.run_summaries`
            {where}
            ORDER BY finished_at DESC
            LIMIT {limit}
        """
        return [dict(row) for row in self._client.query(query).result()]

    def get_metrics_for_run(self, run_id: str) -> list[dict]:
        query = f"""
            SELECT *
            FROM `{config.gcp_project_id}.{self._dataset_id}.visibility_metrics`
            WHERE run_id = '{run_id}'
        """
        return [dict(row) for row in self._client.query(query).result()]
