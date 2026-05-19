"""
One-time GCP setup script.

Creates:
  - BigQuery dataset and tables
  - GCS bucket

Run this once before the first pipeline execution.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.common.config import config
from src.common.logger import get_logger
from src.phase1.storage.bigquery_client import BigQueryClient
from src.phase1.storage.gcs_client import GCSClient

logger = get_logger("setup_gcp")


def main() -> None:
    logger.info("Setting up GCP resources for project: %s", config.gcp_project_id)

    logger.info("--- BigQuery ---")
    bq = BigQueryClient()
    logger.info("BigQuery dataset + tables ready.")

    logger.info("--- GCS ---")
    gcs = GCSClient()
    logger.info("GCS bucket ready: gs://%s", config.gcs_bucket_name)

    logger.info("GCP setup complete.")


if __name__ == "__main__":
    main()
