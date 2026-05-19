"""Google Cloud Storage client — stores raw JSON responses."""
from __future__ import annotations

from pathlib import Path

from google.cloud import storage

from src.common.config import config
from src.common.logger import get_logger
from src.common.models import QueryRecord

logger = get_logger(__name__)


class GCSClient:
    def __init__(self) -> None:
        self._client = storage.Client(project=config.gcp_project_id)
        self._bucket_name = config.gcs_bucket_name
        self._bucket = self._get_or_create_bucket()

    def _get_or_create_bucket(self) -> storage.Bucket:
        try:
            bucket = self._client.get_bucket(self._bucket_name)
            logger.info("GCS bucket exists: %s", self._bucket_name)
        except Exception:
            bucket = self._client.create_bucket(
                self._bucket_name, location=config.gcp_region
            )
            logger.info("GCS bucket created: %s", self._bucket_name)
        return bucket

    def upload_record(self, record: QueryRecord) -> str:
        """Upload a single QueryRecord as a JSON blob. Returns the GCS URI."""
        blob_path = (
            f"raw_responses/{record.phase}/{record.run_id}/"
            f"{record.llm_engine}/{record.query_id}/{record.record_id}.json"
        )
        blob = self._bucket.blob(blob_path)
        blob.upload_from_string(
            record.model_dump_json(indent=2),
            content_type="application/json",
        )
        uri = f"gs://{self._bucket_name}/{blob_path}"
        logger.debug("Uploaded record to GCS: %s", uri)
        return uri

    def upload_string(self, content: str, blob_path: str, content_type: str = "text/plain") -> str:
        """Upload arbitrary string content to GCS. Returns the GCS URI."""
        blob = self._bucket.blob(blob_path)
        blob.upload_from_string(content, content_type=content_type)
        uri = f"gs://{self._bucket_name}/{blob_path}"
        logger.debug("Uploaded content to GCS: %s", uri)
        return uri

    def download_string(self, blob_path: str) -> str:
        """Download blob content as string."""
        blob = self._bucket.blob(blob_path)
        return blob.download_as_text()

    def list_blobs(self, prefix: str) -> list[str]:
        """List blob names under a prefix."""
        return [b.name for b in self._client.list_blobs(self._bucket_name, prefix=prefix)]
