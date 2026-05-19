"""
RAG pipeline — build a FAISS vector index from synthetic content.

Uses OpenAI text-embedding-3-small for embeddings.
Stores the index locally and also uploads to GCS for persistence.
"""
from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import openai
import faiss

from src.common.config import config
from src.common.logger import get_logger

logger = get_logger(__name__)

_INDEX_DIR = Path(__file__).parents[4] / "data" / "results" / "rag_index"
_EMBEDDING_MODEL = "text-embedding-3-small"
_EMBED_DIM = 1536


@dataclass
class Chunk:
    chunk_id: str
    content_id: str
    content_type: str
    title: str
    text: str
    embedding: Optional[list[float]] = field(default=None, repr=False)


def _chunk_text(
    text: str,
    content_id: str,
    content_type: str,
    title: str,
    chunk_size: int,
    overlap: int,
) -> list[Chunk]:
    """Split text into overlapping chunks."""
    words = text.split()
    chunks = []
    i = 0
    idx = 0
    while i < len(words):
        chunk_words = words[i : i + chunk_size]
        chunk_text = " ".join(chunk_words)
        chunks.append(
            Chunk(
                chunk_id=f"{content_id}_chunk{idx:03d}",
                content_id=content_id,
                content_type=content_type,
                title=title,
                text=chunk_text,
            )
        )
        i += chunk_size - overlap
        idx += 1
    return chunks


def _embed_texts(texts: list[str], client: openai.OpenAI) -> list[list[float]]:
    """Batch embed texts using OpenAI embeddings API."""
    # API supports up to 2048 inputs per request; batch conservatively
    batch_size = 100
    all_embeddings: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        response = client.embeddings.create(model=_EMBEDDING_MODEL, input=batch)
        all_embeddings.extend([item.embedding for item in response.data])
        logger.debug("Embedded batch %d-%d", i, i + len(batch))
    return all_embeddings


class RAGIndexer:
    """Builds and persists a FAISS vector index from content pieces."""

    def __init__(self, index_dir: Optional[Path] = None) -> None:
        self._index_dir = index_dir or _INDEX_DIR
        self._index_dir.mkdir(parents=True, exist_ok=True)
        self._client = openai.OpenAI(api_key=config.openai_api_key)
        self._chunks: list[Chunk] = []
        self._index: Optional[faiss.Index] = None

    def build_from_files(self, content_dir: Path) -> None:
        """Load all .md files from content_dir, chunk, embed, and index."""
        md_files = list(content_dir.glob("*.md"))
        if not md_files:
            raise FileNotFoundError(f"No .md files found in {content_dir}")

        all_chunks: list[Chunk] = []
        for md_path in md_files:
            content_id = md_path.stem
            text = md_path.read_text()

            # Try to load metadata
            meta_path = content_dir / f"{content_id}.json"
            if meta_path.exists():
                meta = json.loads(meta_path.read_text())
                content_type = meta.get("content_type", "unknown")
                title = meta.get("title", content_id)
            else:
                content_type = "unknown"
                title = content_id

            chunks = _chunk_text(
                text=text,
                content_id=content_id,
                content_type=content_type,
                title=title,
                chunk_size=config.rag_chunk_size,
                overlap=config.rag_chunk_overlap,
            )
            all_chunks.extend(chunks)
            logger.info("Chunked %s → %d chunks", content_id, len(chunks))

        logger.info("Total chunks: %d — embedding...", len(all_chunks))
        texts = [c.text for c in all_chunks]
        embeddings = _embed_texts(texts, self._client)

        for chunk, emb in zip(all_chunks, embeddings):
            chunk.embedding = emb

        self._chunks = all_chunks
        self._build_faiss_index(embeddings)
        self.save()

    def _build_faiss_index(self, embeddings: list[list[float]]) -> None:
        vectors = np.array(embeddings, dtype="float32")
        faiss.normalize_L2(vectors)
        self._index = faiss.IndexFlatIP(_EMBED_DIM)  # inner-product on normalised = cosine
        self._index.add(vectors)
        logger.info("FAISS index built with %d vectors", self._index.ntotal)

    def save(self) -> None:
        """Persist index and chunk metadata to disk."""
        faiss.write_index(self._index, str(self._index_dir / "index.faiss"))
        chunks_data = [
            {
                "chunk_id": c.chunk_id,
                "content_id": c.content_id,
                "content_type": c.content_type,
                "title": c.title,
                "text": c.text,
            }
            for c in self._chunks
        ]
        (self._index_dir / "chunks.json").write_text(json.dumps(chunks_data, indent=2))
        logger.info("RAG index saved to %s", self._index_dir)

    def load(self) -> None:
        """Load a previously saved index from disk."""
        index_path = self._index_dir / "index.faiss"
        chunks_path = self._index_dir / "chunks.json"
        if not index_path.exists():
            raise FileNotFoundError(f"No FAISS index at {index_path}")
        self._index = faiss.read_index(str(index_path))
        raw = json.loads(chunks_path.read_text())
        self._chunks = [
            Chunk(
                chunk_id=c["chunk_id"],
                content_id=c["content_id"],
                content_type=c["content_type"],
                title=c["title"],
                text=c["text"],
            )
            for c in raw
        ]
        logger.info("RAG index loaded: %d chunks", len(self._chunks))

    def retrieve(self, query: str, top_k: Optional[int] = None) -> list[Chunk]:
        """Retrieve top-k most relevant chunks for a query."""
        if self._index is None:
            raise RuntimeError("Index not loaded. Call build_from_files() or load() first.")
        k = top_k or config.rag_top_k
        response = self._client.embeddings.create(model=_EMBEDDING_MODEL, input=[query])
        q_vec = np.array([response.data[0].embedding], dtype="float32")
        faiss.normalize_L2(q_vec)
        distances, indices = self._index.search(q_vec, k)
        results = []
        for idx in indices[0]:
            if idx >= 0:
                results.append(self._chunks[idx])
        return results

    def build_context_string(self, chunks: list[Chunk]) -> str:
        """Format retrieved chunks into a context string for LLM injection."""
        parts = []
        for i, chunk in enumerate(chunks, 1):
            parts.append(f"[Source {i}: {chunk.title}]\n{chunk.text}")
        return "\n\n".join(parts)
