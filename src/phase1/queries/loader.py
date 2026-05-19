"""Load and render queries from the query bank template file."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from pydantic import BaseModel

from src.common.config import config

_DEFAULT_PATH = Path(__file__).parents[3] / "data" / "queries" / "query_bank.json"


class Query(BaseModel):
    query_id: str
    category: str
    text: str  # rendered from template


def load_queries(path: Optional[Path] = None) -> list[Query]:
    """
    Load query templates and render them with the configured brand/category.

    Templates use {brand} and {category} placeholders:
      "What is the best {category} for production?" 
      → "What is the best large language model provider for production?"
    """
    source = path or _DEFAULT_PATH
    with open(source) as f:
        raw = json.load(f)

    queries = []
    for q in raw:
        template = q.get("template", q.get("text", ""))
        rendered = template.format(
            brand=config.brand_name,
            category=config.brand_category,
        )
        queries.append(Query(
            query_id=q["query_id"],
            category=q["category"],
            text=rendered,
        ))
    return queries
