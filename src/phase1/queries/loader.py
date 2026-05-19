"""Load and validate queries from the query bank JSON file."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from pydantic import BaseModel

_DEFAULT_PATH = Path(__file__).parents[3] / "data" / "queries" / "query_bank.json"


class Query(BaseModel):
    query_id: str
    category: str
    text: str


def load_queries(path: Optional[Path] = None) -> list[Query]:
    """Return all queries from the query bank."""
    source = path or _DEFAULT_PATH
    with open(source) as f:
        raw = json.load(f)
    return [Query(**q) for q in raw]
