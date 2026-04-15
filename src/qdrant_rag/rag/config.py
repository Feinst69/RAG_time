"""Runtime configuration loader.

Values are read from ``config.yaml`` at the project root.  All constants
exported here have the same names as before so the rest of the codebase is
unchanged.

Lookup order:
  1. config.yaml  (authoritative source for tunable parameters)
  2. Hard-coded fallbacks below (used only when a key is absent from the file)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

# ---------------------------------------------------------------------------
# Locate and load config.yaml
# ---------------------------------------------------------------------------

# config.py lives at src/qdrant_rag/rag/config.py → 4 parents = project root
_CONFIG_PATH: Path = Path(__file__).parents[3] / "config.yaml"


def _load() -> dict[str, Any]:
    if not _CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"config.yaml not found at {_CONFIG_PATH}. "
            "Copy config.yaml to the project root and adjust as needed."
        )
    with _CONFIG_PATH.open() as fh:
        return yaml.safe_load(fh) or {}


_cfg = _load()


def _get(section: str, key: str, default: Any) -> Any:
    return _cfg.get(section, {}).get(key, default)


# ---------------------------------------------------------------------------
# Qdrant
# ---------------------------------------------------------------------------
QDRANT_URL: str       = _get("qdrant", "url", "http://localhost:6333")
COLLECTION_NAME: str  = _get("qdrant", "collection_name", "support_tickets")

# ---------------------------------------------------------------------------
# Vector spaces
# ---------------------------------------------------------------------------
DENSE_VECTOR_NAME: str  = _get("vectors", "dense_name", "dense")
SPARSE_VECTOR_NAME: str = _get("vectors", "sparse_name", "sparse")
DENSE_VECTOR_DIM: int   = _get("vectors", "dense_dim", 48)

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
DENSE_COLBERT_MODEL: str  = _get("models", "dense",    "mixedbread-ai/mxbai-edge-colbert-v0-17m")
BM25_MODEL: str           = _get("models", "sparse",   "Qdrant/bm25")
RERANK_COLBERT_MODEL: str = _get("models", "reranker", "mixedbread-ai/mxbai-edge-colbert-v0-32m")

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
RETRIEVAL_CANDIDATES: int = _get("retrieval", "candidates", 100)
DEFAULT_TOP_K: int        = _get("retrieval", "top_k",       10)

# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------
INDEX_BATCH_SIZE: int     = _get("indexing", "batch_size",   64)
TEXT_FIELDS: list[str]    = _get("indexing", "text_fields",  ["issue_description", "resolution_notes"])

# ---------------------------------------------------------------------------
# Payload field schema  (not user-tunable → stays in code, not YAML)
# ---------------------------------------------------------------------------

KEYWORD_FIELDS: list[str] = [
    "customer_name", "customer_email", "product", "category", "priority",
    "status", "channel", "region", "customer_gender", "subscription_type",
    "operating_system", "browser", "payment_method", "language",
    "preferred_contact_time", "customer_segment",
]

INTEGER_FIELDS: list[str] = [
    "customer_age", "issue_complexity_score", "previous_tickets", "customer_tenure_months",
]

FLOAT_FIELDS: list[str] = [
    "customer_satisfaction_score", "resolution_time_hours", "first_response_time_hours",
]

DATETIME_FIELDS: list[str] = [
    "ticket_created_date", "ticket_resolved_date",
]

BOOL_FIELDS: list[str] = [
    "escalated", "sla_breached",
]

FILTERABLE_FIELDS: list[str] = (
    KEYWORD_FIELDS + INTEGER_FIELDS + FLOAT_FIELDS + DATETIME_FIELDS + BOOL_FIELDS
)
