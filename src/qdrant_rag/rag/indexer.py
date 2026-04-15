"""CSV ingestion pipeline: load → embed (dense + sparse) → upsert to Qdrant.

The pipeline processes the CSV in batches to keep memory usage manageable for
the full 200 k-row dataset.  Both dense and sparse embeddings are computed for
the concatenated issue_description + resolution_notes text.  All remaining
columns are stored as typed payload for downstream filtering.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from tqdm import tqdm

from qdrant_rag.rag.config import (
    COLLECTION_NAME,
    DENSE_VECTOR_NAME,
    INDEX_BATCH_SIZE,
    SPARSE_VECTOR_NAME,
    TEXT_FIELDS,
)
from qdrant_rag.rag.embeddings.dense import DenseEmbedder
from qdrant_rag.rag.embeddings.sparse import SparseEmbedder


# ---------------------------------------------------------------------------
# Payload construction
# ---------------------------------------------------------------------------

def _parse_bool(value: Any) -> bool | None:
    """Convert 'Yes'/'No' strings (or actual booleans) to Python bool."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ("yes", "true", "1"):
            return True
        if v in ("no", "false", "0"):
            return False
    return None


def _safe_int(value: Any) -> int | None:
    try:
        v = float(value)
        return int(v) if not math.isnan(v) else None
    except (TypeError, ValueError):
        return None


def _safe_float(value: Any) -> float | None:
    try:
        v = float(value)
        return None if math.isnan(v) else v
    except (TypeError, ValueError):
        return None


def _safe_str(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    s = str(value).strip()
    return s if s else None


def _safe_date(value: Any) -> str | None:
    """Return ISO-8601 date string (YYYY-MM-DD) or None."""
    s = _safe_str(value)
    if s is None:
        return None
    # pandas may parse to Timestamp already; convert to isoformat
    try:
        return pd.Timestamp(s).strftime("%Y-%m-%dT00:00:00Z")
    except Exception:
        return None


def _build_payload(row: pd.Series) -> dict[str, Any]:
    """Map a CSV row to a Qdrant payload dict with correctly typed values."""
    return {
        # identifiers
        "ticket_id": _safe_int(row.get("ticket_id")),
        # keyword fields
        "customer_name": _safe_str(row.get("customer_name")),
        "customer_email": _safe_str(row.get("customer_email")),
        "product": _safe_str(row.get("product")),
        "category": _safe_str(row.get("category")),
        "priority": _safe_str(row.get("priority")),
        "status": _safe_str(row.get("status")),
        "channel": _safe_str(row.get("channel")),
        "region": _safe_str(row.get("region")),
        "customer_gender": _safe_str(row.get("customer_gender")),
        "subscription_type": _safe_str(row.get("subscription_type")),
        "operating_system": _safe_str(row.get("operating_system")),
        "browser": _safe_str(row.get("browser")),
        "payment_method": _safe_str(row.get("payment_method")),
        "language": _safe_str(row.get("language")),
        "preferred_contact_time": _safe_str(row.get("preferred_contact_time")),
        "customer_segment": _safe_str(row.get("customer_segment")),
        # text (kept in payload for snippet display)
        "issue_description": _safe_str(row.get("issue_description")),
        "resolution_notes": _safe_str(row.get("resolution_notes")),
        # integer fields
        "customer_age": _safe_int(row.get("customer_age")),
        "issue_complexity_score": _safe_int(row.get("issue_complexity_score")),
        "previous_tickets": _safe_int(row.get("previous_tickets")),
        "customer_tenure_months": _safe_int(row.get("customer_tenure_months")),
        # float fields
        "customer_satisfaction_score": _safe_float(row.get("customer_satisfaction_score")),
        "resolution_time_hours": _safe_float(row.get("resolution_time_hours")),
        "first_response_time_hours": _safe_float(row.get("first_response_time_hours")),
        # datetime fields (ISO-8601)
        "ticket_created_date": _safe_date(row.get("ticket_created_date")),
        "ticket_resolved_date": _safe_date(row.get("ticket_resolved_date")),
        # boolean fields
        "escalated": _parse_bool(row.get("escalated")),
        "sla_breached": _parse_bool(row.get("sla_breached")),
    }


def _build_text(row: pd.Series) -> str:
    """Concatenate TEXT_FIELDS into a single string for embedding."""
    parts = [str(row.get(f, "") or "") for f in TEXT_FIELDS]
    return " ".join(p for p in parts if p).strip()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def ingest_csv(
    csv_path: str | Path,
    client: QdrantClient,
    batch_size: int = INDEX_BATCH_SIZE,
    limit: int | None = None,
) -> None:
    """Read *csv_path* and upsert all rows into the Qdrant collection.

    Args:
        csv_path:   Path to the CSV file.
        client:     Initialised QdrantClient.
        batch_size: Number of rows processed per upsert batch.
        limit:      If set, only ingest the first *limit* rows (useful for
                    quick smoke-tests without indexing the full dataset).
    """
    df = pd.read_csv(csv_path)
    if limit is not None:
        df = df.head(limit)

    total = len(df)
    n_batches = math.ceil(total / batch_size)
    print(f"Ingesting {total} rows in {n_batches} batches of {batch_size} …")

    dense_embedder = DenseEmbedder()
    sparse_embedder = SparseEmbedder()

    outer = tqdm(range(0, total, batch_size), total=n_batches, desc="Indexing batches", unit="batch")

    for start in outer:
        batch = df.iloc[start : start + batch_size]
        texts = [_build_text(row) for _, row in batch.iterrows()]

        # Compute embeddings (progress bars suppressed inside batches)
        dense_vecs = dense_embedder.embed_documents(texts, batch_size=batch_size, show_progress=False)
        sparse_vecs = sparse_embedder.embed_documents(texts, batch_size=batch_size, show_progress=False)

        points: list[qm.PointStruct] = []
        for j, (_, row) in enumerate(batch.iterrows()):
            sv = sparse_vecs[j]
            points.append(
                qm.PointStruct(
                    id=int(row["ticket_id"]),
                    vector={
                        DENSE_VECTOR_NAME: dense_vecs[j].tolist(),
                        SPARSE_VECTOR_NAME: qm.SparseVector(
                            indices=sv.indices.tolist(),
                            values=sv.values.tolist(),
                        ),
                    },
                    payload=_build_payload(row),
                )
            )

        client.upsert(collection_name=COLLECTION_NAME, points=points, wait=True)

    print(f"Done. {total} points upserted to '{COLLECTION_NAME}'.")
