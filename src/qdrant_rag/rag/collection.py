"""Qdrant collection management: creation, deletion, and payload index setup."""

from __future__ import annotations

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from qdrant_rag.rag.config import (
    BOOL_FIELDS,
    COLLECTION_NAME,
    DATETIME_FIELDS,
    DENSE_VECTOR_DIM,
    DENSE_VECTOR_NAME,
    FLOAT_FIELDS,
    INTEGER_FIELDS,
    KEYWORD_FIELDS,
    SPARSE_VECTOR_NAME,
)


def create_collection(client: QdrantClient, recreate: bool = False) -> None:
    """Create (or recreate) the Qdrant collection with:

    - A named dense vector  (cosine, 128-dim)
    - A named sparse vector (BM25)
    - Payload indexes for every filterable field
    """
    existing = {c.name for c in client.get_collections().collections}

    if COLLECTION_NAME in existing:
        if recreate:
            print(f"Deleting existing collection '{COLLECTION_NAME}' …")
            client.delete_collection(COLLECTION_NAME)
        else:
            print(f"Collection '{COLLECTION_NAME}' already exists. Use recreate=True to reset.")
            return

    print(f"Creating collection '{COLLECTION_NAME}' …")
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            DENSE_VECTOR_NAME: qm.VectorParams(
                size=DENSE_VECTOR_DIM,
                distance=qm.Distance.COSINE,
                on_disk=False,
            ),
        },
        sparse_vectors_config={
            SPARSE_VECTOR_NAME: qm.SparseVectorParams(
                index=qm.SparseIndexParams(on_disk=False),
            ),
        },
        # HNSW optimised for hybrid workloads
        hnsw_config=qm.HnswConfigDiff(m=16, ef_construct=100),
        optimizers_config=qm.OptimizersConfigDiff(memmap_threshold=20_000),
    )

    _create_payload_indexes(client)
    print(f"Collection '{COLLECTION_NAME}' ready.")


def _create_payload_indexes(client: QdrantClient) -> None:
    """Create typed payload indexes to enable efficient metadata filtering."""

    # Keyword fields  (exact-match MatchValue / MatchAny)
    for field in KEYWORD_FIELDS:
        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name=field,
            field_schema=qm.PayloadSchemaType.KEYWORD,
        )

    # Integer range fields
    for field in INTEGER_FIELDS:
        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name=field,
            field_schema=qm.PayloadSchemaType.INTEGER,
        )

    # Float range fields
    for field in FLOAT_FIELDS:
        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name=field,
            field_schema=qm.PayloadSchemaType.FLOAT,
        )

    # Datetime range fields (stored as ISO-8601 strings)
    for field in DATETIME_FIELDS:
        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name=field,
            field_schema=qm.PayloadSchemaType.DATETIME,
        )

    # Boolean fields (stored as Python bool, filtered with MatchValue)
    for field in BOOL_FIELDS:
        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name=field,
            field_schema=qm.PayloadSchemaType.BOOL,
        )

    print(f"  Payload indexes created for {len(KEYWORD_FIELDS)} keyword, "
          f"{len(INTEGER_FIELDS)} integer, {len(FLOAT_FIELDS)} float, "
          f"{len(DATETIME_FIELDS)} datetime, {len(BOOL_FIELDS)} bool fields.")


def collection_info(client: QdrantClient) -> dict:
    """Return basic info about the collection."""
    info = client.get_collection(COLLECTION_NAME)
    return {
        "name": COLLECTION_NAME,
        "vectors_count": info.vectors_count,
        "points_count": info.points_count,
        "status": info.status,
    }
