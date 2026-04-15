"""Hybrid retriever: dense + sparse (RRF fusion) → ColBERT reranking.

Pipeline
--------
1.  Encode query with DenseEmbedder (17m ColBERT, mean-pooled).
2.  Encode query with SparseEmbedder (BM25).
3.  Qdrant hybrid query:
      - Prefetch(dense, limit=candidates)
      - Prefetch(sparse, limit=candidates)
      - Fuse with Reciprocal Rank Fusion (RRF)
      - Apply optional Filter on metadata
4.  Collect up to *candidates* results.
5.  Rerank with ColBERTReranker (32m, MaxSim late interaction).
6.  Return top-*top_k* SearchResult objects.
"""

from __future__ import annotations

from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from qdrant_rag.rag.config import (
    COLLECTION_NAME,
    DEFAULT_TOP_K,
    DENSE_VECTOR_NAME,
    QDRANT_URL,
    RETRIEVAL_CANDIDATES,
    SPARSE_VECTOR_NAME,
    TEXT_FIELDS,
)
from qdrant_rag.rag.embeddings.colbert import ColBERTReranker
from qdrant_rag.rag.embeddings.dense import DenseEmbedder
from qdrant_rag.rag.embeddings.sparse import SparseEmbedder
from qdrant_rag.rag.filters import build_filter
from qdrant_rag.rag.models import FilterParams, SearchResult


class HybridRetriever:
    """End-to-end hybrid retriever with optional metadata filtering and ColBERT reranking."""

    def __init__(
        self,
        qdrant_url: str = QDRANT_URL,
        client: Optional[QdrantClient] = None,
        *,
        dense_embedder: Optional[DenseEmbedder] = None,
        sparse_embedder: Optional[SparseEmbedder] = None,
        reranker: Optional[ColBERTReranker] = None,
    ) -> None:
        self.client = client or QdrantClient(url=qdrant_url)
        self.dense_embedder = dense_embedder or DenseEmbedder()
        self.sparse_embedder = sparse_embedder or SparseEmbedder()
        self.reranker = reranker or ColBERTReranker()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        filters: Optional[FilterParams] = None,
        candidates: int = RETRIEVAL_CANDIDATES,
        top_k: int = DEFAULT_TOP_K,
    ) -> list[SearchResult]:
        """Run the full retrieval pipeline and return ranked results.

        Args:
            query:      Natural-language query string.
            filters:    Optional structured metadata filter.
            candidates: Number of candidates to fetch from Qdrant before
                        ColBERT reranking (higher = better recall, slower rerank).
            top_k:      Final number of results to return.

        Returns:
            List of SearchResult sorted by ColBERT score (descending).
        """
        # 1. Encode query
        dense_vec = self.dense_embedder.embed_query(query)
        sparse_vec = self.sparse_embedder.embed_query(query)

        # 2. Build optional Qdrant filter
        qdrant_filter = build_filter(filters) if (filters and not filters.is_empty()) else None

        # 3. Hybrid query with RRF fusion
        response = self.client.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=[
                qm.Prefetch(
                    query=dense_vec.tolist(),
                    using=DENSE_VECTOR_NAME,
                    limit=candidates,
                    filter=qdrant_filter,
                ),
                qm.Prefetch(
                    query=qm.SparseVector(
                        indices=sparse_vec.indices.tolist(),
                        values=sparse_vec.values.tolist(),
                    ),
                    using=SPARSE_VECTOR_NAME,
                    limit=candidates,
                    filter=qdrant_filter,
                ),
            ],
            query=qm.FusionQuery(fusion=qm.Fusion.RRF),
            limit=candidates,
            with_payload=True,
        )

        points = response.points
        if not points:
            return []

        # 4. Assemble candidate texts for reranking
        candidate_texts = [
            " ".join(str(p.payload.get(f, "") or "") for f in TEXT_FIELDS).strip()
            for p in points
        ]

        # 5. ColBERT reranking
        ranked_pairs = self.reranker.rerank(query, candidate_texts, top_k=top_k)

        # 6. Build SearchResult list
        results: list[SearchResult] = []
        for original_idx, score in ranked_pairs:
            pt = points[original_idx]
            results.append(
                SearchResult(
                    ticket_id=int(pt.id),
                    score=score,
                    payload=pt.payload or {},
                )
            )

        return results

    def search_no_rerank(
        self,
        query: str,
        filters: Optional[FilterParams] = None,
        top_k: int = DEFAULT_TOP_K,
    ) -> list[SearchResult]:
        """Hybrid search without ColBERT reranking (faster, lower quality).

        Useful for quick prototyping or when latency matters more than precision.
        """
        dense_vec = self.dense_embedder.embed_query(query)
        sparse_vec = self.sparse_embedder.embed_query(query)
        qdrant_filter = build_filter(filters) if (filters and not filters.is_empty()) else None

        response = self.client.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=[
                qm.Prefetch(
                    query=dense_vec.tolist(),
                    using=DENSE_VECTOR_NAME,
                    limit=top_k * 2,
                    filter=qdrant_filter,
                ),
                qm.Prefetch(
                    query=qm.SparseVector(
                        indices=sparse_vec.indices.tolist(),
                        values=sparse_vec.values.tolist(),
                    ),
                    using=SPARSE_VECTOR_NAME,
                    limit=top_k * 2,
                    filter=qdrant_filter,
                ),
            ],
            query=qm.FusionQuery(fusion=qm.Fusion.RRF),
            limit=top_k,
            with_payload=True,
        )

        return [
            SearchResult(
                ticket_id=int(pt.id),
                score=pt.score,
                payload=pt.payload or {},
            )
            for pt in response.points
        ]
