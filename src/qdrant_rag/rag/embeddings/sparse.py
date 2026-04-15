"""Sparse embedder: BM25 via fastembed (Qdrant/bm25).

fastembed's SparseTextEmbedding wraps the Qdrant-maintained BM25 implementation
and returns SparseEmbedding objects containing `.indices` and `.values` numpy
arrays that map directly to Qdrant's SparseVector format.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from tqdm import tqdm

from qdrant_rag.rag.config import BM25_MODEL, INDEX_BATCH_SIZE


@dataclass
class RawSparseVector:
    """Lightweight container mirroring fastembed's SparseEmbedding."""

    indices: np.ndarray  # dtype int32
    values: np.ndarray   # dtype float32


class SparseEmbedder:
    """Wraps fastembed BM25 and exposes RawSparseVector objects."""

    def __init__(self, model_name: str = BM25_MODEL) -> None:
        from fastembed.sparse import SparseTextEmbedding  # type: ignore

        self.model = SparseTextEmbedding(model_name=model_name)
        self.model_name = model_name

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_raw(sparse_emb) -> RawSparseVector:
        """Convert a fastembed SparseEmbedding to RawSparseVector."""
        return RawSparseVector(
            indices=np.array(sparse_emb.indices, dtype=np.int32),
            values=np.array(sparse_emb.values, dtype=np.float32),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed_documents(
        self,
        texts: list[str],
        batch_size: int = INDEX_BATCH_SIZE,
        show_progress: bool = True,
    ) -> list[RawSparseVector]:
        """Return one RawSparseVector per document."""
        all_vecs: list[RawSparseVector] = []

        batches = range(0, len(texts), batch_size)
        iterator = tqdm(batches, desc="Sparse (BM25) encoding", unit="batch") if show_progress else batches

        for start in iterator:
            batch = texts[start : start + batch_size]
            sparse_embs = list(self.model.embed(batch))
            all_vecs.extend(self._to_raw(e) for e in sparse_embs)

        return all_vecs

    def embed_query(self, text: str) -> RawSparseVector:
        """Return a single RawSparseVector for a query string.

        Uses ``query_embed`` which applies BM25's IDF-weighted query expansion
        instead of the plain document embedding path.
        """
        emb = next(self.model.query_embed(text))
        return self._to_raw(emb)
