"""ColBERT late-interaction reranker: PyLate + mxbai-edge-colbert-v0-32m.

Late interaction (MaxSim):
    score(q, d) = Σ_{i ∈ query_tokens} max_{j ∈ doc_tokens} ( q_i · d_j )

This gives a per-document score that captures fine-grained token-level
alignment between the query and each candidate document.
"""

from __future__ import annotations

import numpy as np

from qdrant_rag.rag.config import DEFAULT_TOP_K, RERANK_COLBERT_MODEL


class ColBERTReranker:
    """Late-interaction reranker using a PyLate ColBERT model."""

    def __init__(self, model_name: str = RERANK_COLBERT_MODEL) -> None:
        from pylate.models import ColBERT  # type: ignore

        self.model = ColBERT(model_name_or_path=model_name)
        self.model_name = model_name

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _maxsim(query_emb: np.ndarray, doc_emb: np.ndarray) -> float:
        """Compute ColBERT MaxSim score between one query and one document.

        Args:
            query_emb: (Q, dim) float32 – query token embeddings
            doc_emb:   (D, dim) float32 – document token embeddings

        Returns:
            Scalar ColBERT score.
        """
        # (Q, D) similarity matrix
        sim_matrix = query_emb @ doc_emb.T
        # Sum of per-query-token maximum similarities
        return float(np.sum(np.max(sim_matrix, axis=1)))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: int = DEFAULT_TOP_K,
    ) -> list[tuple[int, float]]:
        """Rerank *documents* with respect to *query* using ColBERT MaxSim.

        Args:
            query:     The user query string.
            documents: Candidate document texts (in their original order).
            top_k:     Number of top results to return.

        Returns:
            List of (original_index, score) sorted by descending score,
            truncated to *top_k*.
        """
        if not documents:
            return []

        # Encode query (with query-augmentation tokens)
        query_embs: list[np.ndarray] = self.model.encode(
            sentences=[query],
            is_query=True,
            show_progress_bar=False,
        )
        query_emb = query_embs[0]  # (Q, dim)

        # Encode all candidate documents
        doc_embs: list[np.ndarray] = self.model.encode(
            sentences=documents,
            is_query=False,
            show_progress_bar=False,
        )

        scores = [self._maxsim(query_emb, d) for d in doc_embs]

        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    def score_pairs(
        self,
        query: str,
        documents: list[str],
    ) -> list[float]:
        """Return a raw score for every (query, document) pair (no sorting)."""
        query_emb = self.model.encode([query], is_query=True, show_progress_bar=False)[0]
        doc_embs = self.model.encode(documents, is_query=False, show_progress_bar=False)
        return [self._maxsim(query_emb, d) for d in doc_embs]
