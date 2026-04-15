"""Dense embedder: PyLate ColBERT model → mean-pooled single vector per document.

Model: mixedbread-ai/mxbai-edge-colbert-v0-17m (128-dim token embeddings)

PyLate produces per-token embeddings for ColBERT late interaction.  For the
Qdrant dense index we need one vector per document, so we mean-pool the token
dimension.  At query time the same pooling is applied so dot-product / cosine
similarity is consistent.
"""

from __future__ import annotations

import numpy as np
from tqdm import tqdm

from qdrant_rag.rag.config import DENSE_COLBERT_MODEL, INDEX_BATCH_SIZE


class DenseEmbedder:
    """Wraps a PyLate ColBERT model and exposes mean-pooled dense vectors."""

    def __init__(self, model_name: str = DENSE_COLBERT_MODEL) -> None:
        # Lazy import so the module can be imported without GPU/torch installed
        from pylate.models import ColBERT  # type: ignore

        self.model = ColBERT(model_name_or_path=model_name)
        self.model_name = model_name

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _pool(token_embeddings: list[np.ndarray]) -> list[np.ndarray]:
        """Mean-pool a list of (n_tokens, dim) arrays → list of (dim,) arrays."""
        return [np.mean(emb, axis=0).astype(np.float32) for emb in token_embeddings]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed_documents(
        self,
        texts: list[str],
        batch_size: int = INDEX_BATCH_SIZE,
        show_progress: bool = True,
    ) -> list[np.ndarray]:
        """Return one (dim,) float32 array per document text."""
        all_vecs: list[np.ndarray] = []

        batches = range(0, len(texts), batch_size)
        iterator = tqdm(batches, desc="Dense encoding", unit="batch") if show_progress else batches

        for start in iterator:
            batch = texts[start : start + batch_size]
            token_embs = self.model.encode(
                sentences=batch,
                is_query=False,
                show_progress_bar=False,
            )
            all_vecs.extend(self._pool(token_embs))

        return all_vecs

    def embed_query(self, text: str) -> np.ndarray:
        """Return a single (dim,) float32 vector for a query string."""
        token_embs = self.model.encode(
            sentences=[text],
            is_query=True,
            show_progress_bar=False,
        )
        return self._pool(token_embs)[0]
