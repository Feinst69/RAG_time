"""Hybrid retriever based on BM25 and dense vectors stored in JSONL."""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)


@dataclass(slots=True)
class IndexedChunk:
    """Single retrievable chunk extracted from a ticket JSONL line."""

    text: str
    embedding: np.ndarray
    subject: str
    answer: str
    ticket_type: str
    queue: str
    priority: str
    language: str
    chunk_index: int
    source_line: int


@dataclass(slots=True)
class SearchResult:
    """Returned search item with decomposed scores."""

    chunk: IndexedChunk
    score: float
    bm25_score: float
    vector_score: float


@dataclass(slots=True)
class SearchFilters:
    """Metadata filters applied before retrieval scoring."""

    ticket_type: str | None = None
    queue: str | None = None
    priority: str | None = None
    language: str | None = None


class QueryEmbedder:
    """Embed queries with the same family of dense retrieval models."""

    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-small",
        query_prefix: str = "query: ",
        device: str = "cpu",
        normalize_embeddings: bool = True,
    ):
        self.query_prefix = query_prefix
        self.device = device
        self.normalize_embeddings = normalize_embeddings
        self.model = self._build_model(
            model_name=model_name,
        )

    def _build_model(self, model_name: str) -> Any:
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError as exc:
            raise ImportError(
                "Le mode vectoriel requiert 'langchain_huggingface'. "
                "Installe les dependances du projet avant d'utiliser la recherche dense."
            ) from exc

        return HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": self.device},
            encode_kwargs={"normalize_embeddings": self.normalize_embeddings},
        )

    def embed_query(self, query: str) -> np.ndarray:
        vector = np.asarray(
            self.model.embed_query(f"{self.query_prefix}{query}"),
            dtype=np.float32,
        )
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        return vector


class HybridRetriever:
    """Hybrid retriever over chunked tickets stored in JSONL."""

    def __init__(
        self,
        jsonl_path: str,
        embedding_model: str = "intfloat/multilingual-e5-small",
        device: str = "cpu",
        vector_weight: float = 0.5,
    ):
        self.jsonl_path = Path(jsonl_path)
        if not self.jsonl_path.exists():
            raise FileNotFoundError(f"JSONL introuvable: {self.jsonl_path}")

        if not 0.0 <= vector_weight <= 1.0:
            raise ValueError("vector_weight doit être compris entre 0.0 et 1.0")

        self.vector_weight = vector_weight
        self.chunks = self._load_chunks()
        if not self.chunks:
            raise ValueError("Aucun chunk indexable trouvé dans le JSONL")

        self.tokenized_docs = [self._tokenize(chunk.text) for chunk in self.chunks]
        self.doc_lengths = np.asarray(
            [len(tokens) for tokens in self.tokenized_docs], dtype=np.float32
        )
        self.avg_doc_length = float(self.doc_lengths.mean()) if len(self.doc_lengths) else 0.0
        self._build_bm25_index()
        self.embedding_matrix = np.vstack([chunk.embedding for chunk in self.chunks]).astype(
            np.float32
        )
        self._normalize_document_embeddings()
        self.query_embedder = None
        if self.vector_weight > 0.0:
            self.query_embedder = QueryEmbedder(model_name=embedding_model, device=device)

    def _load_chunks(self) -> list[IndexedChunk]:
        chunks: list[IndexedChunk] = []
        with self.jsonl_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                payload = json.loads(line)
                raw_chunks = payload.get("chunks") or []
                raw_embeddings = payload.get("embeddings") or []
                if len(raw_chunks) != len(raw_embeddings):
                    raise ValueError(
                        "Le nombre de chunks et d'embeddings differe "
                        f"ligne {line_number}: {len(raw_chunks)} != {len(raw_embeddings)}"
                    )

                for chunk_index, (text, embedding) in enumerate(
                    zip(raw_chunks, raw_embeddings, strict=True)
                ):
                    if not text:
                        continue
                    chunks.append(
                        IndexedChunk(
                            text=text,
                            embedding=np.asarray(embedding, dtype=np.float32),
                            subject=payload.get("subject", ""),
                            answer=payload.get("answer", ""),
                            ticket_type=payload.get("type", ""),
                            queue=payload.get("queue", ""),
                            priority=payload.get("priority", ""),
                            language=payload.get("language", ""),
                            chunk_index=chunk_index,
                            source_line=line_number,
                        )
                    )
        return chunks

    def _build_bm25_index(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.term_frequencies: list[Counter[str]] = []
        self.inverted_index: dict[str, list[tuple[int, int]]] = defaultdict(list)
        doc_freq: Counter[str] = Counter()

        for doc_id, tokens in enumerate(self.tokenized_docs):
            frequencies = Counter(tokens)
            self.term_frequencies.append(frequencies)
            for token, count in frequencies.items():
                doc_freq[token] += 1
                self.inverted_index[token].append((doc_id, count))

        total_docs = len(self.tokenized_docs)
        self.idf = {
            token: math.log(1.0 + (total_docs - freq + 0.5) / (freq + 0.5))
            for token, freq in doc_freq.items()
        }

    def _normalize_document_embeddings(self) -> None:
        norms = np.linalg.norm(self.embedding_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.embedding_matrix = self.embedding_matrix / norms

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return TOKEN_PATTERN.findall(text.lower())

    def _score_bm25(self, query: str) -> np.ndarray:
        query_tokens = self._tokenize(query)
        scores = np.zeros(len(self.chunks), dtype=np.float32)

        for token in query_tokens:
            postings = self.inverted_index.get(token)
            if not postings:
                continue

            idf = self.idf[token]
            for doc_id, tf in postings:
                doc_len = self.doc_lengths[doc_id]
                denominator = tf + self.k1 * (
                    1.0 - self.b + self.b * (doc_len / max(self.avg_doc_length, 1e-12))
                )
                scores[doc_id] += idf * (tf * (self.k1 + 1.0)) / denominator

        return scores

    def _score_vectors(self, query: str) -> np.ndarray:
        if self.query_embedder is None:
            return np.zeros(len(self.chunks), dtype=np.float32)
        query_vector = self.query_embedder.embed_query(query)
        return self.embedding_matrix @ query_vector

    @staticmethod
    def _normalize_filter_value(value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip().casefold()
        return normalized or None

    def _get_filtered_indices(self, filters: SearchFilters | None) -> np.ndarray:
        if filters is None:
            return np.arange(len(self.chunks))

        normalized_filters = {
            "ticket_type": self._normalize_filter_value(filters.ticket_type),
            "queue": self._normalize_filter_value(filters.queue),
            "priority": self._normalize_filter_value(filters.priority),
            "language": self._normalize_filter_value(filters.language),
        }
        active_filters = {
            field_name: value
            for field_name, value in normalized_filters.items()
            if value is not None
        }
        if not active_filters:
            return np.arange(len(self.chunks))

        filtered_indices: list[int] = []
        for idx, chunk in enumerate(self.chunks):
            if all(
                self._normalize_filter_value(getattr(chunk, field_name)) == expected_value
                for field_name, expected_value in active_filters.items()
            ):
                filtered_indices.append(idx)

        return np.asarray(filtered_indices, dtype=np.int32)

    @staticmethod
    def _min_max_normalize(scores: np.ndarray) -> np.ndarray:
        if scores.size == 0:
            return scores
        min_score = float(scores.min())
        max_score = float(scores.max())
        if math.isclose(min_score, max_score):
            return np.ones_like(scores, dtype=np.float32)
        return (scores - min_score) / (max_score - min_score)

    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: SearchFilters | None = None,
    ) -> list[SearchResult]:
        if not query.strip():
            raise ValueError("La requete ne peut pas etre vide")

        filtered_indices = self._get_filtered_indices(filters)
        if filtered_indices.size == 0:
            return []

        bm25_scores = self._score_bm25(query)
        vector_scores = self._score_vectors(query)

        filtered_bm25_scores = bm25_scores[filtered_indices]
        filtered_vector_scores = vector_scores[filtered_indices]

        bm25_norm = self._min_max_normalize(filtered_bm25_scores)
        vector_norm = self._min_max_normalize(filtered_vector_scores)
        hybrid_scores = (
            self.vector_weight * vector_norm + (1.0 - self.vector_weight) * bm25_norm
        )

        limit = min(top_k, len(filtered_indices))
        ranked_positions = np.argsort(hybrid_scores)[::-1][:limit]
        top_indices = filtered_indices[ranked_positions]

        return [
            SearchResult(
                chunk=self.chunks[idx],
                score=float(hybrid_scores[position]),
                bm25_score=float(bm25_scores[idx]),
                vector_score=float(vector_scores[idx]),
            )
            for position, idx in zip(ranked_positions, top_indices, strict=True)
        ]
