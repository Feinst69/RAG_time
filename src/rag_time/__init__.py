"""RAG Time - Système de recherche assistée pour tickets de support."""

__version__ = "0.1.0"

from rag_time.retriever import HybridRetriever, SearchResult

__all__ = ["HybridRetriever", "SearchResult", "__version__"]
