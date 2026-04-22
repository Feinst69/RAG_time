from qdrant_client import QdrantClient, models as qdrant_models 
from rag_time.config.settings import settings
from rag_time.embeddings import EmbeddingsGenerator, SparseEmbeddingsGenerator
from typing import Any

client = QdrantClient(host="localhost", port=6333)
embedding_model = EmbeddingsGenerator(settings.embedding_model, max_length=settings.embedding_dimension)
sparse_embedding_model = SparseEmbeddingsGenerator(settings.sparse_model)

_dense_cache: dict[str, EmbeddingsGenerator] = {}
_sparse_cache: dict[str, SparseEmbeddingsGenerator] = {}


def _get_dense(model_name: str) -> EmbeddingsGenerator:
    if model_name not in _dense_cache:
        print(f"  [retrieval] loading dense model: {model_name}")
        _dense_cache[model_name] = EmbeddingsGenerator(model_name)
    return _dense_cache[model_name]


def _get_sparse(model_name: str) -> SparseEmbeddingsGenerator:
    if model_name not in _sparse_cache:
        print(f"  [retrieval] loading sparse model: {model_name}")
        _sparse_cache[model_name] = SparseEmbeddingsGenerator(model_name)
    return _sparse_cache[model_name]


def hybrid_search_with_models(
    query_text: str,
    collection_name: str,
    dense_model_name: str,
    sparse_model_name: str,
    limit: int = 30,
):
    """Hybrid search using explicitly specified dense and sparse models."""
    dense = _get_dense(dense_model_name)
    sparse = _get_sparse(sparse_model_name)
    dense_query = dense.encode_query(query_text)
    sparse_query = sparse.encode_query(query_text)
    result = client.query_points(
        collection_name=collection_name,
        prefetch=[
            qdrant_models.Prefetch(query=dense_query, using="vector", limit=limit),
            qdrant_models.Prefetch(query=sparse_query, using="sparse-vector", limit=limit),
        ],
        query=qdrant_models.FusionQuery(fusion=qdrant_models.Fusion.RRF),
        with_payload=qdrant_models.PayloadSelectorInclude(include=["ref_id", "chunk"]),
        limit=limit,
    )
    return result.points


def semantic_search_tickets(query_text: str, collection_name: str, limit: int = 3):
    query_vector = embedding_model.encode_query(query_text)
    search_result = client.query_points(
        query=query_vector,
        collection_name=collection_name,
        using="vector",         
        with_payload=qdrant_models.PayloadSelectorInclude(include=["ref_id", "chunk"]),
        limit=limit
    )
    return search_result.points

def lexical_search_tickets(query_text: str, collection_name: str, limit: int = 3):
    query_vector = sparse_embedding_model.encode_query(query_text)
    search_result = client.query_points(
        query=query_vector,
        collection_name=collection_name,
        using="sparse-vector",         
        with_payload=qdrant_models.PayloadSelectorInclude(include=["ref_id", "chunk"]),
        limit=limit
    )
    return search_result.points


def hybrid_search_tickets(query_text: str, collection_name: str, limit: int = 3):
    dense_query = embedding_model.encode_query(query_text)
    sparse_query = sparse_embedding_model.encode_query(query_text)

    search_result = client.query_points(
        collection_name=collection_name,
        prefetch=[
            qdrant_models.Prefetch(
                query=dense_query,
                using="vector",
                limit=limit
            ),
            qdrant_models.Prefetch(
                query=sparse_query,
                using="sparse-vector",
                limit=limit
            )
        ],
        query=qdrant_models.FusionQuery(fusion=qdrant_models.Fusion.RRF),
        with_payload=qdrant_models.PayloadSelectorInclude(include=["ref_id", "chunk"]),
        limit=limit
    )
    return search_result.points


def get_tickets_by_ids(collection_name: str, ids: list):
    return client.retrieve(
        collection_name=collection_name,
        ids=ids,
        with_payload=qdrant_models.PayloadSelectorExclude(exclude=["ref_id", "chunk"]), 
        with_vectors=False
    )

def clean_results(results: list[Any], collection_name: str | None = None):
    collection_name = collection_name or settings.collection_name
    cleaned_results = {}
    for res in results:
        ref_id = res.payload.get("ref_id")
        if ref_id not in cleaned_results:
            cleaned_results[ref_id] = {"score": res.score}
        if len(cleaned_results) >= settings.top_k:
            break

    documents = get_tickets_by_ids(collection_name, list(cleaned_results.keys()))
    for doc in documents:
        cleaned_results[doc.id] |= doc.payload

    return cleaned_results

def do_semantic_search(query: str):
    results = semantic_search_tickets(query, settings.collection_name, limit=settings.top_k*4)
    return clean_results(results)


def do_lexical_search(query: str):
    results =  lexical_search_tickets(query, settings.collection_name, limit=settings.top_k*4)
    return clean_results(results)


def do_hybrid_search(query: str):
    results =  hybrid_search_tickets(query, settings.collection_name, limit=settings.top_k*4)
    return clean_results(results)


if __name__ == "__main__":
    query = "Comment réinitialiser mon mot de passe ?"
    print("Keyword Search Results:")
    results = sparse_embedding_model.encode_query(query)
    print(results)
    results = lexical_search_tickets(query, settings.collection_name, limit=settings.top_k*4)
    print(results)
    