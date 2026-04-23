from qdrant_client import QdrantClient, models as qdrant_models
from pylate import models, rank
from rag_time.config.settings import settings
from rag_time.embeddings import EmbeddingsGenerator, SparseEmbeddingsGenerator
from typing import Any, Optional

client = QdrantClient(settings.qdrant_url)
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

reranker = models.ColBERT(model_name_or_path=settings.reranker_model)

_reranker_cache: dict[str, models.ColBERT] = {}


def _get_reranker(model_name: str) -> models.ColBERT:
    if model_name not in _reranker_cache:
        print(f"  [retrieval] loading reranker model: {model_name}")
        _reranker_cache[model_name] = models.ColBERT(model_name_or_path=model_name)
    return _reranker_cache[model_name]


def rerank_with_model(query_text: str, results: dict, reranker_model_name: str) -> dict:
    """Rerank results dict using the specified ColBERT model. Returns a re-ordered dict."""
    model = _get_reranker(reranker_model_name)
    documents = []
    documents_ids = []
    for key, val in results.items():
        documents.append((val.get("subject") or "") + " " + (val.get("body") or ""))
        documents_ids.append(key)

    if not documents:
        return results

    embedded_query = model.encode([query_text], is_query=True)
    embedded_documents = model.encode([documents], is_query=False)

    reranked = rank.rerank(
        documents_ids=[documents_ids],
        queries_embeddings=embedded_query,
        documents_embeddings=embedded_documents,
    )[0]

    for doc in reranked:
        results[doc["id"]]["score"] = doc["score"]

    return dict(sorted(results.items(), key=lambda item: item[1]["score"], reverse=True))

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

def create_filter(filter: dict):
    filters = []
    for key, value in filter.items():
        filters.append(
            qdrant_models.FieldCondition(
                key=key,
                match=qdrant_models.MatchValue(value=value)
            )
        )
    return qdrant_models.Filter(must=filters)



def rag_search(query_text: str, collection_name: str, filter: dict = None, method: str = "hybrid", limit: int = 3) -> dict:
    if filter:
        filter = create_filter(filter)
    limit = limit * 4
    if method == "semantic":
        print("Performing semantic search...")  
        query = embedding_model.encode_query(query_text)
        results = client.query_points(
            collection_name=collection_name,
            query=query,
            using="vector",
            query_filter=filter,
            with_payload=qdrant_models.PayloadSelectorInclude(include=["ref_id", "chunk"]),
            limit=limit
        ).points
        return clean_results(results)
    
    if method == "lexical":
        print("Performing lexical search...")
        query = sparse_embedding_model.encode_query(query_text)
        results = client.query_points(
            collection_name=collection_name,
            query=query,
            using="sparse-vector",
            query_filter=filter,
            with_payload=qdrant_models.PayloadSelectorInclude(include=["ref_id", "chunk"]),
            limit=limit
        ).points
        return clean_results(results)
    
    if method == "hybrid":
        print("Performing hybrid search...")
        prefetch = []
        query = embedding_model.encode_query(query_text)
        prefetch.append(
            qdrant_models.Prefetch(
                query=query,
                using="vector",
                filter=filter,
                limit=limit
            )
        )
        query = sparse_embedding_model.encode_query(query_text)
        prefetch.append(
            qdrant_models.Prefetch(
                query=query,
                using="sparse-vector",
                filter=filter,
                limit=limit
            )
        )
        results = client.query_points(
            collection_name=collection_name,
            prefetch=prefetch,
            query=qdrant_models.FusionQuery(fusion=qdrant_models.Fusion.RRF),
            with_payload=qdrant_models.PayloadSelectorInclude(include=["ref_id", "chunk"]),
            limit=limit,
        ).points
        return clean_results(results)

def rerank(query_text: str, results: dict) -> dict:
    embeded_query = reranker.encode(
        [query_text],
        is_query=True,
    )
    documents = []
    documents_ids = []

    for key, val in results.items():
        documents.append(val["subject"]+" "+val["body"])
        documents_ids.append(key)

    if not documents:
        return []

    embeded_documents = reranker.encode(
        [documents],
        is_query=False,
    )

    reranked_documents = rank.rerank(
        documents_ids=[documents_ids],
        queries_embeddings=embeded_query,
        documents_embeddings=embeded_documents,
    )[0]

    for doc in reranked_documents:
        results[doc["id"]]["score"] = doc["score"]
    return sorted(results.items(), key=lambda item: item[1]["score"], reverse=True)


if __name__ == "__main__":
    import argparse
    import json
    parser = argparse.ArgumentParser()
    parser.add_argument("-q", "--query", required=True)
    parser.add_argument("-m", "--method", default="hybrid", choices=["semantic", "lexical", "hybrid"])
    parser.add_argument("-f", "--filter", type=str, default=None)
    args = parser.parse_args()
    filter = json.loads(args.filter) if args.filter else None
    results = rag_search(args.query, settings.collection_name, method=args.method, filter=filter, limit=settings.top_k)
    for key, val in results.items():
        print(f"ID: {key}\nScore: {val['score']}\nSujet: {val['subject']}\nQuestion: {val['body']}\nRéponse: {val['answer']}\n{'-'*50}")