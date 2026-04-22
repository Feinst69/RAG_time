from qdrant_client import QdrantClient, models as qdrant_models
from pylate import models, rank
from rag_time.config.settings import settings
from rag_time.embeddings import EmbeddingsGenerator, SparseEmbeddingsGenerator
from typing import Any, Optional

client = QdrantClient(settings.qdrant_url)
embedding_model = EmbeddingsGenerator(settings.embedding_model, max_length=settings.embedding_dimension)
sparse_embedding_model = SparseEmbeddingsGenerator(settings.sparse_model)
reranker = models.ColBERT(model_name_or_path=settings.reranker_model)

def get_tickets_by_ids(collection_name: str, ids: list):
    return client.retrieve(
        collection_name=collection_name,
        ids=ids,
        with_payload=qdrant_models.PayloadSelectorExclude(exclude=["ref_id", "chunk"]), 
        with_vectors=False
    )

def clean_results(results: list[Any]):
    cleaned_results = {}
    for res in results:
        ref_id = res.payload.get("ref_id")
        if ref_id not in cleaned_results:
            cleaned_results[ref_id] = {"score": res.score}
        if len(cleaned_results) >= settings.top_k:
            break
    
    documents = get_tickets_by_ids(settings.collection_name, list(cleaned_results.keys())) 
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
    results =  client.query_points(
        collection_name=collection_name,
        prefetch=prefetch,
        query=qdrant_models.FusionQuery(fusion=qdrant_models.Fusion.RRF),
        with_payload=qdrant_models.PayloadSelectorInclude(include=["ref_id", "chunk"]),
        limit=limit).points
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


