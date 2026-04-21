from __future__ import annotations

import asyncio

from sklearn.metrics import ndcg_score

from rag_time.dspy.agents.relevance_judge import RelevanceJudge
from rag_time.dspy.objects.ticket_resolution import RelevanceJudgement


def precision_at_k(retrieved_docs, relevant_docs, k):
    if k <= 0:
        return 0.0
    retrieved_subset = set(retrieved_docs[:k])
    relevant_and_retrieved = retrieved_subset & relevant_docs
    return len(relevant_and_retrieved) / k


def recall_at_k(retrieved_docs, relevant_docs, k):
    if k <= 0:
        return 0.0
    retrieved_subset = set(retrieved_docs[:k])
    relevant_and_retrieved = retrieved_subset & relevant_docs
    return len(relevant_and_retrieved) / len(relevant_docs) if relevant_docs else 0.0


def ndcg_at_k(retrieved_docs, relevant_docs, k):
    relevance_scores = [relevant_docs.get(doc_id, 0) for doc_id in retrieved_docs[:k]]
    ideal_relevance_scores = sorted(relevant_docs.values(), reverse=True)[:k]
    return ndcg_score([ideal_relevance_scores], [relevance_scores])


async def judge_relevance_async(query: str, retrieved_docs: list) -> float:
    """DSPy agent that judges the relevance of retrieved documents for a query.

    Returns a relevance score between 0.0 and 1.0.
    """
    judge = RelevanceJudge()
    judgement: RelevanceJudgement = await judge.forward(query=query, documents=retrieved_docs)
    return judgement.relevance_score


def judge_relevance(query: str, retrieved_docs: list) -> float:
    """Synchronous wrapper around the async DSPy relevance judge."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, judge_relevance_async(query, retrieved_docs))
                return future.result()
        return loop.run_until_complete(judge_relevance_async(query, retrieved_docs))
    except Exception:
        return 0.0
