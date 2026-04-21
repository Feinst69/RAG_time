from rag_time.llm.openrouter import openrouter_llm_api_call

def precision_at_k(retrieved_docs, relevant_docs, k):
    """
    Calculate Precision@K for a single query.

    Args:
        retrieved_docs (list): List of retrieved document IDs.
        relevant_docs (set): Set of relevant document IDs.
        k (int): The cutoff rank.
    Returns:
        float: The precision at K.
    """
    if k <= 0:
        return 0.0

    retrieved_subset = set(retrieved_docs[:k])
    relevant_and_retrieved = retrieved_subset & relevant_docs

    return len(relevant_and_retrieved) / k if k > 0 else 0.0

def recall_at_k(retrieved_docs, relevant_docs, k):
    """
    Calculate Recall@K for a single query.

    Args:
        retrieved_docs (list): List of retrieved document IDs.
        relevant_docs (set): Set of relevant document IDs.
        k (int): The cutoff rank.
    Returns:
        float: The recall at K.
    """    
    if k <= 0:
        return 0.0
    retrieved_subset = set(retrieved_docs[:k])
    relevant_and_retrieved = retrieved_subset & relevant_docs
    return len(relevant_and_retrieved) / len(relevant_docs) if relevant_docs else 0.0

def judge_relevance(query, retrieved_docs):
    """
    A LLM will be called to judge the relevance of retrieved documents for a given query 
    using the RAGAS library. and return a % relevance score for each document and a global
    percentage of relevance for the whole set of retrieved documents."""

    def llm_eval(query: str, retrieved_docs: list) -> float:
        
        pass

    # Call the LLM to judge relevance
    score = llm_eval(query, retrieved_docs)

    return score