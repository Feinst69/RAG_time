"""RAG evaluation pipeline.

For every combination of (dense, sparse, reranker, query_rephrasing, llm_model):
  - Loops over all q&a pairs from eval_config.yaml
  - Computes per-question retrieval metrics + LLM metrics
  - Averages metrics across all questions → combination score
  - Ranks all combinations by avg LLM answer quality (or retrieval score if LLM disabled)
  - Saves full results (per-question + per-combination) to JSON
"""

from __future__ import annotations

import asyncio
import json
import os
from statistics import mean
from typing import Any

import dspy
from dotenv import load_dotenv

from rag_time.dspy.config import DSPyConfig
from rag_time.dspy.agents.query_rephraser import QueryRephraser
from rag_time.dspy.agents.ticket_answer import generate_answer
from rag_time.dspy.agents.relevance_judge import RelevanceJudge, AnswerQualityJudgeSignature
from rag_time.dspy.objects.ticket_resolution import TicketResolution, AnswerQualityJudgement
from rag_time.eval.metrics import precision_at_k, recall_at_k, judge_relevance_async, ndcg_at_k


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def yaml_loader(path: str) -> dict:
    import yaml
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _average_metrics(metrics_list: list[dict]) -> dict[str, float]:
    """Average a list of per-question metric dicts into one dict."""
    if not metrics_list:
        return {}
    keys = metrics_list[0].keys()
    return {
        k: mean(m[k] for m in metrics_list if m.get(k) is not None)
        for k in keys
    }


def _combo_llm_score(combo: dict) -> float:
    """Primary ranking score: avg answer_quality across LLM models, or retrieval fallback."""
    avg_llm = combo.get("avg_llm_metrics", {})
    if avg_llm:
        scores = [m.get("answer_quality", 0.0) for m in avg_llm.values()]
        return mean(scores) if scores else 0.0
    return combo.get("avg_retrieval_metrics", {}).get("relevance_score", 0.0) or 0.0


# ---------------------------------------------------------------------------
# Per-question functions
# ---------------------------------------------------------------------------

async def rephrase_query(query: str) -> str:
    """Rephrase query using the QueryRephraser agent."""
    rephraser = QueryRephraser()
    empty_history = dspy.History(messages=[])
    rephrased = await rephraser(question=query, history=empty_history)
    return rephrased


async def compute_retrieval_metrics(
    retrieved_docs: list,
    relevant_docs: set,
    k: int,
    query: str,
) -> dict[str, float]:
    """Compute all retrieval metrics for a single query."""
    relevance_score = await judge_relevance_async(query, retrieved_docs)
    return {
        "precision_at_k": precision_at_k(retrieved_docs, relevant_docs, k),
        "recall_at_k": recall_at_k(retrieved_docs, relevant_docs, k),
        "relevance_score": relevance_score,
    }


async def evaluate_llm_answers(
    query: str,
    retrieved_docs: list,
    generated_answer: str | TicketResolution,
    reference_answer: str,
) -> dict[str, float]:
    """Evaluate a generated answer via an LLM-as-judge DSPy agent.

    The judge produces an AnswerQualityJudgement Pydantic object covering
    answer_quality, retrieval_relevance, and retrieval_usage scores.
    """
    answer_text = (
        generated_answer.model_dump_json(indent=2)
        if isinstance(generated_answer, TicketResolution)
        else str(generated_answer)
    )
    docs_text = "\n\n".join(str(d) for d in retrieved_docs) if retrieved_docs else "(none)"

    judge_module = dspy.asyncify(dspy.ChainOfThought(AnswerQualityJudgeSignature))
    result = await judge_module(
        query=query,
        retrieved_documents=docs_text,
        generated_answer=answer_text,
        reference_answer=reference_answer,
    )

    # Parse raw JSON string output into AnswerQualityJudgement Pydantic model
    import json as _json
    try:
        raw = result.judgement if isinstance(result.judgement, dict) else _json.loads(result.judgement)
        judgement = AnswerQualityJudgement(**raw)
    except Exception:
        judgement = AnswerQualityJudgement(
            answer_quality=0.0,
            retrieval_relevance=0.0,
            retrieval_usage=0.0,
            reasoning="Parsing failed",
        )

    return {
        "answer_quality": judgement.answer_quality,
        "retrieval_relevance": judgement.retrieval_relevance,
        "retrieval_usage": judgement.retrieval_usage,
    }


def save_results(results: dict, output_path: str = "eval_results.json") -> None:
    """Save full evaluation results (per-question + ranked combinations) to JSON."""
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved to {output_path}")


# ---------------------------------------------------------------------------
# Core evaluation loop
# ---------------------------------------------------------------------------

async def _evaluate_combination(
    dense: str,
    sparse: str,
    reranker: str,
    use_rephrasing: bool,
    qa_pairs: list[dict],
    k: int,
    llm_models: list[str],
    dataset_path: str = "data/aa_sample.csv",
) -> dict[str, Any]:
    """Run all q&a pairs through one model combination and return aggregated results."""

    question_results: list[dict] = []

    for qa in qa_pairs:
        original_query: str = qa["question"]
        reference_answer: str = qa["answer"]

        # ── Query rephrasing ─────────────────────────────────────────────────
        if use_rephrasing:
            query = await rephrase_query(original_query)
        else:
            query = original_query

        # ── Retrieval ────────────────────────────────────────────────────────
        # TODO: replace with actual retriever call using (dense, sparse, reranker)
        # dataset: dataset_path
        retrieved_docs: list = []
        relevant_docs: set = set()

        # ── Retrieval metrics ────────────────────────────────────────────────
        ret_metrics = await compute_retrieval_metrics(retrieved_docs, relevant_docs, k, query)

        q_result: dict[str, Any] = {
            "question": original_query,
            "rephrased_query": query if use_rephrasing else None,
            "retrieval_metrics": ret_metrics,
        }

        # ── LLM generation + metrics ─────────────────────────────────────────
        if llm_models:
            q_result["llm_results"] = {}
            for llm_model in llm_models:
                _configure_lm(model=llm_model)
                answer = await generate_answer(
                    query=query,
                    documents=str(retrieved_docs),
                    streaming=False,
                )
                llm_metrics = await evaluate_llm_answers(
                    query, retrieved_docs, answer, reference_answer
                )
                q_result["llm_results"][llm_model] = {
                    "answer": answer,
                    "metrics": llm_metrics,
                }

        question_results.append(q_result)

    # ── Aggregate across all questions ───────────────────────────────────────
    avg_retrieval = _average_metrics([r["retrieval_metrics"] for r in question_results])

    avg_llm: dict[str, dict] = {}
    if llm_models:
        for llm_model in llm_models:
            avg_llm[llm_model] = _average_metrics(
                [r["llm_results"][llm_model]["metrics"] for r in question_results]
            )

    return {
        "combination": {
            "dense": dense,
            "sparse": sparse,
            "reranker": reranker,
            "query_rephrasing": use_rephrasing,
        },
        "avg_retrieval_metrics": avg_retrieval,
        "avg_llm_metrics": avg_llm,
        "question_results": question_results,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _get_api_key() -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY missing – set it in .env")
    return api_key


def _configure_lm(model: str | None = None) -> None:
    """Configure the DSPy global LM.

    If *model* is provided (from eval_config.yaml), it is used directly —
    prepending 'openrouter/' if the name does not already include a provider prefix.
    Otherwise falls back to the model defined in the DSPy config YAML.
    """
    api_key = _get_api_key()
    if model is None:
        model = DSPyConfig().lm_model
    elif "/" not in model.split("/")[0]:
        # plain model name like "google/gemini-..." → add openrouter provider prefix
        model = f"openrouter/{model}"
    lm = dspy.LM(
        model=model,
        api_base="https://openrouter.ai/api/v1",
        api_key=api_key,
    )
    dspy.configure(lm=lm)


async def full_eval(config_path: str) -> dict[str, Any]:
    load_dotenv(override=True)
    _get_api_key()  # fail fast if key is missing
    config = yaml_loader(config_path)

    qa_pairs: list[dict] = config["retrieval"]["q_&_a_pairs"]
    k: int = config["retrieval"].get("k", 10)
    dataset_path: str = config["retrieval"].get("dataset_path", "data/aa_sample.csv")
    dense_models: list[str] = config["retrieval"]["retrieval_models"]["dense"]
    sparse_models: list[str] = config["retrieval"]["retrieval_models"]["sparse"]
    reranker_models: list[str] = config["retrieval"]["retrieval_models"]["reranker"]
    rephrasing_options: list[bool] = config["retrieval"]["query_rephrasing"]

    llm_activated: bool = config["llm"]["activated"]
    llm_models: list[str] = config["llm"]["models"] if llm_activated else []

    all_combinations: list[dict] = []

    for dense in dense_models:
        for sparse in sparse_models:
            for reranker in reranker_models:
                for use_rephrasing in rephrasing_options:
                    label = f"{dense} | {sparse} | {reranker} | rephrasing={use_rephrasing}"
                    print(f"\n{'─'*60}\nEvaluating: {label}")

                    combo = await _evaluate_combination(
                        dense=dense,
                        sparse=sparse,
                        reranker=reranker,
                        use_rephrasing=use_rephrasing,
                        qa_pairs=qa_pairs,
                        k=k,
                        llm_models=llm_models,
                        dataset_path=dataset_path,
                    )
                    all_combinations.append(combo)
                    print(f"  avg retrieval : {combo['avg_retrieval_metrics']}")
                    if llm_activated:
                        print(f"  avg llm       : {combo['avg_llm_metrics']}")

    # ── Rank all combinations by LLM answer quality ──────────────────────────
    ranked = sorted(all_combinations, key=_combo_llm_score, reverse=True)

    output = {
        "ranked_combinations": ranked,
        "best_combination": ranked[0]["combination"] if ranked else None,
    }

    save_results(output)
    return output


if __name__ == "__main__":
    results = asyncio.run(full_eval("eval_config.yaml"))
    print("\nBest combination:")
    print(results["best_combination"])
