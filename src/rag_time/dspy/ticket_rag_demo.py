"""Ticket RAG demo pipeline.

Based on dev_test.py with:
  - Template classification / routing removed (part 3)
  - Hardcoded conversation history removed
  - fake_rag_chunks() replaced by real HybridRetriever
  - Direct answer signature tailored for customer support tickets

Flow:
  Step 1 – DSPy + LM setup
  Step 2 – Query rephrasing
  Step 3 – Hybrid retrieval (Qdrant: dense + sparse + ColBERT rerank)
  Step 4 – Answer generation (streaming)
  Step 5 – Guardrail check
  Step 6 – Cost summary
"""

from __future__ import annotations

import asyncio
import os
from textwrap import dedent

import dspy
from dotenv import load_dotenv
from qdrant_client import QdrantClient

from rag_time.dspy.agents.guardian import Guardrail
from rag_time.dspy.agents.query_rephraser import QueryRephraser
from rag_time.dspy.config import ConfigError, DSPyConfig
from rag_time.dspy.cost import get_info
from rag_time.rag.config import DEFAULT_TOP_K, RETRIEVAL_CANDIDATES, QDRANT_URL
from rag_time.rag.models import FilterParams, SearchResult
from rag_time.rag.retriever import HybridRetriever


# ---------------------------------------------------------------------------
# LM setup
# ---------------------------------------------------------------------------

def configure_lm() -> None:
    load_dotenv(override=True)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY missing – set it in .env")
    lm = dspy.LM(
        model="openrouter/google/gemini-2.5-pro-preview",
        api_base="https://openrouter.ai/api/v1",
        api_key=api_key,
    )
    dspy.configure(lm=lm)


# ---------------------------------------------------------------------------
# Answer signature (replaces RoutedAnswerSignature – no template routing)
# ---------------------------------------------------------------------------

class TicketAnswerSignature(dspy.Signature):
    """Answer a customer support question using retrieved ticket data.

    You are an expert support analyst. Use ONLY the provided ticket data to
    answer the question. Be concise, factual, and reference specific tickets
    when relevant. If the data does not contain enough information, say so.
    """

    query: str = dspy.InputField(desc="The user's question about customer support tickets")
    documents: str = dspy.InputField(desc="Relevant ticket data retrieved from the support database")
    answer: str = dspy.OutputField(desc="Clear, factual answer grounded in the provided ticket data")


# ---------------------------------------------------------------------------
# Ticket formatting
# ---------------------------------------------------------------------------

def format_results(results: list[SearchResult]) -> str:
    """Format HybridRetriever results into a document block for the LLM."""
    if not results:
        return "No relevant tickets found."

    lines: list[str] = []
    for r in results:
        p = r.payload
        lines.append(
            f"Ticket #{r.ticket_id} "
            f"[{p.get('priority', '?')} | {p.get('product', '?')} | {p.get('category', '?')}]"
        )
        lines.append(f"  Customer : {p.get('customer_name', '?')} "
                     f"({p.get('customer_segment', '?')}, {p.get('region', '?')})")
        lines.append(f"  Issue    : {p.get('issue_description', '')}")
        lines.append(f"  Resolution: {p.get('resolution_notes', '')}")
        lines.append(
            f"  SLA breached: {p.get('sla_breached')} | "
            f"Escalated: {p.get('escalated')} | "
            f"Resolution time: {p.get('resolution_time_hours')}h | "
            f"Satisfaction: {p.get('customer_satisfaction_score')}"
        )
        lines.append("")

    return "\n".join(lines).strip()


# ---------------------------------------------------------------------------
# Cost helper (unchanged from dev_test.py)
# ---------------------------------------------------------------------------

def summarize_history(entries: list[dict]) -> dict[str, float]:
    prompt_tokens = completion_tokens = 0
    total_cost = 0.0
    temp_entries = [dict(item) for item in entries]
    while temp_entries:
        current_prompt, current_completion, current_cost = get_info(temp_entries)
        temp_entries.pop(0)
        prompt_tokens += current_prompt
        completion_tokens += current_completion
        total_cost += current_cost
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cost": total_cost,
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

async def run_pipeline(
    query: str,
    filters: FilterParams | None = None,
    top_k: int = DEFAULT_TOP_K,
    candidates: int = RETRIEVAL_CANDIDATES,
    qdrant_url: str = QDRANT_URL,
) -> dict:
    lm_history = getattr(dspy.settings.lm, "history", [])

    # ── Step 1 – DSPy setup ──────────────────────────────────────────────────
    print("---\nStep 1 – DSPy setup")
    query_rephraser = QueryRephraser()
    guardian = Guardrail()

    # ── Step 2 – Query rephrasing ────────────────────────────────────────────
    print("---\nStep 2 – Query rephrasing")
    print(f"Original query : {query}")

    empty_history = dspy.History(messages=[])
    before_rephrase = len(lm_history)
    rephrased_query = await query_rephraser(question=query, history=empty_history)
    rephrase_cost = summarize_history(lm_history[before_rephrase:])
    print(f"Rephrased query: {rephrased_query}")

    # ── Step 3 – Hybrid retrieval ────────────────────────────────────────────
    print("---\nStep 3 – Hybrid retrieval (Qdrant)")
    client = QdrantClient(url=qdrant_url)
    retriever = HybridRetriever(client=client)
    results = retriever.search(
        query=rephrased_query,
        filters=filters,
        candidates=candidates,
        top_k=top_k,
    )
    print(f"Retrieved {len(results)} tickets")
    for r in results:
        print(f"  #{r.ticket_id} score={r.score:.4f}  {r.payload.get('category')} | {r.payload.get('product')}")

    documents_block = format_results(results)

    # ── Step 4 – Answer generation (streaming) ───────────────────────────────
    print("---\nStep 4 – Answer generation (streaming)")
    writer_module = dspy.ChainOfThought(TicketAnswerSignature)
    listener = dspy.streaming.StreamListener(signature_field_name="answer")
    stream_writer = dspy.streamify(writer_module, stream_listeners=[listener])

    before_writer = len(lm_history)
    stream = stream_writer(query=rephrased_query, documents=documents_block)

    final_answer_parts: list[str] = []
    chunk_index = 1
    async for chunk in stream:
        print(f"[stream chunk {chunk_index}] {chunk}", flush=True)
        if hasattr(chunk, "chunk") and chunk.chunk:
            final_answer_parts.append(chunk.chunk)
        chunk_index += 1

    final_answer = "".join(final_answer_parts)
    generation_cost = summarize_history(lm_history[before_writer:])
    print("\n---\n(Streaming done)")

    # ── Step 5 – Guardrail ───────────────────────────────────────────────────
    print("---\nStep 5 – Guardrail")
    before_guard = len(lm_history)
    guard = await guardian(query=final_answer)
    guard_cost = summarize_history(lm_history[before_guard:])
    print(f"Hazard: {guard.hazard.category} (confidence {guard.confidence:.2f})")

    # ── Step 6 – Cost summary ────────────────────────────────────────────────
    summary = {
        "answer": final_answer,
        "costs": {
            "query_rephrase": rephrase_cost,
            "main_answer": generation_cost,
            "guardrail": guard_cost,
        },
    }
    print("---\nStep 6 – Cost summary")
    print(summary["costs"])

    return summary


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    configure_lm()

    # Example: plain semantic query
    asyncio.run(run_pipeline(
        query="payment failed but money was deducted",
    ))

    # Example: query with metadata filters
    # asyncio.run(run_pipeline(
    #     query="subscription cancelled without request",
    #     filters=FilterParams(product="Payment Gateway", priority="Urgent", sla_breached=True),
    #     top_k=5,
    # ))
