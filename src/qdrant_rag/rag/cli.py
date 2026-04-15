"""Typer CLI for the Qdrant RAG pipeline.

Commands
--------
  create-collection   Set up the Qdrant collection with vectors and payload indexes.
  index               Ingest a CSV file into Qdrant.
  search              Run a hybrid search query with optional metadata filters.
  info                Print collection statistics.

Usage (via uv)
--------------
  uv run python -m qdrant_rag.rag.cli --help
  uv run python -m qdrant_rag.rag.cli create-collection
  uv run python -m qdrant_rag.rag.cli index --csv data/customer_support_tickets_200k.csv
  uv run python -m qdrant_rag.rag.cli search "payment issue with subscription"
  uv run python -m qdrant_rag.rag.cli search "crash on upload" --product "Web Portal" --priority Urgent
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from qdrant_client import QdrantClient

from qdrant_rag.rag.collection import collection_info, create_collection
from qdrant_rag.rag.config import (
    COLLECTION_NAME,
    DEFAULT_TOP_K,
    INDEX_BATCH_SIZE,
    QDRANT_URL,
    RETRIEVAL_CANDIDATES,
)
from qdrant_rag.rag.indexer import ingest_csv
from qdrant_rag.rag.models import FilterParams
from qdrant_rag.rag.retriever import HybridRetriever

app = typer.Typer(
    name="qdrant-rag",
    help="Hybrid RAG pipeline: dense (ColBERT) + sparse (BM25) + ColBERT reranking.",
    no_args_is_help=True,
)

# ---------------------------------------------------------------------------
# create-collection
# ---------------------------------------------------------------------------

@app.command("create-collection")
def cmd_create_collection(
    url: str = typer.Option(QDRANT_URL, "--url", help="Qdrant server URL."),
    recreate: bool = typer.Option(False, "--recreate", help="Drop and recreate the collection."),
) -> None:
    """Create the Qdrant collection with payload indexes."""
    client = QdrantClient(url=url)
    create_collection(client, recreate=recreate)


# ---------------------------------------------------------------------------
# index
# ---------------------------------------------------------------------------

@app.command("index")
def cmd_index(
    csv: Path = typer.Option(
        Path("data/customer_support_tickets_200k.csv"),
        "--csv", "-f",
        help="Path to the customer support tickets CSV.",
    ),
    url: str = typer.Option(QDRANT_URL, "--url", help="Qdrant server URL."),
    batch_size: int = typer.Option(INDEX_BATCH_SIZE, "--batch-size", "-b", help="Rows per upsert batch."),
    limit: Optional[int] = typer.Option(None, "--limit", "-n", help="Ingest only the first N rows (for testing)."),
) -> None:
    """Embed and ingest the CSV into Qdrant."""
    if not csv.exists():
        typer.echo(f"CSV not found: {csv}", err=True)
        raise typer.Exit(1)

    client = QdrantClient(url=url)
    ingest_csv(csv_path=csv, client=client, batch_size=batch_size, limit=limit)


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------

@app.command("search")
def cmd_search(
    query: str = typer.Argument(..., help="Natural-language search query."),
    # retrieval params
    url: str = typer.Option(QDRANT_URL, "--url", help="Qdrant server URL."),
    top_k: int = typer.Option(DEFAULT_TOP_K, "--top-k", "-k", help="Number of results to return."),
    candidates: int = typer.Option(RETRIEVAL_CANDIDATES, "--candidates", "-c", help="Candidates before reranking."),
    no_rerank: bool = typer.Option(False, "--no-rerank", help="Skip ColBERT reranking (faster)."),
    json_output: bool = typer.Option(False, "--json", help="Output results as JSON."),
    # ---- keyword filters ----
    customer_name: Optional[str] = typer.Option(None, "--customer-name"),
    customer_email: Optional[str] = typer.Option(None, "--customer-email"),
    product: Optional[str] = typer.Option(None, "--product"),
    category: Optional[str] = typer.Option(None, "--category"),
    priority: Optional[str] = typer.Option(None, "--priority"),
    channel: Optional[str] = typer.Option(None, "--channel"),
    region: Optional[str] = typer.Option(None, "--region"),
    customer_gender: Optional[str] = typer.Option(None, "--customer-gender"),
    subscription_type: Optional[str] = typer.Option(None, "--subscription-type"),
    operating_system: Optional[str] = typer.Option(None, "--os"),
    browser: Optional[str] = typer.Option(None, "--browser"),
    payment_method: Optional[str] = typer.Option(None, "--payment-method"),
    language: Optional[str] = typer.Option(None, "--language"),
    customer_segment: Optional[str] = typer.Option(None, "--segment"),
    # ---- numeric range filters ----
    age_min: Optional[int] = typer.Option(None, "--age-min"),
    age_max: Optional[int] = typer.Option(None, "--age-max"),
    complexity_min: Optional[int] = typer.Option(None, "--complexity-min"),
    complexity_max: Optional[int] = typer.Option(None, "--complexity-max"),
    satisfaction_min: Optional[float] = typer.Option(None, "--satisfaction-min"),
    satisfaction_max: Optional[float] = typer.Option(None, "--satisfaction-max"),
    resolution_hours_min: Optional[float] = typer.Option(None, "--resolution-hours-min"),
    resolution_hours_max: Optional[float] = typer.Option(None, "--resolution-hours-max"),
    # ---- datetime range filters ----
    created_from: Optional[str] = typer.Option(None, "--created-from", help="ISO date e.g. 2023-01-01"),
    created_to: Optional[str] = typer.Option(None, "--created-to"),
    resolved_from: Optional[str] = typer.Option(None, "--resolved-from"),
    resolved_to: Optional[str] = typer.Option(None, "--resolved-to"),
    # ---- boolean filters ----
    escalated: Optional[bool] = typer.Option(None, "--escalated/--not-escalated"),
    sla_breached: Optional[bool] = typer.Option(None, "--sla-breached/--sla-ok"),
) -> None:
    """Hybrid search with optional metadata filters."""
    filters = FilterParams(
        customer_name=customer_name,
        customer_email=customer_email,
        product=product,
        category=category,
        priority=priority,
        channel=channel,
        region=region,
        customer_gender=customer_gender,
        subscription_type=subscription_type,
        operating_system=operating_system,
        browser=browser,
        payment_method=payment_method,
        language=language,
        customer_segment=customer_segment,
        customer_age_min=age_min,
        customer_age_max=age_max,
        issue_complexity_score_min=complexity_min,
        issue_complexity_score_max=complexity_max,
        customer_satisfaction_score_min=satisfaction_min,
        customer_satisfaction_score_max=satisfaction_max,
        resolution_time_hours_min=resolution_hours_min,
        resolution_time_hours_max=resolution_hours_max,
        ticket_created_date_from=created_from,
        ticket_created_date_to=created_to,
        ticket_resolved_date_from=resolved_from,
        ticket_resolved_date_to=resolved_to,
        escalated=escalated,
        sla_breached=sla_breached,
    )

    client = QdrantClient(url=url)
    retriever = HybridRetriever(client=client)

    if no_rerank:
        results = retriever.search_no_rerank(query=query, filters=filters, top_k=top_k)
    else:
        results = retriever.search(query=query, filters=filters, candidates=candidates, top_k=top_k)

    if json_output:
        typer.echo(json.dumps([r.model_dump() for r in results], indent=2))
    else:
        typer.echo(f"\n{'─'*70}")
        typer.echo(f"Query : {query!r}")
        if not filters.is_empty():
            active = {k: v for k, v in filters.model_dump().items() if v is not None}
            typer.echo(f"Filter: {active}")
        typer.echo(f"{'─'*70}\n")
        for i, result in enumerate(results, 1):
            typer.echo(f"#{i:02d}  {result}\n")


# ---------------------------------------------------------------------------
# info
# ---------------------------------------------------------------------------

@app.command("info")
def cmd_info(
    url: str = typer.Option(QDRANT_URL, "--url", help="Qdrant server URL."),
) -> None:
    """Print collection statistics."""
    client = QdrantClient(url=url)
    info = collection_info(client)
    typer.echo(json.dumps(info, indent=2, default=str))


# ---------------------------------------------------------------------------
# Entry point (allows `python -m qdrant_rag.rag.cli`)
# ---------------------------------------------------------------------------

def main() -> None:
    app()


if __name__ == "__main__":
    main()
