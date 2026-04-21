from fastapi import FastAPI
from pydantic import BaseModel
from typing import Literal
import rag_time.retrieval as ragrt
from rag_time.config.settings import settings


app = FastAPI(title="Search API")

class SupportTicketsRequest(BaseModel):
    query: str
    mode: Literal["semantic", "lexical", "hybrid"] = "hybrid"
    filter: dict | None = None


@app.post("/support_tickets")
def support_tickets(payload: SupportTicketsRequest) -> dict:
    return ragrt.rag_search(
        query_text=payload.query,
        collection_name=settings.collection_name,
        filter=payload.filter,
        method=payload.mode,
    )
