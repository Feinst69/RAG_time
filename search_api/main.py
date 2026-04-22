from fastapi import FastAPI
from pydantic import BaseModel
from typing import Literal
from rag_time.data_loader import RetrievedTicketsDict
import rag_time.retrieval as ragrt
from rag_time.config.settings import settings
from rag_time.translator import translate_tickets


app = FastAPI(title="Search API")

class SupportTicketsRequest(BaseModel):
    query: str
    mode: Literal["semantic", "lexical", "hybrid"] = "hybrid"
    filter: dict | None = None


@app.post("/support_tickets")
def support_tickets(payload: SupportTicketsRequest) -> dict:
    tickets= ragrt.rag_search(
        query_text=payload.query,
        collection_name=settings.collection_name,
        filter=payload.filter,
        method=payload.mode,
    )
    return {"translated": "not asked", "payload": tickets}

@app.post("/translated_support_tickets")
def translated_support_tickets(payload: SupportTicketsRequest) -> dict:
    tickets= ragrt.rag_search(
        query_text=payload.query,
        collection_name=settings.collection_name,
        filter=payload.filter,
        method=payload.mode,
    )
    tickets = RetrievedTicketsDict(root=tickets)
    try:
        translated_tickets = translate_tickets(payload.query, tickets)
        translated_tickets = translated_tickets.tickets_out.root
        return {"translated": "success", "payload": translated_tickets}
    except Exception as e:
        print(f"Translation failed: {e}")
        return {"translated": "failed", "payload": tickets.root}
