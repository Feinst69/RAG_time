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

# Raw result
@app.post(f"/{settings.collection_name}")
def root(payload: SupportTicketsRequest) -> dict:
    tickets= ragrt.rag_search(
        query_text=payload.query,
        collection_name=settings.collection_name,
        filter=payload.filter,
        method=payload.mode,
    )
    return {"translated": "not asked",
            "reranked": "not asked",
            "payload": tickets}

# Translated result
@app.post(f"/{settings.collection_name}/translated")
def translated(payload: SupportTicketsRequest) -> dict:
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
        return {"translated": "success", 
                "reranked": "not asked",
                "payload": translated_tickets}
    except Exception as e:
        print(f"Translation failed: {e}")
        return {"translated": "failed", 
                "payload": tickets.root}


# Reranked result  
@app.post(f"/{settings.collection_name}/reranked")
def reranked(payload: SupportTicketsRequest) -> dict:
    tickets= ragrt.rag_search(
        query_text=payload.query,
        collection_name=settings.collection_name,
        filter=payload.filter,
        method=payload.mode,
    )
    tickets = ragrt.rerank(payload.query, tickets)
    return {"translated": "not asked", 
            "reranked": "success",
            "payload": tickets}


# Reranked then  translated result
@app.post(f"/{settings.collection_name}/reranked/translated")
def reranked_translated(payload: SupportTicketsRequest) -> dict:
    tickets= ragrt.rag_search(
        query_text=payload.query,
        collection_name=settings.collection_name,
        filter=payload.filter,
        method=payload.mode,
    )
    tickets = ragrt.rerank(payload.query, tickets)
    tickets = RetrievedTicketsDict(root=tickets)
    try:
        translated_tickets = translate_tickets(payload.query, tickets)
        translated_tickets = translated_tickets.tickets_out.root
        return {"translated": "success", 
                "reranked": "success",
                "payload": translated_tickets}
    except Exception as e:
        print(f"Translation failed: {e}")
        return {"translated": "failed", 
                "reranked": "success",
                "payload": tickets.root}

@app.post(f"/{settings.collection_name}/translated/reranked")
def translated_reranked_(payload: SupportTicketsRequest) -> dict:
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
        translated_tickets = ragrt.rerank(payload.query, translated_tickets)
        return {"translated": "success", 
                "reranked": "success",
                "payload": translated_tickets}
    except Exception as e:
        print(f"Translation failed: {e}")
        tickets = ragrt.rerank(payload.query, tickets.root)
        return {"translated": "failed", 
                "reranked": "success",
                "payload": tickets}