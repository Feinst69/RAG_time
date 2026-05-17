from pathlib import Path
from typing import Literal

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from rag_time.data_loader import RetrievedTicketsDict
import rag_time.retrieval as ragrt
from rag_time.config.settings import settings
from rag_time.translator import translate_tickets
from rag_time.synthesizer import synthesize_answer


app = FastAPI(title="Search API")
STATIC_DIR = Path(__file__).parent / "static"

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

class SupportTicketsRequest(BaseModel):
    query: str
    mode: Literal["semantic", "lexical", "hybrid"] = "hybrid"
    filter: dict | None = None


class SynthesisRequest(BaseModel):
    query: str
    tickets: dict


@app.get("/")
def home() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "collection": settings.collection_name}


@app.get("/filters")
def filters() -> dict[str, list[str]]:
    return ragrt.get_filter_options(settings.collection_name)

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
            "synthesized": "not asked",
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
                "synthesized": "not asked",
                "payload": translated_tickets}
    except Exception as e:
        print(f"Translation failed: {e}")
        return {"translated": "failed", 
                "reranked": "not asked",
                "synthesized": "not asked",
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
            "synthesized": "not asked",
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
                "synthesized": "not asked",
                "payload": translated_tickets}
    except Exception as e:
        print(f"Translation failed: {e}")
        return {"translated": "failed", 
                "reranked": "success",
                "synthesized": "not asked",
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
                "synthesized": "not asked",
                "payload": translated_tickets}
    except Exception as e:
        print(f"Translation failed: {e}")
        tickets = ragrt.rerank(payload.query, tickets.root)
        return {"translated": "failed", 
                "reranked": "success",
                "synthesized": "not asked",
                "payload": tickets}


@app.post(f"/{settings.collection_name}/synthesized")
def synthesized(payload: SupportTicketsRequest) -> dict:
    tickets = ragrt.rag_search(
        query_text=payload.query,
        collection_name=settings.collection_name,
        filter=payload.filter,
        method=payload.mode,
    )
    tickets_in = RetrievedTicketsDict(root=tickets)
    try:
        answer = synthesize_answer(payload.query, tickets_in)
        return {
            "translated": "not asked",
            "reranked": "not asked",
            "synthesized": "success",
            "answer": answer,
            "payload": tickets,
        }
    except Exception as e:
        print(f"Synthesis failed: {e}")
        return {
            "translated": "not asked",
            "reranked": "not asked",
            "synthesized": "failed",
            "answer": "",
            "payload": tickets,
        }


@app.post(f"/{settings.collection_name}/synthesize")
def synthesize_existing(payload: SynthesisRequest) -> dict:
    tickets_in = RetrievedTicketsDict(root=payload.tickets)
    try:
        answer = synthesize_answer(payload.query, tickets_in)
        return {
            "synthesized": "success",
            "answer": answer,
            "payload": payload.tickets,
        }
    except Exception as e:
        print(f"Synthesis failed: {e}")
        return {
            "synthesized": "failed",
            "answer": "",
            "error": str(e),
            "payload": payload.tickets,
        }
