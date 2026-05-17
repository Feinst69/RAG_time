import importlib
import sys
import types
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


SAMPLE_TICKETS = {
    "ticket-1": {
        "score": 0.42,
        "subject": "Application crash",
        "body": "The app crashes on a 4K monitor.",
        "answer": "Update the graphics driver.",
        "type": "Problem",
        "queue": "Technical Support",
        "priority": "high",
        "language": "en",
    },
    "ticket-2": {
        "score": 0.21,
        "subject": "Billing question",
        "body": "The invoice total is wrong.",
        "answer": "Contact the billing queue.",
        "type": "Question",
        "queue": "Billing",
        "priority": "medium",
        "language": "en",
    },
}


def translated_copy(tickets: dict) -> dict:
    translated = {}
    for ticket_id, ticket in tickets.items():
        ticket_data = ticket.model_dump() if hasattr(ticket, "model_dump") else ticket
        translated[ticket_id] = {
            **ticket_data,
            "subject": f"[fr] {ticket_data['subject']}",
            "body": f"[fr] {ticket_data['body']}",
            "answer": f"[fr] {ticket_data['answer']}",
        }
    return translated


@pytest.fixture()
def api_client(monkeypatch):
    calls = []

    retrieval = types.ModuleType("rag_time.retrieval")

    def rag_search(query_text, collection_name, filter=None, method="hybrid"):
        calls.append(
            {
                "step": "search",
                "query": query_text,
                "collection": collection_name,
                "filter": filter,
                "method": method,
            }
        )
        return SAMPLE_TICKETS

    def rerank(query_text, results):
        calls.append(
            {
                "step": "rerank",
                "query": query_text,
                "ticket_ids": list(results.keys()),
            }
        )
        reranked = {}
        for ticket_id, ticket in reversed(results.items()):
            ticket_data = ticket.model_dump() if hasattr(ticket, "model_dump") else ticket
            reranked[ticket_id] = {**ticket_data, "score": ticket_data["score"] + 1}
        return reranked

    def get_filter_options(collection_name):
        calls.append({"step": "filters", "collection": collection_name})
        return {
            "language": ["en", "fr"],
            "priority": ["high", "medium"],
            "queue": ["Billing", "Technical Support"],
            "type": ["Problem", "Question"],
        }

    retrieval.rag_search = rag_search
    retrieval.rerank = rerank
    retrieval.get_filter_options = get_filter_options

    translator = types.ModuleType("rag_time.translator")

    def translate_tickets(query, tickets_in):
        calls.append(
            {
                "step": "translate",
                "query": query,
                "ticket_ids": list(tickets_in.root.keys()),
            }
        )
        return types.SimpleNamespace(
            tickets_out=types.SimpleNamespace(root=translated_copy(tickets_in.root))
        )

    translator.translate_tickets = translate_tickets

    synthesizer = types.ModuleType("rag_time.synthesizer")

    def synthesize_answer(query, tickets_in):
        calls.append(
            {
                "step": "synthesize",
                "query": query,
                "ticket_ids": list(tickets_in.root.keys()),
            }
        )
        return "Réponse synthétique basée sur les tickets récupérés."

    synthesizer.synthesize_answer = synthesize_answer

    monkeypatch.setitem(sys.modules, "rag_time.retrieval", retrieval)
    monkeypatch.setitem(sys.modules, "rag_time.translator", translator)
    monkeypatch.setitem(sys.modules, "rag_time.synthesizer", synthesizer)
    sys.modules.pop("search_api.main", None)

    module = importlib.import_module("search_api.main")
    return TestClient(module.app), calls


def payload(mode="hybrid"):
    return {
        "query": "écran noir application",
        "mode": mode,
        "filter": {"language": "en", "priority": "high"},
    }


def assert_search_call(calls, mode):
    assert calls[0] == {
        "step": "search",
        "query": "écran noir application",
        "collection": "support_tickets",
        "filter": {"language": "en", "priority": "high"},
        "method": mode,
    }


def assert_translated_payload(data):
    assert data["payload"]["ticket-1"]["subject"] == "[fr] Application crash"
    assert data["payload"]["ticket-1"]["body"] == "[fr] The app crashes on a 4K monitor."
    assert data["payload"]["ticket-1"]["answer"] == "[fr] Update the graphics driver."
    assert data["payload"]["ticket-1"]["language"] == "en"


@pytest.mark.parametrize("mode", ["semantic", "lexical", "hybrid"])
def test_raw_search_endpoint_calls_rag_search(api_client, mode):
    client, calls = api_client

    response = client.post("/support_tickets", json=payload(mode))

    assert response.status_code == 200
    data = response.json()
    assert data["translated"] == "not asked"
    assert data["reranked"] == "not asked"
    assert data["payload"] == SAMPLE_TICKETS
    assert_search_call(calls, mode)
    assert [call["step"] for call in calls] == ["search"]


@pytest.mark.parametrize("mode", ["semantic", "lexical", "hybrid"])
def test_translated_endpoint_searches_then_translates(api_client, mode):
    client, calls = api_client

    response = client.post("/support_tickets/translated", json=payload(mode))

    assert response.status_code == 200
    data = response.json()
    assert data["translated"] == "success"
    assert data["reranked"] == "not asked"
    assert_translated_payload(data)
    assert_search_call(calls, mode)
    assert [call["step"] for call in calls] == ["search", "translate"]


@pytest.mark.parametrize("mode", ["semantic", "lexical", "hybrid"])
def test_reranked_endpoint_searches_then_reranks(api_client, mode):
    client, calls = api_client

    response = client.post("/support_tickets/reranked", json=payload(mode))

    assert response.status_code == 200
    data = response.json()
    assert data["translated"] == "not asked"
    assert data["reranked"] == "success"
    assert list(data["payload"]) == ["ticket-2", "ticket-1"]
    assert data["payload"]["ticket-2"]["score"] == pytest.approx(1.21)
    assert_search_call(calls, mode)
    assert [call["step"] for call in calls] == ["search", "rerank"]


@pytest.mark.parametrize("mode", ["semantic", "lexical", "hybrid"])
def test_synthesized_endpoint_searches_then_synthesizes(api_client, mode):
    client, calls = api_client

    response = client.post("/support_tickets/synthesized", json=payload(mode))

    assert response.status_code == 200
    data = response.json()
    assert data["translated"] == "not asked"
    assert data["reranked"] == "not asked"
    assert data["synthesized"] == "success"
    assert data["answer"] == "Réponse synthétique basée sur les tickets récupérés."
    assert data["payload"] == SAMPLE_TICKETS
    assert_search_call(calls, mode)
    assert [call["step"] for call in calls] == ["search", "synthesize"]


def test_synthesize_existing_endpoint_does_not_search(api_client):
    client, calls = api_client

    response = client.post(
        "/support_tickets/synthesize",
        json={
            "query": "écran noir application",
            "tickets": SAMPLE_TICKETS,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["synthesized"] == "success"
    assert data["answer"] == "Réponse synthétique basée sur les tickets récupérés."
    assert data["payload"] == SAMPLE_TICKETS
    assert calls == [
        {
            "step": "synthesize",
            "query": "écran noir application",
            "ticket_ids": ["ticket-1", "ticket-2"],
        }
    ]


@pytest.mark.parametrize("mode", ["semantic", "lexical", "hybrid"])
def test_reranked_then_translated_endpoint_uses_expected_order(api_client, mode):
    client, calls = api_client

    response = client.post("/support_tickets/reranked/translated", json=payload(mode))

    assert response.status_code == 200
    data = response.json()
    assert data["translated"] == "success"
    assert data["reranked"] == "success"
    assert data["payload"]["ticket-2"]["subject"] == "[fr] Billing question"
    assert_search_call(calls, mode)
    assert [call["step"] for call in calls] == ["search", "rerank", "translate"]


@pytest.mark.parametrize("mode", ["semantic", "lexical", "hybrid"])
def test_translated_then_reranked_endpoint_uses_expected_order(api_client, mode):
    client, calls = api_client

    response = client.post("/support_tickets/translated/reranked", json=payload(mode))

    assert response.status_code == 200
    data = response.json()
    assert data["translated"] == "success"
    assert data["reranked"] == "success"
    assert list(data["payload"]) == ["ticket-2", "ticket-1"]
    assert data["payload"]["ticket-2"]["subject"] == "[fr] Billing question"
    assert_search_call(calls, mode)
    assert [call["step"] for call in calls] == ["search", "translate", "rerank"]


def test_health_endpoint_returns_collection(api_client):
    client, calls = api_client

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "collection": "support_tickets"}
    assert calls == []


def test_filters_endpoint_delegates_to_retrieval(api_client):
    client, calls = api_client

    response = client.get("/filters")

    assert response.status_code == 200
    assert response.json()["language"] == ["en", "fr"]
    assert calls == [{"step": "filters", "collection": "support_tickets"}]
