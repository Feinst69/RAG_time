import dspy
import re

from rag_time.config.settings import settings
from rag_time.data_loader import RetrievedTicketsDict


def init_lm() -> dspy.LM:
    if not settings.openrouter_api_key:
        raise RuntimeError("OPENROUTER_API_KEY is required to synthesize tickets")

    model = settings.openrouter_model.strip()
    if not model:
        raise RuntimeError("OPENROUTER_MODEL is required to synthesize tickets")
    if not model.startswith("openrouter/"):
        model = f"openrouter/{model}"

    return dspy.LM(
        api_key=settings.openrouter_api_key,
        api_base=settings.openrouter_api_base,
        model=model,
        temperature=settings.openrouter_temperature,
        timeout=settings.openrouter_timeout,
    )


def detect_query_language(query: str) -> str:
    text = query.lower()
    words = set(re.findall(r"[a-zàâçéèêëîïôûùüÿñæœ]+", text))

    if re.search(r"[\u0400-\u04ff]", query):
        return "Russian"

    if re.search(r"[àâçéèêëîïôûùüÿæœ]", text):
        return "French"

    language_markers = {
        "French": {
            "le", "la", "les", "des", "du", "un", "une", "avec", "dans", "pour",
            "comment", "pourquoi", "probleme", "problème", "ecran", "écran",
            "tactile", "ordinateur", "logiciel", "application", "panne",
        },
        "English": {
            "the", "with", "and", "for", "how", "why", "issue", "problem",
            "screen", "touchscreen", "software", "application", "computer",
        },
        "Spanish": {
            "el", "los", "las", "con", "para", "como", "por", "problema",
            "pantalla", "tactil", "táctil", "software", "aplicacion", "aplicación",
        },
        "German": {
            "der", "die", "das", "mit", "und", "fur", "für", "warum", "problem",
            "bildschirm", "touchscreen", "software", "anwendung", "computer",
        },
    }

    scores = {
        language: len(words & markers)
        for language, markers in language_markers.items()
    }
    best_language, best_score = max(scores.items(), key=lambda item: item[1])
    return best_language if best_score > 0 else "the same language as the user query"


class TicketSynthesisSignature(dspy.Signature):
    """
    Answer the user query using only the retrieved support tickets.
    Produce a factual, coherent answer strictly in target_language.
    Translate the relevant facts from the retrieved tickets into target_language when needed.
    Never choose the answer language from the retrieved tickets or from their metadata.
    If the retrieved tickets do not contain enough information, say so explicitly.
    Do not invent procedures, causes, phone numbers, account numbers or policy details.
    """

    query: str = dspy.InputField(desc="user query")
    target_language: str = dspy.InputField(desc="mandatory output language")
    tickets_in: RetrievedTicketsDict = dspy.InputField(desc="retrieved support tickets")
    answer: str = dspy.OutputField(
        desc="factual synthesized answer grounded in the tickets; write only in target_language"
    )


def synthesize_answer(query: str, tickets_in: RetrievedTicketsDict) -> str:
    target_language = detect_query_language(query)
    lm = init_lm()
    with dspy.context(lm=lm):
        synthesizer = dspy.Predict(signature=TicketSynthesisSignature)
        result = synthesizer(
            query=query,
            target_language=target_language,
            tickets_in=tickets_in,
        )
        return result.answer
