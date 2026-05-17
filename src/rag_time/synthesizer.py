import dspy

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


class TicketSynthesisSignature(dspy.Signature):
    """
    Answer the user query using only the retrieved support tickets.
    Produce a factual, coherent answer in the same language as the query.
    If the retrieved tickets do not contain enough information, say so explicitly.
    Do not invent procedures, causes, phone numbers, account numbers or policy details.
    """

    query: str = dspy.InputField(desc="user query")
    tickets_in: RetrievedTicketsDict = dspy.InputField(desc="retrieved support tickets")
    answer: str = dspy.OutputField(desc="factual synthesized answer grounded in the tickets")


def synthesize_answer(query: str, tickets_in: RetrievedTicketsDict) -> str:
    lm = init_lm()
    with dspy.context(lm=lm):
        synthesizer = dspy.Predict(signature=TicketSynthesisSignature)
        result = synthesizer(query=query, tickets_in=tickets_in)
        return result.answer
