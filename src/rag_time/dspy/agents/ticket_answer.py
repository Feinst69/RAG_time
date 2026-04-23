import os

import dspy
from dotenv import load_dotenv

from rag_time.dspy.config import DSPyConfig
from rag_time.dspy.objects.ticket_resolution import TicketResolution


def _setup_lm() -> None:
    """Configure DSPy global LM from config + env if not already set."""
    if dspy.settings.lm is not None:
        return
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY missing – set it in .env")
    config = DSPyConfig()
    dspy.configure(lm=dspy.LM(
        model=config.lm_model,
        api_base="https://openrouter.ai/api/v1",
        api_key=api_key,
    ))


def _format_payload(payload: dict) -> list[str]:
    """Extract and format ticket documents from the API response payload."""
    docs = []
    for ticket_id, ticket in payload.items():
        parts = [f"[Ticket #{ticket_id}]"]
        if subject := ticket.get("subject"):
            parts.append(f"Subject: {subject}")
        if body := ticket.get("body"):
            parts.append(f"Body: {body}")
        if answer := ticket.get("answer"):
            parts.append(f"Previous answer: {answer}")
        if ticket.get("type") or ticket.get("queue") or ticket.get("priority"):
            parts.append(f"Type: {ticket.get('type')} | Queue: {ticket.get('queue')} | Priority: {ticket.get('priority')}")
        docs.append("\n".join(parts))
    return docs


async def answer_from_api_response(query: str, api_response: dict, streaming: bool = False) -> TicketResolution | str:
    """Production entry point: accepts the full API JSON response, extracts documents, generates answer."""
    _setup_lm()
    payload = api_response.get("payload", api_response)
    documents = _format_payload(payload)
    return await generate_answer(query=query, documents=documents, streaming=streaming)


async def generate_answer(query: str, documents: list[str] | str, streaming: bool = True) -> TicketResolution | str:
    """Generate a structured resolution from retrieved ticket documents.

    Args:
        query:     The (optionally rephrased) user question.
        documents: Formatted ticket data block from the retriever.
        streaming: If True, streams chunks to stdout and returns the joined answer string.
                   If False, runs a single synchronous prediction and returns a TicketResolution.

    Returns:
        TicketResolution (non-streaming) or raw answer string (streaming).
    """
    docs_str = "\n\n".join(documents) if isinstance(documents, list) else documents
    writer_module = dspy.ChainOfThought(TicketAnswerSignature)

    if not streaming:
        writer = dspy.asyncify(writer_module)
        result = await writer(query=query, documents=docs_str)
        return result.resolution

    # Streaming path: field-level streaming not yet supported for Pydantic output fields,
    # so we stream the raw answer string from a plain-string signature.
    listener = dspy.streaming.StreamListener(signature_field_name="answer")
    plain_module = dspy.ChainOfThought(TicketAnswerSignatureStreaming)
    stream_writer = dspy.streamify(plain_module, stream_listeners=[listener])
    parts: list[str] = []
    async for chunk in stream_writer(query=query, documents=docs_str):
        if hasattr(chunk, "chunk") and chunk.chunk:
            parts.append(chunk.chunk)
    return "".join(parts)


class TicketAnswerSignature(dspy.Signature):
    """Generate a structured resolution for a customer support query using retrieved ticket data.

    You are an expert support analyst. Use ONLY the provided ticket data to produce a
    three-part structured resolution:
      1. Clarify the identified issue.
      2. List actionable solutions, each citing the ticket(s) they come from.
      3. Draft a ready-to-send message to the user using [Username] for the user and
         [helper_name] for the support agent who will follow up.

    Be concise, factual, and answer in the original language of the helper.
    If the data does not contain enough information, say so explicitly.
    """

    query: str = dspy.InputField(desc="The helper's question about previous customer support tickets")
    documents: str = dspy.InputField(desc="Relevant ticket data retrieved from the support database")
    resolution: TicketResolution = dspy.OutputField(
        desc="Structured three-part resolution: issue clarification, proposed solutions with ticket citations, and a ready-to-send user message"
    )


class TicketAnswerSignatureStreaming(dspy.Signature):
    """Answer a customer support question using retrieved ticket data (streaming mode).

    You are an expert support analyst. Use ONLY the provided ticket data to build a
    clear, factual summary identifying the issue's key points and potential solutions.
    Structure your answer in three parts:
      1. Issue clarification
      2. Proposed solutions with ticket citations ([Ticket #<id>])
      3. A proposed message to the user using [Username] and [helper_name] as placeholders.

    Be concise, factual, and answer in the original language of the helper.
    """

    query: str = dspy.InputField(desc="The helper's question about previous customer support tickets")
    documents: str = dspy.InputField(desc="Relevant ticket data retrieved from the support database")
    answer: str = dspy.OutputField(
        desc="Structured three-part answer: issue clarification / solutions with ticket citations / proposed user message"
    )


__all__ = ["TicketAnswerSignature", "TicketAnswerSignatureStreaming", "generate_answer"]
