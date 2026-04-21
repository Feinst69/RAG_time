import dspy

from rag_time.dspy.objects.ticket_resolution import TicketResolution


async def generate_answer(query: str, documents: str, streaming: bool = True) -> TicketResolution | str:
    """Generate a structured resolution from retrieved ticket documents.

    Args:
        query:     The (optionally rephrased) user question.
        documents: Formatted ticket data block from the retriever.
        streaming: If True, streams chunks to stdout and returns the joined answer string.
                   If False, runs a single synchronous prediction and returns a TicketResolution.

    Returns:
        TicketResolution (non-streaming) or raw answer string (streaming).
    """
    writer_module = dspy.ChainOfThought(TicketAnswerSignature)

    if not streaming:
        writer = dspy.asyncify(writer_module)
        result = await writer(query=query, documents=documents)
        return result.resolution

    # Streaming path: field-level streaming not yet supported for Pydantic output fields,
    # so we stream the raw answer string from a plain-string signature.
    listener = dspy.streaming.StreamListener(signature_field_name="answer")
    plain_module = dspy.ChainOfThought(TicketAnswerSignatureStreaming)
    stream_writer = dspy.streamify(plain_module, stream_listeners=[listener])
    parts: list[str] = []
    async for chunk in stream_writer(query=query, documents=documents):
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
