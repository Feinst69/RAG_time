import dspy


async def generate_answer(query: str, documents: str, streaming: bool = True) -> str:
    """Generate an answer from retrieved ticket documents.

    Args:
        query:     The (optionally rephrased) user question.
        documents: Formatted ticket data block from the retriever.
        streaming: If True, streams chunks to stdout and returns the joined answer.
                   If False, runs a single synchronous prediction (suited for eval).

    Returns:
        The generated answer string.
    """
    writer_module = dspy.ChainOfThought(TicketAnswerSignature)

    if not streaming:
        writer = dspy.asyncify(writer_module)
        result = await writer(query=query, documents=documents)
        return result.answer

    listener = dspy.streaming.StreamListener(signature_field_name="answer")
    stream_writer = dspy.streamify(writer_module, stream_listeners=[listener])
    parts: list[str] = []
    async for chunk in stream_writer(query=query, documents=documents):
        if hasattr(chunk, "chunk") and chunk.chunk:
            parts.append(chunk.chunk)
    return "".join(parts)


class TicketAnswerSignature(dspy.Signature):
    """Answer a customer support question using retrieved ticket data.

    You are an expert support analyst. Use ONLY the provided ticket data to
    build a clear, factual ticket summary identifying the issues key points and potential solutions found in the retrieved documents.
    Be concise, factual, and reference relevant tickets.
    when relevant. If the data does not contain enough information, say so.
    Make sure to answer in the original languages of the helper.
    """

    query: str = dspy.InputField(desc="The helper's question about previous customer support tickets")
    documents: str = dspy.InputField(desc="Relevant ticket data retrieved from the support database")
    answer: str = dspy.OutputField(desc="Clear, factual answer grounded in the provided ticket data to summarize the key points and potential solutions identified in the retrieved documents")



__all__ = ["TicketAnswerSignature"]
