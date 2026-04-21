from pydantic import BaseModel, Field


class TicketResolution(BaseModel):
    issue_clarification: str = Field(
        desc=(
            "A concise summary of the identified issue based on the retrieved tickets "
            "and the user's query. Clarify what the root problem is in 2-4 sentences."
        )
    )
    proposed_solutions: str = Field(
        desc=(
            "A structured list of actionable solutions drawn from the retrieved tickets. "
            "Each solution must cite the relevant ticket(s) it comes from using the format "
            "[Ticket #<id>]. Include at least one citation per solution step."
        )
    )
    user_message: str = Field(
        desc=(
            "A ready-to-send message addressed to the user that summarises the issue and "
            "presents the proposed resolution steps. Use [Username] to refer to the user "
            "and [helper_name] to refer to the support agent who will follow up. "
            "Keep a professional, empathetic tone."
        )
    )


class RelevanceJudgement(BaseModel):
    relevance_score: float = Field(
        desc=(
            "A score between 0.0 and 1.0 representing how relevant the retrieved documents "
            "are to the query. 1.0 = perfectly relevant, 0.0 = completely irrelevant."
        )
    )
    reasoning: str = Field(
        desc="Brief explanation of why the retrieved documents are or are not relevant to the query."
    )


class AnswerQualityJudgement(BaseModel):
    answer_quality: float = Field(
        desc=(
            "A score between 0.0 and 1.0 evaluating the quality of the generated answer "
            "compared to the reference answer. Consider relevance, accuracy, and completeness. "
            "1.0 = perfect answer, 0.0 = completely wrong or irrelevant."
        )
    )
    retrieval_relevance: float = Field(
        desc=(
            "A score between 0.0 and 1.0 evaluating how relevant the retrieved documents "
            "were to the query. 1.0 = perfectly relevant, 0.0 = completely irrelevant."
        )
    )
    retrieval_usage: float = Field(
        desc=(
            "A score between 0.0 and 1.0 evaluating how effectively the retrieved documents "
            "were used in the generated answer. 1.0 = fully grounded in retrieved docs, "
            "0.0 = answer ignores the retrieved context entirely."
        )
    )
    reasoning: str = Field(
        desc="Brief explanation justifying each of the three scores above."
    )


__all__ = ["TicketResolution", "RelevanceJudgement", "AnswerQualityJudgement"]
