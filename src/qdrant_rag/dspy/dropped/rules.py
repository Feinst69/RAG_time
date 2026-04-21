import dspy
from dotenv import load_dotenv
import litellm
import os
from typing import Literal


class Rules(user_request :str, dspy.Signature):
    """
    A DSPy signature that defines the rules the llm must follow regarding the RAG answer generation.
    """
    user_request: str = dspy.InputField()
    relevance_rule: str = dspy.OutputField(description="A rule that ensures that the generated answer is relevant to the user request.")
    factuality_rule: str = dspy.OutputField(description="A rule that ensures that the generated answer is factual.")
    groundedness_rule: str = dspy.OutputField(description="A rule that ensures that the generated answer is grounded in the retrieved documents.")
    overall_quality_rule: str = dspy.OutputField(description="A rule that ensures the overall quality of the generated answer.")
    audience_rule: str = dspy.OutputField(description="A rule that ensures that the generated answer is adapted to the target technological institute scientific audience.")

__all__ = ["Rules"]
