# Lib Imports
import dspy
from dotenv import load_dotenv
import litellm
import os
from typing import Literal

load_dotenv()
# Dspy Imports

from rag_time.dspy.checks.request_classifier import Classify_response_type


class DSPyPipeline(user_request: str):
    """A DSPy pipeline that classifies the user request and then routes it to the appropriate module."""

    # Step 1: Guardrails checks
    from rag_time.dspy.checks.guardrails import Guardrails
    from .rules import Rules

    guardrails_result = dspy.Predict(Guardrails)(
        user_request=user_request,
        retrieved_documents=["Les lentilles sont des légumes très nutritifs dont la particularité est de contenir beaucoup de fibres et de protéines."],  # This should be replaced with the actual retrieved documents
        generated_answer="",  # This should be replaced with the actual generated answer
    )

    # Step 3: Rules
    
    rules_result = dspy.Predict(Rules)(user_request=user_request)

    # Step 1: Classify the user request
    rag_answer_format = dspy.Predict(Classify_response_type)(sentence=user_request)

    # Step 3: Route the request to the appropriate module based on the classification
    


    from ..answer_builder.rag_summary import RagAnswer
    answer = dspy.Predict(RagAnswer)(classification_result=rag_answer_format.classification)


    



__all__ = ["DSPyPipeline"]
