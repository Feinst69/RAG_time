# Lib Imports
import dspy
from dotenv import load_dotenv
import litellm
import os
from typing import Literal

load_dotenv()
# Dspy Imports


class RagAnswer(dspy.Signature, classification_result: str):
    """A DSPy signature that answer the user request based on the retrieved documents.
    The generated answer must address the user request and be based on the retrieved documents.
    """




__all__ = ["RagAnswer"]