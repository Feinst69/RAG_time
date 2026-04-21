import dspy
from dotenv import load_dotenv
import litellm
import os
from typing import Literal
import logging

from qdrant_rag.dspy.constants import HAZARD_CATEGORY



class GuardrailSignature(dspy.Signature):
    """Analyse la query pour détecter des faiblesses potentielles du contenu ou des violations de sécurité.
    """

    query: str = dspy.InputField(desc="Texte à analyser pour les risques de contenu")
    hazard: HAZARD_CATEGORY = dspy.OutputField()
    confidence: float = dspy.OutputField(desc="Niveau de confiance de la classification, entre 0 et 1")


class Guardrail(dspy.Module):
    logger = logging.getLogger("terres_inovia.guardrail")
    
    def __init__(self):
        super().__init__()
        self.predictor = dspy.asyncify(
            dspy.Predict(
                GuardrailSignature
            )
        )
 
    async def forward(self, query: str) -> dspy.Prediction:
        self.logger.debug("Guardrail: analyzing query for content risks...")
        output =  await self.predictor(query=query)
        self.logger.debug(f"Guardrail: hazard={output.hazard.category}, confidence={output.confidence}\n")
        return output
           
    



__all__ = ["GuardrailSignature", "Guardrail"]
