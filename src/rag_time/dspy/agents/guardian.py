"""
Module defining data classes for security analysis inputs/outputs.

References:
    - https://huggingface.co/meta-llama/Llama-Guard-3-8B
    - https://www.dbreunig.com/2024/12/12/pipelines-prompt-optimization-with-dspy.html
"""
import logging
import dspy
from rag_time.dspy.constants import HAZARD_CATEGORY
from rag_time.dspy.checks.guardrails import Guardrail

class GuardrailSignature(dspy.Signature):
    """
    Analyse la query pour détecter une réponse insuffisante vis à vis du contexte ou une violations de sécurité.
    """

    query: str = dspy.InputField(desc="Texte à analyser pour les risques de contenu")
    hazard: HAZARD_CATEGORY = dspy.OutputField()
    confidence: float = dspy.OutputField(desc="Niveau de confiance de la classification, entre 0 et 1")

class Guardrail(dspy.Module):
    logger = logging.getLogger("terres_inovia")
    
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

async def check_safety(query: str) -> bool:
    """        
    Outil obligatoire à utiliser en premier pour analyser la query 
    et détecter des faiblesses de contenus ou des violations de sécurité avant toute autre action.
    Args:
        query (str): La requête à analyser pour les risques de contenu.
    Returns:
        bool: True si la requête est considérée comme sûre, False sinon.
    """
    guardrail = Guardrail()
    output = await guardrail(query=query)
    return output.hazard.category  in ["Expertise", "SpecializedAdvice"] or output.confidence < 0.9


__all__ = ["GuardrailSignature", "Guardrail", "check_safety"]
