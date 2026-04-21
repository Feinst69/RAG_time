import logging
from typing import Tuple

import dspy

from rag_time.dspy.checks.request_classifier import Response_Type_Classifier
from rag_time.dspy.constants import (
    DEFAULT_RESPONSE_TEMPLATE,
    RESPONSE_CLASSIFICATION_TASKS,
    RESPONSE_TEMPLATES,
)


class RoutedAnswerSignature(dspy.Signature):
  """Génère la réponse finale en respectant un template imposé."""

  query: str = dspy.InputField(desc="Question reformulée de l'utilisateur")
  conversation_history: dspy.History = dspy.InputField(desc="Historique récent de la conversation pour contexte")
  documents: str = dspy.InputField(desc="Bloc de documents ou extraits fournis par le RAG")
  answer: str = dspy.OutputField(desc="Réponse finale respectant le template choisi")


class ResponseRouter(dspy.Module):
  logger = logging.getLogger("terres_inovia")

  def __init__(self) -> None:
    super().__init__()
    self.classifier = Response_Type_Classifier()
    self._writers: dict[str, dspy.Module] = {}

  async def _select_template(self, query: str, history: dspy.History) -> Tuple[str, str]:
    for template_key, task in RESPONSE_CLASSIFICATION_TASKS.items():
      result = await self.classifier(task=task, query=query, context=history)
      if result.response_type:
        return template_key, result.justification or "classification positive"
    return DEFAULT_RESPONSE_TEMPLATE, "fallback"

  def _get_writer(self, template_key: str) -> dspy.Module:
    if template_key not in self._writers:
      instructions = RESPONSE_TEMPLATES.get(template_key, RESPONSE_TEMPLATES[DEFAULT_RESPONSE_TEMPLATE])
      signature = RoutedAnswerSignature.with_instructions(instructions)
      self._writers[template_key] = dspy.asyncify(dspy.ChainOfThought(signature))
    return self._writers[template_key]

  async def forward(self, query: str, conversation_history: dspy.History) -> dspy.Prediction:
    template_key, justification = await self._select_template(query, conversation_history)
    writer = self._get_writer(template_key)
    prediction: dspy.Prediction = await writer(query=query, conversation_history=conversation_history)
    prediction.selected_template = template_key
    prediction.template_justification = justification
    self.logger.debug("ResponseRouter: template=%s justification=%s", template_key, justification)
    return prediction


__all__ = ["ResponseRouter", "RoutedAnswerSignature"]
