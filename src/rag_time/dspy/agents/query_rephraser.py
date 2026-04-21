from __future__ import annotations

import logging
from textwrap import dedent

import dspy

from rag_time.dspy.config import ConfigError, DSPyConfig

LOGGER = logging.getLogger("terres_inovia")


def _load_agent_profile() -> dict[str, object]:
  try:
    cfg = DSPyConfig()
  except ConfigError as exc:  # pragma: no cover - configuration missing
    LOGGER.warning("DSPy config missing (%s); fallback profile used.", exc)
    return {
        "domain": "Ticketing support",
        "domain_knowledge": "",
        "model_id": "google/gemini-3-pro-preview",
        "temperature": 0.2,
    }

  data = cfg.as_dict()
  domain = cfg.domain
  knowledge = data.get("domain_knowledge") or ""
  agent = next((item for item in cfg.agents if item.get("name") == "query_rephraser"), {})
  model_id = agent.get("model_id") or cfg.fallback_model
  temperature = float(agent.get("temperature") or 0.2)
  return {
      "domain": domain,
      "domain_knowledge": knowledge,
      "model_id": model_id,
      "temperature": temperature,
  }

class QueryRephraserSignature(dspy.Signature):
  query: str = dspy.InputField(desc="Texte à reformuler pour améliorer la recherche documentaire.")
  history: dspy.History = dspy.InputField(desc="Historique de conversation pour contexte.")
  domain: str = dspy.InputField(desc="Domaine métier pour guider la reformulation.")
  domain_knowledge: str = dspy.InputField(desc="Connaissances métier à rappeler dans la reformulation.")
  rephrased_query: str = dspy.OutputField(desc="Question reformulée en une seule phrase concise.")

class QueryRephraser(dspy.Module):
  def __init__(self) -> None:
    super().__init__()
    profile = _load_agent_profile()
    self.domain: str = str(profile["domain"])
    self.domain_knowledge: str = str(profile["domain_knowledge"])

    instructions = dedent(
        f"""
        Tu es un expert du domaine "{self.domain}" chargé d'optimiser les requêtes
        pour le moteur documentaire Terres Innovia.
        - Analyse l'historique uniquement pour extraire les termes utiles.
        - Reformule EXCLUSIVEMENT la dernière question sous forme d'une seule phrase courte.
        - Ajoute des synonymes métiers issus du domaine ({self.domain_knowledge}) si cela
          améliore le rappel, sans jamais traduire ou modifier les acronymes, codes produit
          ou noms propres présents dans la question initiale.
        - Ne réponds pas à la question et ne poses pas de nouvelle question.
        - Retourne uniquement la requête reformulée, sans explication ni décor.
        """
    ).strip()

    signature = QueryRephraserSignature.with_instructions(instructions)
    self.predictor = dspy.asyncify(dspy.ChainOfThought(signature))

  async def forward(self, *, question: str, history: dspy.History) -> str:
    if isinstance(history, dspy.History):
      history_obj = history
    else:
      raw_messages = getattr(history, "messages", history) or []
      history_obj = dspy.History(messages=list(raw_messages))
    result = await self.predictor(
        query=question,
        history=history_obj,
        domain=self.domain,
        domain_knowledge=self.domain_knowledge,
    )
    rewritten = (result.rephrased_query or "").strip()
    LOGGER.debug("QueryRephraser output: %s", rewritten)
    return rewritten


__all__ = ["QueryRephraser", "QueryRephraserSignature"]
