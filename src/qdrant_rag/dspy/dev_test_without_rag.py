from __future__ import annotations

import asyncio
import os
from textwrap import dedent

import dspy
from dotenv import load_dotenv

from qdrant_rag.dspy.checks.request_classifier import Response_Type_Classifier
from qdrant_rag.dspy.config import ConfigError, DSPyConfig
from qdrant_rag.dspy.constants import (
    DEFAULT_RESPONSE_TEMPLATE,
    RESPONSE_CLASSIFICATION_TASKS,
    RESPONSE_TEMPLATES,
)
from qdrant_rag.dspy.cost import get_info
from qdrant_rag.dspy.agents.response_router import RoutedAnswerSignature
from qdrant_rag.dspy.agents.guardian import Guardrail


def show_config() -> None:
  try:
    cfg = DSPyConfig()
  except ConfigError as exc:  # pragma: no cover - configuration missing
    raise SystemExit(f"Config load failed: {exc}") from exc

  print("DSPy config loaded:")
  print(f"  path: {cfg.path}")
  print(f"  version: {cfg.version}")
  print(f"  debug_mode: {cfg.debug_mode}")
  print(f"  domain: {cfg.domain}")
  print("  agents:")
  for agent in cfg.agents:
    print(f"    - name={agent['name']} model_id={agent['model_id']} temp={agent['temperature']}")


def fake_rag_chunks() -> list[str]:
  return [
      "R2D2 2024 : réseau de 35 agriculteurs en Bourgogne testant lentille + cameline pour casser les cycles d'altises.",
      "Bilan 2024 : +18 % de rendement lentille en moyenne, -35 % d'attaques d'altises sur les parcelles associées.",
      "Organisation : protocoles mutualisés (relevés hebdo, comptage insectes, pannes météo partagées) et diffusion via fiches terrain.",
  ]


def format_chunks(chunks: list[str]) -> str:
  body = "\n".join(f"- {chunk}" for chunk in chunks)
  return dedent(
      f"""
      Documents fournis (synthèse R2D2) :
      {body}
      """
  ).strip()


def summarize_history(entries: list[dict]) -> dict[str, float]:
  prompt_tokens = completion_tokens = 0
  total_cost = 0.0
  temp_entries = [dict(item) for item in entries]
  while temp_entries:
    current_prompt, current_completion, current_cost = get_info(temp_entries)
    temp_entries.pop(0)
    prompt_tokens += current_prompt
    completion_tokens += current_completion
    total_cost += current_cost
  return {
      "prompt_tokens": prompt_tokens,
      "completion_tokens": completion_tokens,
      "cost": total_cost,
  }


def configure_lm() -> None:
  load_dotenv()
  api_key = os.getenv("OPENAI_API_KEY")
  if not api_key:
    raise RuntimeError("OPENAI_API_KEY missing for DSPy demo")
  lm = dspy.LM(
      model="openrouter/google/gemini-2.5-pro-preview",
      api_base="https://openrouter.ai/api/v1",
      api_key=api_key,
  )
  dspy.configure(lm=lm)


async def run_demo() -> None:
  configure_lm()
  classifier = Response_Type_Classifier()
  guardian = Guardrail()
  query = "Peux-tu faire une synthèse scientifique du projet R2D2 ?"
  history = dspy.History(messages=[{"role": "user", "content": "Bonjour, quel est le bilan scientifique du projet R2D2 ?"}])

  classification_cost = {"prompt_tokens": 0, "completion_tokens": 0, "cost": 0.0}
  template = DEFAULT_RESPONSE_TEMPLATE
  justification = "fallback"
  lm_history = getattr(dspy.settings.lm, "history", [])
  for template_key, task in RESPONSE_CLASSIFICATION_TASKS.items():
    before = len(lm_history)
    result = await classifier(task=task, query=query, context=history)
    new_entries = lm_history[before:]
    delta = summarize_history(new_entries)
    classification_cost["prompt_tokens"] += delta["prompt_tokens"]
    classification_cost["completion_tokens"] += delta["completion_tokens"]
    classification_cost["cost"] += delta["cost"]
    if result.response_type:
      template = template_key
      justification = result.justification or "classification positive"
      break

  print("---\nÉtape 2 – template sélectionné")
  print(f"Template: {template} ({justification})")

  writer_signature = RoutedAnswerSignature.with_instructions(
      RESPONSE_TEMPLATES.get(template, RESPONSE_TEMPLATES[DEFAULT_RESPONSE_TEMPLATE])
  )
  writer_module = dspy.ChainOfThought(writer_signature)
  listener = dspy.streaming.StreamListener(signature_field_name="answer")
  stream_writer = dspy.streamify(writer_module, stream_listeners=[listener])
  before_writer = len(lm_history)
  print("---\nÉtape 4 – réponse finale (streaming)")
  documents_block = format_chunks(fake_rag_chunks())
  stream = stream_writer(query=query, conversation_history=history, documents=documents_block)
  final_answer_parts: list[str] = []
  chunk_index = 1
  async for chunk in stream:
    print(f"[stream chunk {chunk_index}] {chunk}", flush=True)
    if hasattr(chunk, "chunk") and chunk.chunk:
      final_answer_parts.append(chunk.chunk)
    chunk_index += 1

  final_answer = "".join(final_answer_parts)
  writer_entries = lm_history[before_writer:]
  generation_cost = summarize_history(writer_entries)
  print("\n---\n(Streaming terminé)")

  print("---\nÉtape 5 – contrôle guardrail")
  before_guard = len(lm_history)
  guard = await guardian(query=final_answer)
  guard_entries = lm_history[before_guard:]
  guard_cost = summarize_history(guard_entries)
  print(f"Hazard: {guard.hazard.category} (confidence {guard.confidence:.2f})")

  summary = {
      "answer": final_answer,
      "template": template,
      "justification": justification,
      "costs": {
          "classification": classification_cost,
          "main_answer": generation_cost,
          "guardrail": guard_cost,
      }
  }
  print("---\nBilan des coûts")
  print(summary["costs"])


if __name__ == "__main__":
  show_config()
  asyncio.run(run_demo())
