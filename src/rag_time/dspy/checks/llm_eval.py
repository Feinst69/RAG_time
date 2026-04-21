from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable

import dspy
import litellm
import yaml
from dotenv import load_dotenv


class PipelineEvaluation:
  """
  Thin orchestrator that loads an evaluation config, spins up DSPy with the requested LLMs,
  and triggers the evaluation pipeline signature.

  The config file must follow the structure defined in ``evaluation_config.yaml`` (see repo root)
  and at least contain the ``llm_eval`` section with a default ``model_name``. The class can also
  receive an explicit list of model identifiers to evaluate.
  """

  def __init__(
      self,
      config_path: str,
      *,
      models: Iterable[str] | None = None,
      pipeline_signature: type[dspy.Signature] | None = None
  ) -> None:
    load_dotenv()
    self.config_path = Path(config_path).resolve()
    if not self.config_path.exists():
      raise FileNotFoundError(f"Evaluation config not found: {self.config_path}")
    self.config: dict[str, Any] = self._load_config()
    self.llm_config = self.config.get("llm_eval", {})
    default_model = self.llm_config.get("model_name")
    if models:
      self.models = list(models)
    elif default_model:
      self.models = [default_model]
    else:
      raise ValueError("Provide at least one model via the config or the 'models' parameter.")

    self.pipeline_signature = pipeline_signature or self._import_pipeline_signature()

  def evaluate(self) -> dict[str, Any]:
    """
    Run the evaluation pipeline for each requested model.

    Returns a dictionary keyed by model name containing the pipeline outputs.
    """
    results: dict[str, Any] = {}
    for model in self.models:
      predictor = self._prepare_predictor(model)
      results[model] = predictor(config_path=str(self.config_path))
    return results

  def _prepare_predictor(self, model_name: str):
    api_key = self._resolve_api_key()
    litellm.openrouter_key = api_key
    lm_kwargs = self._llm_kwargs(model_name, api_key)
    lm = dspy.LM(**lm_kwargs)
    dspy.configure(lm=lm)
    return dspy.Predict(self.pipeline_signature)

  def _llm_kwargs(self, model_name: str, api_key: str) -> dict[str, Any]:
    temp_cfg = self.llm_config.get("temperature", {})
    temperature = temp_cfg.get("initial")
    if temperature is None:
      temperature = 0.2
    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": os.getenv("OPENROUTER_SITE_URL", "https://lab-ia.fr"),
        "X-Title": os.getenv("OPENROUTER_APP_TITLE", "Terra Innovia Eval")
    }
    return {
        "model": model_name,
        "api_base": self.llm_config.get("api_base", "https://openrouter.ai/api/v1"),
        "api_key": api_key,
        "temperature": temperature,
        "headers": headers,
    }

  def _resolve_api_key(self) -> str:
    token = (
        os.getenv("OPENROUTER_API_KEY")
        or os.getenv("OR_API_KEY")
        or os.getenv("OPENAI_API_KEY")
    )
    if not token:
      raise RuntimeError("Set OPENROUTER_API_KEY (or OR_API_KEY) before running the DSPy evaluation.")
    return token

  def _load_config(self) -> dict[str, Any]:
    with self.config_path.open("r", encoding="utf-8") as handle:
      return yaml.safe_load(handle) or {}

  def _import_pipeline_signature(self) -> type[dspy.Signature]:
    try:
      from ..dropped.pipeline import DSPyPipeline
    except ImportError as exc:  # pragma: no cover - developer misconfiguration
      raise RuntimeError("Missing DSPy pipeline signature (expected .pipeline.DSPyPipeline).") from exc
    if not issubclass(DSPyPipeline, dspy.Signature):
      raise TypeError("DSPyPipeline must inherit from dspy.Signature.")
    return DSPyPipeline


__all__ = ["PipelineEvaluation"]
