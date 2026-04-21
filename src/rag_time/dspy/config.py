from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


class ConfigError(RuntimeError):
  """Raised when the DSPy config file is missing or malformed."""


class DSPyConfig:
  """
  Minimal loader for DSPy settings.

  The configuration is stored as YAML (see ``config_template.yaml``) and lives next to this module.
  By default the loader reads ``config.yaml`` if present; otherwise it falls back to the template.
  """

  def __init__(self, file_path: str | Path | None = None) -> None:
    self.path = self._resolve_config_path(file_path)
    self._payload = self._load_yaml(self.path)

  @property
  def debug_mode(self) -> bool:
    return bool(self._payload.get("debug_mode", False))

  @property
  def version(self) -> str:
    return str(self._payload.get("version", "0.0.1"))
  
  @property
  def domain(self) -> str:
    return str(self._payload.get("domain", "Unknown"))
  
  # Would be nice to use it on failures, look at https://digitalrain.studio/posts/2025-07-08-dspy-openrouter-integration
  @property
  def fallback_model(self) -> str:
    return str(self._payload.get("fallback_model", "google/gemini-2.5-pro-preview"))

  @property
  def lm_model(self) -> str:
    """Global DSPy LM model ID (LiteLLM format, e.g. openrouter/...)."""
    return str(self._payload.get("lm_model", f"openrouter/{self.fallback_model}"))

  def as_dict(self) -> dict[str, Any]:
    return json.loads(json.dumps(self._payload))

  @property
  def agents(self) -> list[dict[str, Any]]:
    raw_agents = self._payload.get("agents") or []
    if not isinstance(raw_agents, list):
      return []
    sanitized: list[dict[str, Any]] = []
    for entry in raw_agents:
      if isinstance(entry, dict):
        sanitized.append(
            {
                "name": entry.get("name", "unknown"),
                "model_id": entry.get("model_id", ""),
                "temperature": entry.get("temperature", 0),
            }
        )
    return sanitized

  @staticmethod
  def _resolve_config_path(file_path: str | Path | None) -> Path:
    if file_path is not None:
      path = Path(file_path).expanduser().resolve()
      if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
      return path
    base_dir = Path(__file__).resolve().parent
    candidate = base_dir / "config.yaml"
    if not candidate.exists():
      candidate = base_dir / "config_template.yaml"
    if not candidate.exists():
      raise ConfigError(f"No config.yaml or config_template.yaml found in {base_dir}")
    return candidate

  @staticmethod
  def _load_yaml(path: Path) -> dict[str, Any]:
    try:
      with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    except yaml.YAMLError as exc:  # pragma: no cover - invalid yaml
      raise ConfigError(f"Invalid YAML in {path}: {exc}") from exc
    if not isinstance(data, dict):
      raise ConfigError(f"Config file must contain a mapping at top level (file: {path}).")
    return data


__all__ = ["ConfigError", "DSPyConfig"]
