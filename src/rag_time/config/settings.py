"""Configuration du projet RAG."""

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional


class Settings(BaseSettings):
    """Configuration globale du projet."""

    # OpenRouter API
    openrouter_api_key: str = Field(..., env="OPENROUTER_API_KEY")

    # OpenSearch
    opensearch_url: str = Field("http://localhost:9200", env="OPENSEARCH_URL")

    # Dataset
    dataset_path: Optional[str] = Field(None, env="DATASET_PATH")

    # Index configuration
    index_name: str = "support_tickets"

    # Embedding configuration
    embedding_model: str = "nomic-ai/nomic-embed-text-v1.5"
    embedding_dimension: int = 768

    # LLM configuration
    llm_model: str = "meta-llama/llama-3-8b-instruct"

    # Search configuration
    top_k: int = 10
    hybrid_search_weight: float = 0.5  # 0 = full text, 1 = vector only

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
