"""Configuration du projet RAG."""

from pydantic_settings import BaseSettings
from pydantic import Field, computed_field
from typing import Optional


class Settings(BaseSettings):
    """Configuration globale du projet."""

    # OpenRouter API
    openrouter_api_key: Optional[str] = Field(None, env="OPENROUTER_API_KEY")
    openrouter_model: Optional[str] = Field(None, env="OPENROUTER_MODEL")
    openrouter_temperature: float = Field(0.0, env="OPENROUTER_TEMPERATURE", ge=0.0)
    openrouter_timeout: int = Field(30, env="OPENROUTER_TIMEOUT", ge=0)

    # Qdrant configuration
    qdrant_host: str = Field("localhost", env="QDRANT_HOST")
    collection_name: str = Field("support_tickets", env="COLLECTION_NAME")

    # Dataset
    dataset_path: Optional[str] = Field(None, env="DATASET_PATH")

    # Embedding configuration
    chunk_size: int = Field(512, env="CHUNK_SIZE", ge=1)
    embedding_model: str = Field("mixedbread-ai/mxbai-edge-colbert-v0-17m", env="EMBEDDING_MODEL")
    embedding_dimension: int = Field(128, env="EMBEDDING_DIMENSION")
    sparse_model: str = Field("Qdrant/bm25", env="SPARSE_MODEL")
    reranker_model: str = Field("mixedbread-ai/mxbai-edge-colbert-v0-32m", env="RERANKER_MODEL")
    
    # Search configuration
    top_k: int = Field(10, env="TOP_K", ge=1)
    hybrid_search_weight: float = Field(0.5, env="HYBRID_SEARCH_WEIGHT", ge=0, le=1)  # 0 = full text, 1 = vector only

    @computed_field
    def qdrant_url(self) -> str:
        return f"http://{self.qdrant_host}:6333"
    
    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
