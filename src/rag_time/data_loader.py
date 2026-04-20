"""Module pour charger et prétraiter les données de tickets."""

import pandas as pd
import numpy as np
from typing import Optional
from pathlib import Path
from pydantic import BaseModel, ConfigDict, field_serializer

from rag_time.embeddings import EmbeddingMatrix


class DataLoader:
    """Charge et prétraitement les données de tickets de support."""

    def __init__(self, dataset_path: str = None):
        """
        Initialiser le chargeur de données.

        Args:
            dataset_path: Chemin vers le fichier CSV du dataset
        """
        self.dataset_path = dataset_path

    def load(self, limit: int = None) -> pd.DataFrame:
        """
        Charge les données depuis un fichier CSV.

        Args:
            path: Chemin du fichier (surcharge le chemin par défaut)
            limit: Nombre maximum de lignes à charger

        Returns:
            DataFrame contenant les tickets
        """

        if not self.dataset_path:
            raise ValueError("Aucun chemin de dataset spécifié")

        if not Path(self.dataset_path).exists():
            raise FileNotFoundError(f"Fichier non trouvé: {self.dataset_path}")

        columns = ["subject", "body", "answer", "type", "queue", "priority", "language"]

        if limit:
            df = pd.read_csv(self.dataset_path, usecols=columns, nrows=limit)
        else:
            df = pd.read_csv(self.dataset_path, usecols=columns)
        
        return self._preprocess(df)

    def _preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prétraitement des données.

        Args:
            df: DataFrame brut

        Returns:
            DataFrame nettoyé et standardisé
        """
        result_data = df.dropna(subset=["body", "answer"])\
                        .reset_index(drop=True)
        
        result_data.fillna("", inplace=True)

        return result_data

class Ticket(BaseModel):
    """
    Représente un ticket de support avec ses métadonnées et embeddings.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    subject: str
    body: str 
    answer: str
    type: str
    queue: str
    priority: str
    language: str
    chunks: Optional[list[str]] = []
    embeddings: Optional[list[EmbeddingMatrix]]=[]
    time: Optional[float] = 0.0

    @field_serializer('embeddings')
    def serialize_embeddings(self, embs: list[EmbeddingMatrix]):
        return [e.tolist() if isinstance(e, np.ndarray) else e for e in embs]
    
