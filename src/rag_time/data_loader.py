"""Module pour charger et prétraiter les données de tickets."""

import json

import pandas as pd
import numpy as np
from typing import Any, Optional
from pathlib import Path
from pydantic import BaseModel, field_serializer


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
    subject: str
    body: str 
    answer: str
    type: str
    queue: str
    priority: str
    language: str
    chunks: Optional[list[str]] = []
    embeddings: Optional[list[Any]]=[]
    sparse_embeddings: Optional[list[Any]]=[]
    time: Optional[float] = 0.0

    @field_serializer('embeddings')
    def serialize_embeddings(self, embs: list[Any]):
        return [e.tolist() if isinstance(e, np.ndarray) else e for e in embs]

    @field_serializer('sparse_embeddings')
    def serialize_sparse_embeddings(self, embs: list[Any]):
        """Convertit les objets sparse (fastembed/numpy) en dictionnaires compatibles JSON."""
        if not embs:
            return []
        
        serialized = []
        for e in embs:
            # Cas 1 : C'est l'objet SparseEmbedding de fastembed
            if hasattr(e, "indices") and hasattr(e, "values"):
                indices = e.indices.tolist() if isinstance(e.indices, np.ndarray) else list(e.indices)
                values = e.values.tolist() if isinstance(e.values, np.ndarray) else list(e.values)
                serialized.append({"indices": indices, "values": values})
            
            # Cas 2 : C'est un tableau numpy direct (rare pour du sparse mais possible)
            elif isinstance(e, np.ndarray):
                serialized.append(e.tolist())
            
            # Cas 3 : C'est déjà un format compatible
            else:
                serialized.append(e)
        return serialized

def stream_tickets(file_path: str):
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            yield Ticket(**data)
