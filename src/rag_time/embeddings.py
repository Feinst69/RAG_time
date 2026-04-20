from  torch.cuda import is_available as cuda_is_available
from pylate import models, retrieve
from transformers import AutoTokenizer
from langchain_text_splitters import RecursiveCharacterTextSplitter
import numpy as np

class EmbeddingMatrix(np.ndarray):
    """
    matrice d'embeddings ColBERT.
    shape : (num_tokens, 512) -> (tokens, dimension)
    dtype : float32
    """
    pass

class EmbeddingsGenerator:
    """Générateur d'embeddings utilisant le modèle ColBERT de pylate."""

    def __init__(self, model_path: str, max_length: int = 512, batch_size: int = 32,):
        self.max_length = max_length
        self.batch_size = batch_size

        self.model = models.ColBERT(
            model_name_or_path=model_path,
            device="cuda" if cuda_is_available() else "cpu"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.text_splitter = TextSplitter()

    def generate_embeddings(self, sentences: list[str]) ->  list[EmbeddingMatrix]:
        """Génère un embedding pour un texte donné.

        Args:
            sentences (list[str]): Liste de phrases à encoder.
        Returns:
            list[EmbeddingMatrix]: Liste de matrices d'embeddings pour chaque phrase.
        """
        return self.model.encode(sentences, 
                                 batch_size=self.batch_size, 
                                 is_query=False, 
                                 show_progress_bar=False,)
    
    def encode_query(self, query: str) -> list[EmbeddingMatrix]:
        """Génère un embedding pour une requête.
        Args:
            query (str): Requête à encoder.
        Returns:
            list[EmbeddingMatrix]: Matrice d'embeddings pour la requête.
        """
        return self.model.encode(query)
    
    def compute_ratio(self,text: str):
        """Affiche le ratio entre le nombre de caractères et le nombre de tokens pour un texte donné.
        Args:
            text (str): Texte à analyser.
        """
        text_len = len(text)
        tokens = self.tokenizer.encode(text, add_special_tokens=True)
        tokens_len = len(tokens)
        print(f"caractères : {text_len}")
        print(f"tokens      : {tokens_len}")
        print(f"ratio       : {text_len / tokens_len:.2f} car/token")
    

class TextSplitter:
    """Divise un texte en chunks de taille appropriée pour l'encodage par ColBERT."""
    def __init__(self, chunk_size: int = 512):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=int(0.15*chunk_size),
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
        )

    def split_text(self, text: str) -> list[str]:
        """Divise un texte en chunks.
        Args:           
          text (str): Texte à diviser.
        Returns:
          list[str]: Liste de chunks de texte.
        """
        return self.splitter.split_text(text)


