from pydantic import BaseModel, Field
from typing import  Optional

max_words = dict(
    tldr=20,
    introduction=100,
    presentation_intro=800,
    presentation_content=5000,
    presentation_conclusion=200)

for key,value in max_words.items():
    max_words[key] = f"en moins de {value} mots"

class PresentationWriter(BaseModel):
    tldr: Optional[str] = Field(desc=f"Un résumé très court de la réponse générée {max_words['tldr']}.")
    introduction: str = Field(desc=f"Section d'introduction qui reformule la demande de l'utilisateur {max_words['introduction']}.")
    presentation_intro: str = Field(desc=f"Section d'introduction de la présentation qui présente les éléments clés de la réponse de manière claire et engageante {max_words['presentation_intro']}.")
    presentation_content: str = Field(desc=f"Section de contenu de la présentation qui développe les éléments clés de la réponse de manière claire, structurée et engageante {max_words['presentation_content']}.")
    presentation_conclusion: str = Field(desc=f"Section de conclusion de la présentation qui résume les éléments clés de la réponse et répond à la demande de l'utilisateur de manière structurée et factuelle{max_words['presentation_conclusion']}")