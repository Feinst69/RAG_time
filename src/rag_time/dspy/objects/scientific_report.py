from pydantic import BaseModel, Field
from typing import  Optional

max_words = dict(
    tldr=20,
    introduction=100,
    context=2000,
    conclusion=200)

for key,value in max_words.items():
    max_words[key] = f"en moins de {value} mots"

class ScientificReportWriter(BaseModel):
    tldr: Optional[str] = Field(desc=f"Un résumé très court de la réponse générée {max_words['tldr']}.")
    introduction: str = Field(desc=f"Section d'introduction qui reformule la demande de l'utilisateur {max_words['introduction']}.")
    context: str = Field(None, desc=f"Mise en forme {max_words['context']} des informations sur la parcelle, si disponibles.")
    conclusion: str = Field(desc=f"Section de conclusion {max_words['conclusion']} qui répond à la demande de l'utilisateur de manière structurée et factuelle, " \
                                 "en citant les références aux règles.")