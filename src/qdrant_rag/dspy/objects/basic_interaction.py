from pydantic import BaseModel, Field
from typing import  Optional

max_words = dict(
    simple_answer=150)

for key,value in max_words.items():
    max_words[key] = f"en moins de {value} mots"

class SimpleAnswer(BaseModel):
    simple_answer: str = Field(desc=f"Une réponse simple et concise à la demande de l'utilisateur {max_words['simple_answer']}.")