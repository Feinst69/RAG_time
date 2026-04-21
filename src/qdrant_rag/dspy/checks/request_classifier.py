import dspy
from dotenv import load_dotenv
import litellm
import os
from typing import Literal
import logging        

class Response_Type_Classifier_Signature(dspy.Signature):
    """
    Sur la base du contenu de la requête et l'historique de conversation,
    Classify wether the user is asking for a scientific rapport, a summary, a synthesis rapport or an opinion in output task.
    """
    query: str = dspy.InputField(desc="Requête de l'utilisateur")
    conversation_history: dspy.History = dspy.InputField(desc="Historique de conversation, pour contexte")
    task: str = dspy.InputField(desc="Affirmation à valider par True ou False")   
    response_type: bool = dspy.OutputField(desc="Résultat de la classification binaire : True ou False")
    justification: str = dspy.OutputField(desc="Explication brève du choix en minuscules")

class Response_Type_Classifier(dspy.Module):
    logger = logging.getLogger("terres_inovia")

    def __init__(self):
        super().__init__()
        self.agent : dspy.Prediction = dspy.asyncify(
                        dspy.Predict(
                            Response_Type_Classifier_Signature,
                        )
        )

    async def forward(self, task: str, query: str, context: dspy.History) -> dspy.Prediction :
        self.logger.debug("Response_Type_Classifier: classifying the query... \n%s",task)
        output : dspy.Prediction = await self.agent(
            query=query,
            conversation_history = context,
            task=task
        )
        self.logger.debug(f"Response_Type_Classifier Output: {output}\n")
        return output

async def classify_type(query: str, context: dspy.History, task: str) -> bool:
    """
    Classe une query ypour orienter la suite du traitement.
    Args:
        query (str): La requête de l'utilisateur à qualifier.
        context (dspy.History): L'historique des échanges précédents pour contexte.
        task (str): La tâche de classification binaire à effectuer.
    Returns:
        dspy.Prediction: La prédiction contenant la valeur booléenne de retour et la justification.
    """
    result = await Response_Type_Classifier()(task=task,query=query, context=context)
    return result.response_type
    

# load_dotenv()
# api_key = os.getenv("OPENAI_API_KEY")
# print(f"Using OpenRouter API key: {api_key[:6]}...")  # Print only the first 8 characters for security
# if not api_key:
#         raise RuntimeError("Missing RAG_OPENAI_API_KEY in the environment.")

# class Classify_response_type(dspy.Signature):
#     """Classify wether the user is asking for a scientific rapport, a summary, a synthesis rapport or an opinion."""
#     sentence: str = dspy.InputField()
#     classification: Literal["scientific_rapport", "summary", "synthesis_rapport", "opinion"] = dspy.OutputField()
#     confidence: float = dspy.OutputField()

__all__ = ["Response_Type_Classifier_Signature", "Response_Type_Classifier", "classify_type"]

