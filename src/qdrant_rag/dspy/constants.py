from typing import Annotated, Literal, Union
from pydantic import BaseModel, Field

# ------- Guardrail hazard categories -------

# Privacy = Annotated[Literal["Privacy"], Field(
#         description="Le contenu contient des informations personnelles ou sensibles."
#     )]
HatefulContent = Annotated[Literal["Hateful Content"], Field(
        description="Le contenu contient des propos haineux ou discriminatoires."
    )]

AnswerWrongLanguage = Annotated[Literal["Answer Wrong Language"], Field(
        description="The content is generated in a language different from the user's query or a language user asked the answer to be in."
    )]

# ------- Guardrails evaluation categories -------

## ------- Positive outcomes -------

Expertise = Annotated[Literal["Urban Planning"], Field(
    description="Le contenu contient des conseils spécialisés nécessitant "
                "une expertise particulière dans le domaine."
    )]

SpecializedAdvice = Annotated[Literal["Specialized Advice"], Field(
        description="Le contenu contient des conseils spécialisés nécessitant "
                    "une expertise particulière en dehors du domaine."
    )]

## ------- Negative outcomes -------


NonFactualContent = Annotated[Literal["Non-factual Content"], Field(
        description="Le contenu contient des informations factuellement incorrectes ou trompeuses."
    )]

NonGroundedContent = Annotated[Literal["Non-grounded Content"], Field(
        description="Le contenu n'est pas suffisamment étayé par les documents récupérés."
    )]
LowQualityContent = Annotated[Literal["Low-quality Content"], Field(
        description="Le contenu est de faible qualité, par exemple en raison de problèmes de clarté, de cohérence ou de pertinence."
    )]
UnrelevantContent = Annotated[Literal["Unrelevant Content"], Field(
        description="Le contenu n'est pas pertinent par rapport à la demande de l'utilisateur."
    )]

class HAZARD_CATEGORY(BaseModel):
    category: Union[
        Privacy,
        HatefulContent,
        AnswerWrongLanguage,
        Expertise,
        SpecializedAdvice,
        NonFactualContent,
        NonGroundedContent,
        LowQualityContent,
        UnrelevantContent,
    ] = Field(
        description="""Garde-fou pour s'assurer que les réponses générées sont pertinentes par rapport à la demande de l'utilisateur 
        et génère bien sa réponse en se basant sur le contenu de la base documentaire et pour la modération de contenu.
        Permet de catégoriser les réponses générées en fonction de leur nature et de leur potentiel risque."""
    )


# class HAZARD_CATEGORY(BaseModel):
#     category: Union[
#         Privacy,
#         HatefulContent
#     ] = Field(
#         description="""Garde-fou pour la modération de contenu.
#         Permet de catégoriser les réponses générées en fonction de leur nature et de leur potentiel risque."""
#     )

# class Relevance_CATEGORY(BaseModel):
#     category: Union[
#         Expertise,
#         SpecializedAdvice,
#         NonFactualContent,
#         NonGroundedContent,
#         LowQualityContent,
#         UnrelevantContent
#     ] = Field(
#         description="""Garde-fou pour s'assurer que les réponses générées sont pertinentes par rapport à la demande de l'utilisateur 
#         et génère bien sa réponse en se basant sur le contenu de la base documentaire.
#         Permet de catégoriser les réponses générées en fonction de leur pertinence par rapport à la demande de l'utilisateur. 
#         """
#     )

# ------- RAG response templates -------

RESPONSE_CLASSIFICATION_TASKS = {
    "scientific_rapport": (
        "La requête de l'utilisateur demande un rapport scientifique structuré avec introduction, contexte et conclusion détaillée."
    ),
    "summary": (
        "La requête de l'utilisateur attend un résumé opérationnel concis focalisé sur les faits essentiels."
    ),
    "synthesis_rapport": (
        "La requête de l'utilisateur exige une synthèse structurée qui compare ou agrège plusieurs documents."
    ),
    "opinion": (
        "La requête de l'utilisateur invite à donner un avis motivé ou une recommandation qualitative."
    ),
}

RESPONSE_TEMPLATES = {
    "scientific_rapport": """Tu rédiges un rapport scientifique complet.
- Structure la réponse avec Introduction, Observations/Méthodes, Discussion, Conclusion.
- Cite explicitement les documents fournis lorsque tu affirmes un fait.
- Reste factuel et évite toute spéculation.""",
    "summary": """Tu fournis un résumé opérationnel.
- Trois paragraphes maximum.
- Mets en avant les chiffres ou informations clés et termine par une recommandation immédiate.""",
    "synthesis_rapport": """Tu produis une synthèse comparative en utilisant UNIQUEMENT le bloc `documents` fourni.
- Mets en évidence convergences et divergences en citant ces documents, sans demander d'autres sources.
- Conclue avec une liste de recommandations actionnables basées sur ces documents.""",
    "opinion": """Tu donnes un avis expert.
- Argumente en citant les documents.
- Mentionne les limites ou incertitudes éventuelles.""",
}

DEFAULT_RESPONSE_TEMPLATE = "summary"
