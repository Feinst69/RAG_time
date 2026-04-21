import dspy
import logging
import re
from typing import  Optional,TYPE_CHECKING

from qdrant_rag.dspy.objects import basic_interpretation, presentation, scientific_report

class ReportWriterSignature(dspy.Signature):
    """
    S'assurer de comprendre la demande de l'utilisateur dans le contexte fournit.
    Rédiger un rapport structuré pour répondre à la demande de l'utilisateur en 
    s'appuyant uniquement sur les dispositions générales et les règles applicables à la zone
    fournies. Tout autre source d'information est strictement interdite.

    Par défaut, ce sont les dispositions générales qui s'appliquent.
    Si des règles spécifiques à la zone contredisent les dispositions générales, elles supplantent ces dernières.
    Si des règles spécifiques à la zone complètent les dispositions générales, elles s'ajoutent à ces dernières.    
    
    Le rapport doit être factuel et citer les références aux règles.

    IMPORTANT : 
    - ne jamais ajouter de titre au début des sections du rapport structuré.
    - ne jamais mentionner les noms de champs techniques (ex: inond, eboul) ni les valeurs 
    brutes du json (ex: = a, = None). Traduire chaque information en langage naturel.
    """

    query: str = dspy.InputField(desc="La demande de l'utilisateur")
    context: dspy.History = dspy.InputField(desc="L'historique de conversation")   


class ReportWriter(dspy.Module):
    logger = logging.getLogger("terres_inovia")

    def __init__(self):
        super().__init__()
        self.agent = dspy.asyncify(
                        dspy.ChainOfThought(
                            ReportWriterSignature
                        )
                    )

    async def forward(self, query: str, 
                        context: dspy.History, 
                        dg: str, rules: str, 
                        plot_info: PlotInformation) -> str:
        if dg == "" and rules == "":
            self.logger.info("Writer : no doc to write ...")
            return ""
        self.logger.info("Writer : writing doc...")
        outputs : dspy.Prediction = await self.agent(
            query=query,
            context=context,
            dg=dg,
            rules=rules,
            plot_info=plot_info)
        
        report = outputs.report
        self.logger.debug(f"Writer : introduction   ->  {count_words(report.introduction)} mots")
        self.logger.debug(f"Writer : context        ->  {count_words(report.context) if report.context else 0} mots")
        self.logger.debug(f"Writer : dg_analysis    ->  {count_words(report.dg_analysis)} mots")
        self.logger.debug(f"Writer : rules_analysis ->  {count_words(report.rules_analysis) if report.rules_analysis else 0} mots")
        self.logger.debug(f"Writer : conclusion     ->  {count_words(report.conclusion)} mots")
        output="### Introduction\n"
        output += f"{report.introduction}\n\n" 
        if report.context:
            output += "### Contexte\n"    
            output += f"{report.context}\n\n"
        output += "### Dispositions générales\n"
        output += f"{report.dg_analysis}\n\n"
        if report.rules_analysis:
            output += "### Règles spécifiques\n"
            output += f"{report.rules_analysis}\n\n"
        output += "### Conclusion\n"
        output += f"{report.conclusion}\n\n"
        return output


class QueryRewriterSignature(dspy.Signature):
    """
    Reformuler requête de l'utilisateur pour qu'elle soit 
    compréhensible de manière autonome en t'appuyant sur l'historique de conversation.
    La reformulation doit être écrite du point de vue de l'utilisateur.
    Si l'utilisateur a déjà fourni des informations précises dans une requête précédente,
    tu les réutilises dans la reformulation. Si la requête fournit une correction partielle
    d'information, tu reconstruis l'information complète en te basant sur l'historique.
    """

    query :str = dspy.InputField(desc="la requête brute à reformuler")
    context : dspy.History = dspy.InputField(desc="historique de la conversation")
    new_query :str = dspy.OutputField(desc="la requête réécrite de façon claire et complète")

class QueryRewriter(dspy.Module):
    logger = logging.getLogger("terres_inovia")

    def __init__(self):
        super().__init__()
        self.agent = dspy.asyncify(
                dspy.ChainOfThought(QueryRewriterSignature)
            )

    async def forward(self, context, query):
        output = await self.agent(context=context, query=query)
        self.logger.debug(f"Rewriter : new query -> {output.new_query}")
        return output.new_query


class SimpleWriterSignature(dspy.Signature):
    """
    Répondre à la requête de l'utilisateur en se basant uniquement sur les informations-
    fournies dans l'historique de conversation et les informations sur la parcelle.
    
    IMPORTANT :
    ne jamais mentionner les noms de champs techniques (ex: inond, eboul) ni les valeurs 
    brutes du json (ex: = a, = None). Traduire chaque information en langage naturel."
    """

    query :str = dspy.InputField(desc="la demande de l'utilisateur")
    context : dspy.History = dspy.InputField(desc="historique de la conversation")
    plot_info : Optional[PlotInformation] = dspy.InputField(desc="Les informations sur la parcelle, si disponibles")
    answer :str = dspy.OutputField(desc="la réponse")

class SimpleWriter(dspy.Module):
    logger = logging.getLogger("terres_inovia")

    def __init__(self, persona: str):
        super().__init__()
        self.agent = dspy.asyncify(
                dspy.ChainOfThought(SimpleWriterSignature.with_instructions(persona))
            )


    async def forward(self, context, query, plot_info=None):
        output = await self.agent(context=context, query=query, plot_info=plot_info)
        self.logger.debug(f"SimpleWriter : answer -> {output.answer}")
        return output.answer


async def rewrite_query(context, query):
    return await QueryRewriter()(context=context, query=query)


async def write_structured_report(query: str,
                                  context: dspy.History,
                                  zone_info: ZoneInformation, 
                                  plot_info: PlotInformation) -> str|tuple[str, str, str]:
    """
    Cet outil ne doit être utilisé que :
     - si la query ne concerne pas une réponse donnée précédemment
     - s'il y a nécessité de consulter les règles de PLUi
    Args:
        query (str): La demande de l'utilisateur.
        dg (str): Les dispositions générales à appliquer.
        rules (str): Les règles applicables à la zone.
        zone_info (ZoneInformation): Les informations sur la zone cadastrale.
        plot_info (PlotInformation): Les informations sur la parcelle, si disponibles.
    Returns:
        str: Le rapport structuré en markdown.
    """
    plui_rules: RagResult = await get_plui_rules(query=query, zone_info=zone_info)
    dg = plui_rules.dg_rules
    rules = plui_rules.zone_rules
    return await  ReportWriter()(query=query, context=context, dg=dg, rules=rules, plot_info=plot_info), dg, rules


def write_simple_answer(query: str, context: dspy.History, persona, 
                        plot_info: PlotInformation = None):
    return SimpleWriter(persona)(context=context, query=query, plot_info=plot_info)


def count_words(text: str) -> int:
    return len(re.findall(r'\w+', text))

