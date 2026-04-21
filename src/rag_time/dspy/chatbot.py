import asyncio
from time import time, gmtime, strftime
from typing import Optional
import dspy
import logging

from rag_time.dspy.agents.guardian import check_safety
from rag_time.dspy.agents.writer import writer

from rag_time.dspy.cost import get_info


class Chatbot():
    logger: logging.Logger = logging.getLogger("terres_inovia")
    
    def __init__(self):    
        self.chat_history = dspy.History(messages=[])
        self.stop_event = asyncio.Event()

    @property
    def context(self):
        return dspy.History(
            messages = self.chat_history.messages[self.start_index:] # demander de détails a Xavier
            #messages = self.chat_history.messages[:-5]
            )

    async def pass_checks(self, query: str, task: str) -> bool:  
        return await binary_classify(query=query, 
                                     context=self.context, 
                                     task=task)

    
    
    async def workflow(self, query: str) -> str:
        self.logger.debug("Workflow démarré")
        do_retrieve = False

        with dspy.context(lm=self.lm):

            # 1. Check safety
            safe =  await check_safety(query)
            if not safe: 
                return ("Désolé, je ne peux pas traiter votre requête.")
            
            # 2. Qualify query for routing

            # 2.1. Courtesy doesn't need much resources
            task = ("La requête de l'utilisateur est une formule de politesse "
                    "un commentaire, un compliment, un remerciement.")
            if await self.pass_checks(query, task):
                return await self.simple_answer(query)

            # 2.1. Off-topic query
            task = ("La requête de l'utilisateur est en rapport " 
                    "avec le domaine.")  
            if not await self.pass_checks(query, task):
                return ("Votre requête semble hors sujet par rapport au domaine de l'urbanisme. \n"
                        "Pouvez-vous reformuler votre question ?")
            

            # 5.Final answer
            return "final answer placeholder"
            
            return await final_answer
        
    def stop_workflow(self):
        self.stop_event.set()
        self.logger.debug("Signal d'arrêt envoyé.")
        
    async def start_workflow(self, query: str) -> tuple[str, list, str]:
        try:
            duration = time()
            response = await self.workflow(query)
            duration = time() - duration
            duration = strftime("%M:%S", gmtime(duration))
            costs = self.get_costs()
        except AttributeError as e:
            self.logger.error(f"Error in workflow: {e}")
            response = "Désolé, une erreur est survenue lors du traitement de votre requête. Veuillez réessayer."
        self.chat_history.messages.append({"query": query, "answer": response})

        return response, costs, duration
    
    def get_costs(self) -> list:
        costs = [0,0,0.0]
        while len(self.lm.history) > 0:
            current_costs = get_info(self.lm.history)
            self.lm.history.pop(0)
            for i in range(3):
                costs[i] += current_costs[i]
        return costs
    


