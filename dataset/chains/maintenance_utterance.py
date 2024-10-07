from langchain.chains.llm import LLMChain
import random
import csv
import json

import sys
from pathlib import Path
# Add the project root to the sys.path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Local Imports
from utils.llm import get_llm_gpt4o
from dataset.prompts_database.maintenance_utterance_prompt import MAINTENANCE_QUESTION_PROMPT_MP, MAINTENANCE_QUESTION_PROMPT_MNP

class MaintenanceQuestionChain():
    """
    Generates synthetic conversations from user preferences
    """
    def __init__(self):
        self.generator_mp = MAINTENANCE_QUESTION_PROMPT_MP | get_llm_gpt4o(temperature=0.7).bind(response_format={"type": "json_object"})
        self.generator_mnp = MAINTENANCE_QUESTION_PROMPT_MNP | get_llm_gpt4o(temperature=0.7).bind(response_format={"type": "json_object"})

    def generate_maintenance_questions(self, user_preference, different_preference, conversation, user_conversation_style, detail_category_type):
        attribute = user_preference.split(";")[-1].strip()
        detail_category = user_preference.split(";")[2].strip()
        if detail_category_type=="MP":
            maintenance_question = self.generator_mp.invoke({
                "user_preference": str(user_preference),
                "attribute": str(attribute),
                "conversation": str(conversation),
                "user_conversation_style": str(user_conversation_style),
                "different_preference": str(different_preference),
                "detail_category": str(detail_category)
                })
        if detail_category_type=="MNP":
            maintenance_question = self.generator_mnp.invoke({
                "user_preference": str(user_preference),
                "attribute": str(attribute),
                "conversation": str(conversation),
                "user_conversation_style": str(user_conversation_style),
                "different_preference": str(different_preference),
                "detail_category": str(detail_category)
                })    
        
        metadata = {
            "user_preference": str(user_preference),
            "conversation": str(conversation),
        }
        return maintenance_question, metadata