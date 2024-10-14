import csv
import json
import random
import sys
from pathlib import Path

# Add the project root to the sys.path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from dataset.prompts_database.retrieval_utterance_prompt_v4 import (
    NEXT_CONVERSATION_QUESTIONS_PROMPT,
)

# Local Imports
from utils.llm import get_llm_gpt4o


class NextConversationQuestionChain:
    """
    Generates synthetic conversations from user preferences
    """

    def __init__(self):
        self.generator = NEXT_CONVERSATION_QUESTIONS_PROMPT | get_llm_gpt4o(
            temperature=0.7
        ).bind(response_format={"type": "json_object"})

    def generate_next_conversation_questions(
        self,
        topic,
        user_preference,
        conversation,
        user_conversation_style,
        random_seed=None,
    ):
        follow_up_questions = self.generator.invoke(
            {
                "topic": str(topic),
                "user_preference": str(user_preference),
                "conversation": str(conversation),
                "user_conversation_style": str(user_conversation_style),
            }
        )

        metadata = {
            "user_preference": str(user_preference),
            "conversation": str(conversation),
        }
        return follow_up_questions, metadata
