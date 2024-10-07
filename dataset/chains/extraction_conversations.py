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
from prompts_database.extraction_conversation_prompts_dynamic_v3 import EXTRACTION_CONVERSATION_PROMPT, EXTRACTION_CONVERSATION_PROMPT_NO_PREFERENCE


def sample_random_city(random_seed = None):
    random.seed(random_seed)
    random_row = random.randint(0, 44692-1)
    with open('data/worldcities/worldcities.csv', 'r') as file:
        reader = csv.reader(file)
        for i, row in enumerate(reader):
            if i == random_row:
                return row[1] + ", " + row[4]
            
def sample_few_shot_example(category, random_seed):
    random.seed(random_seed)
    with open('data/real_conversations/real.conversations.json', 'r') as file:
        real_conversations = json.load(file)
        return random.choice(real_conversations[category])

class ExtractionConversationChain():
    """
    Generates synthetic conversations from user preferences
    """
    def __init__(self):
        self.generator = EXTRACTION_CONVERSATION_PROMPT | get_llm_gpt4o(temperature=0.7).bind(response_format={"type": "json_object"})
        self.generator_no_preference = EXTRACTION_CONVERSATION_PROMPT_NO_PREFERENCE | get_llm_gpt4o(temperature=0.7).bind(response_format={"type": "json_object"})

    def sample_dynamic_prompt_inputs(self, random_seed):
        random.seed(random_seed) # below dynamic pieces should be random but fixed within a user
        self.user_profile = {
            "Age": "Placeholder", 
            "Technological_Proficiency": ["low", "middle", "high"],
            }
        
        self.user_profile["Age"] = random.randrange(start=20, stop=90, step=10)
        self.user_profile["Technological_Proficiency"] = random.choice(self.user_profile["Technological_Proficiency"])
        
        user_conversation_style = [
            "Keyword only: direct, to-the-point.",
            "Commanding: straightforward, imperative sentences.", 
            "Questioning: seeking information, clarification.", 
            "Conversational: casual, human-like manner."
            ]
        self.user_conversation_style = random.choice(user_conversation_style)

        self.car_location_city = sample_random_city(random_seed)

        # random.seed(None) # below dynamic pieces should be sampled for every new conversation
        level_of_assistant_proactivity = [
            "medium",
            "high",
            "very high - no questions"
        ]
        self.level_of_proactivity_assistant = random.choice(level_of_assistant_proactivity)

        preference_strength_modulation = [
            "subtly hinted at",
            "clearly stated",
            "strongly emphasized"
        ]
        self.preference_strength_modulation = random.choice(preference_strength_modulation)

        self.conversation_length = random.randrange(start=2, stop=10, step=2) # can only be even
        # self.conversation_length = 4

        self.position_user_preference_in_conv = random.randrange(start=1, stop=self.conversation_length, step=2) # can only be odd
        
        return

    def generate_one_conversation(self, topic: str, user_preference: str, random_seed = None):
        self.sample_dynamic_prompt_inputs(random_seed=random_seed)
        self.few_shot_example = sample_few_shot_example(category=topic.split('; ')[-1], random_seed=random_seed)
        user_preference_topic = user_preference.split("; ")[:-1]
        attribute = user_preference.split("; ")[-1]

        # if random_seed==0:
        #     self.user_conversation_style = "Keyword only: direct, to-the-point."
        # if random_seed==1:
        #     self.user_conversation_style = "Commanding: straightforward, imperative sentences." 
        # if random_seed==2:
        #     self.user_conversation_style = "Questioning: seeking information, clarification."
        # if random_seed==4:
        #     self.user_conversation_style = "Conversational: casual, human-like manner."

        conversation = self.generator.invoke({
            "topic": str(user_preference_topic), 
            "attribute": str(attribute),
            "user_preference": str(user_preference),
            "user_profile": str(self.user_profile),
            "user_conversation_style": str(self.user_conversation_style),
            "car_location_city": str(self.car_location_city),
            "level_of_proactivity_assistant": str(self.level_of_proactivity_assistant),
            "preference_strength_modulation": str(self.preference_strength_modulation),
            "conversation_length": str(self.conversation_length),
            "position_user_preference_in_conv": str(self.position_user_preference_in_conv),
            "few_shot_examples": str(self.few_shot_example)
            })
        
        metadata = {
            "topic": str(topic), 
            "user_preference": str(user_preference),
            "user_profile": str(self.user_profile),
            "user_conversation_style": str(self.user_conversation_style),
            "car_location_city": str(self.car_location_city),
            "level_of_proactivity_assistant": str(self.level_of_proactivity_assistant),
            "preference_strength_modulation": str(self.preference_strength_modulation),
            "conversation_length": str(self.conversation_length),
            "position_user_preference_in_conv": str(self.position_user_preference_in_conv),
            "few_shot_examples": str(self.few_shot_example)
        }
        return conversation, metadata
    
    def generate_one_conversation_no_preference(self, topic: str, user_preference: str, random_seed = None, user_profile=None, user_conversation_style=None, car_location_city=None, level_of_proactivity_assistant=None, preference_strength_modulation=None, conversation_length=None, position_user_preference_in_conv=None, few_shot_example=None):
        self.sample_dynamic_prompt_inputs(random_seed=random_seed)
        self.few_shot_example = sample_few_shot_example(category=topic.split('; ')[-1], random_seed=random_seed)
        user_preference_topic = user_preference.split("; ")[:-2]
        attribute = user_preference.split("; ")[-1]
        user_profile = user_profile or self.user_profile
        user_conversation_style = user_conversation_style or self.user_conversation_style
        car_location_city = car_location_city or self.car_location_city
        level_of_proactivity_assistant = level_of_proactivity_assistant or self.level_of_proactivity_assistant
        preference_strength_modulation = preference_strength_modulation or self.preference_strength_modulation
        conversation_length = conversation_length or self.conversation_length
        position_user_preference_in_conv = position_user_preference_in_conv or self.position_user_preference_in_conv
        few_shot_example = few_shot_example or self.few_shot_example
        conversation = self.generator_no_preference.invoke({
            "topic": str(user_preference_topic), 
            "attribute": str(attribute),
            "user_preference": str(user_preference),
            "user_profile": str(user_profile),
            "user_conversation_style": str(user_conversation_style),
            "car_location_city": str(car_location_city),
            "level_of_proactivity_assistant": str(level_of_proactivity_assistant),
            "preference_strength_modulation": str(preference_strength_modulation),
            "conversation_length": str(conversation_length),
            "position_user_preference_in_conv": str(position_user_preference_in_conv),
            "few_shot_examples": str(few_shot_example)
            })
        
        metadata = {
            "topic": str(topic), 
            "user_preference": str(user_preference),
            "user_profile": str(user_profile),
            "user_conversation_style": str(user_conversation_style),
            "car_location_city": str(car_location_city),
            "level_of_proactivity_assistant": str(level_of_proactivity_assistant),
            "preference_strength_modulation": str(preference_strength_modulation),
            "conversation_length": str(conversation_length),
            "position_user_preference_in_conv": str(position_user_preference_in_conv),
            "few_shot_examples": str(few_shot_example)
        }
        return conversation, metadata