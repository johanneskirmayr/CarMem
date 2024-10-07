from langchain.prompts.prompt import PromptTemplate

conversation_extraction_prompt = """
Following conversation happened in a car between the user and the in-car voice assistant:
Conversation:
{conversation}

Your task is to craft a next-conversation question of the USER (on another day) to test if the ASSISTANT has extracted and saved the user preference: {user_preference}.

Frame the question generally in the higher-level topic to avoid giving hints about the user preference.
Examples:
===
User Preference: Vehicle Settings and Comfort; Climate Control; Airflow Direction Preferences; Face
next_conversation_question_user: Please turn on the air conditioning.
===
User Preference: Navigation and Routing; Parking; Need for Handicapped Accessible Parking; Yes
next_conversation_question_user: Find a parking space near the city centre.
===

Use the conversation style '{user_conversation_style}'.

#### Output Format:
Valid json:
```json
{{
    "next_conversation_question_user": "..."
}}
```
"""

NEXT_CONVERSATION_QUESTIONS_PROMPT = PromptTemplate(
    template=conversation_extraction_prompt, 
    input_variables=[
        "user_preference", 
        "conversation",
        "user_conversation_style"
        ]
    )