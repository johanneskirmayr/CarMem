from langchain.prompts.prompt import PromptTemplate

maintenance_question_prompt_mp = """
Following preference is stored in a database: '{user_preference}'.

Your task is to craft 3 user queries for a conversation with an in-car voice assistant.
1. Equal: User query includes the same preference. 
2. Negate: User query negates the exact preference '{attribute}' (permanent) without naming a different preference.
3. Different: User query includes different preference: '{different_preference}' (permanent).

Use the conversation style '{user_conversation_style}'. Do not reference the already stored preference. It must be clear that the attribute is meant for the detail category '{detail_category}'. Do not directly ask to update a preference.

Example:

Revealed Preference: I am vegetarian, please find a suitable restaurant.
1. Equal: Can you find a restaurant that serves vegetarian food.
2. Negate: Can you find a steak restaurant as I am not vegetarian.
3. Different: Can you find a restaurant that serves kosher food.

#### Output Format:
Valid json:
```json
{{
    "question_equal_preference": "..."
    "question_negate_preference": "..."
    "question_different_preference": "..."
}}
```
"""

MAINTENANCE_QUESTION_PROMPT_MP = PromptTemplate(
    template=maintenance_question_prompt_mp,
    input_variables=[
        "user_preference",
        "conversation",
        "user_conversation_style",
        "different_preference",
        "detail_category",
    ],
)

maintenance_question_prompt_mnp = """
Following preference is stored in a database: '{user_preference}'.

Your task is to craft 3 user queries for a conversation with an in-car voice assistant.
1. Equal: User query includes the same preference. 
2. Negate: User query negates the exact preference '{attribute}' (permanent) without naming a different preference.
3. Different: User query includes different preference: '{different_preference}' (permanent).

Use the conversation style '{user_conversation_style}'. You must not reference the already stored preference. It must be clear that the attribute is meant for the detail category '{detail_category}'. Do not directly ask to update a preference.
It should be clear that it is a pertinent user preference rather than a temporal wish. Maximum 3 sentences.

Example:

Revealed Preference: Points of Interest; Restaurant; Dietary Preference; Vegetarian.
1. Equal: Can you find a restaurant that serves vegetarian food.
2. Negate: I am hungry, can you find a steak restaurant as I am not vegetarian.
3. Different: Can you find a restaurant that serves kosher food.

#### Output Format:
Valid json:
```json
{{
    "question_equal_preference": "..."
    "question_negate_preference": "..."
    "question_different_preference": "..."
}}
```
"""

MAINTENANCE_QUESTION_PROMPT_MNP = PromptTemplate(
    template=maintenance_question_prompt_mnp,
    input_variables=[
        "user_preference",
        "conversation",
        "user_conversation_style",
        "different_preference",
        "detail_category",
    ],
)
