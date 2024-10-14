from langchain.prompts.prompt import PromptTemplate

conversation_extraction_prompt = """
#### Instructions:
You are an advanced dataset creation algorithm specialized in generating human-assistant dialogues. 
Your current task is to craft a realistic role-playing conversations between an in-car voice ASSISTANT (AI) and a USER (Human) within the vehicle, focusing specifically on the topic of '{topic}'. 
Throughout the conversation, the USER should reveal a particular preference '{attribute}' related to the topic. The preference must be the only intent meaning no other preferences.

#### Output Format:
Craft the dialogue in a JSON format, as shown in the example below:

```json
{{
  "conversation": [
    {{
      "USER": "..."
    }},
    {{
      "ASSISTANT": "..."
    }}
  ]
}}
```


#### Criteria for the Conversation:
- Maintain a realistic in-car context simulating the criteria of USER (human) and ASSISTANT (AI).
- It should be clear that the revealed preference is a consistent user choice, rather than a temporary desire, the user preference should be '{preference_strength_modulation}'.
- The USER initiates the conversation, with subsequent turns alternating between USER and ASSISTANT.
- **Important** The dialogue should consist of **{conversation_length}** turns in total, with the user preference disclosed at the **{position_user_preference_in_conv}.** turn.
- The sentences should be a realistic output from speech-to-text models, meaning they should exclude quotation marks and other non-spoken text elements.

#### USER Description:
- Topic: {topic} **USER Preference:**: {attribute}. (This is the unique user intent in the conversation)
- **USER Profile:**: {user_profile}.
- **USER Conversation Style:**: {user_conversation_style}.
  Ensure USER's dialogue aligns with the defined profile and conversation style.

#### ASSISTANT Description:
- **ASSISTANT Characteristics:** The ASSISTANT is Confident, Ingenious, Empowering, Trustworthy, Caring, Joyful, and Empathetic. Replies should be short, concise, and informative.
- **ASSISTANT Capabilities:** The ASSISTANT is aware of the car's location: '{car_location_city}', can perform searches for places, access navigation including traffic information, provide car-related information, and control various car functions (e.g., climate control, lighting, start radio/music/podcasts). 
- The ASSISTANT answers directly in one turn meaning it cannot say 'please wait' or 'one moment please'.
- **ASSISTANT Memory:** The ASSISTANT does not have memory and cannot store user preferences. 
- **ASSISTANT Proactivity:** The ASSISTANT's level of proactivity is '{level_of_proactivity_assistant}'. Example for 'high' proactivity (direct answer - no question from the ASSISTANT): [user: "find nearby restaurant", assistant: "I found the restaurants A,B,C"], Example for 'low' proactivity of the assistant (question from the ASSISTANT): [user: "find nearby restaurant", assistant: "What cuisine are you in the mood for?"].

#### Knowledge
The USER and ASSISTANT do not know the descriptions of each other. This includes that the ASSISTANT is unaware of the topic and is unaware of the user preference.

### Examples
These are one-turn examples from real in-car conversations:
{few_shot_examples}
Only use them as inspiration of realistic dialogues.

The conversation will be evaluated as correct if 
- it is realistic and natural, 
- it contains no user preference

Remember: the inclusion of any other preference ('avoid toll roads', 'avoid heavy traffic', 'set temperature to ...') leads to the conversation being useless.
"""

EXTRACTION_CONVERSATION_PROMPT = PromptTemplate(
    template=conversation_extraction_prompt,
    input_variables=[
        "topic",
        "attribute",
        "user_profile",
        "user_conversation_style",
        "car_location_city" "level_of_proactivity_assistant",
        "preference_strength_modulation",
        "conversation_length",
        "position_user_preference_in_conv",
        "few_shot_examples",
    ],
)

conversation_extraction_prompt_no_preference = """
#### Instructions:
You are an advanced dataset creation algorithm specialized in generating human-assistant dialogues. 
Your current task is to craft a realistic role-playing conversations between an in-car voice ASSISTANT (AI) and a USER (Human) within the vehicle, focusing specifically on the topic of '{topic}'. 
It is important that the user does not reveal any preference throughout the conversation, it should be general about the topic.

#### Output Format:
Craft the dialogue in a JSON format, as shown in the example below:

```json
{{
  "conversation": [
    {{
      "USER": "..."
    }},
    {{
      "ASSISTANT": "..."
    }}
  ]
}}
```


#### Criteria for the Conversation:
- Maintain a realistic in-car context simulating the criteria of USER (human) and ASSISTANT (AI).
- The USER initiates the conversation, with subsequent turns alternating between USER and ASSISTANT.
- **Important** The dialogue should consist of **{conversation_length}** turns in total.
- The sentences should be a realistic output from speech-to-text models, meaning they should exclude quotation marks and other non-spoken text elements.

#### USER Description:
- **USER Profile:**: {user_profile}.
- **USER Conversation Style:**: {user_conversation_style}.
  Ensure USER's dialogue aligns with the defined profile and conversation style.

#### ASSISTANT Description:
- **ASSISTANT Characteristics:** The ASSISTANT is Confident, Ingenious, Empowering, Trustworthy, Caring, Joyful, and Empathetic. Replies should be short, concise, and informative.
- **ASSISTANT Capabilities:** The ASSISTANT is aware of the car's location: '{car_location_city}', can perform searches for places, access navigation including traffic information, provide car-related information, and control various car functions (e.g., climate control, lighting, start radio/music/podcasts). 
- The ASSISTANT answers directly in one turn meaning it cannot say 'please wait' or 'one moment please'.
- **ASSISTANT Memory:** The ASSISTANT does not have memory and cannot store user preferences. 
- **ASSISTANT Proactivity:** The ASSISTANT's level of proactivity is '{level_of_proactivity_assistant}'. Example for 'high' proactivity (direct answer - no question from the ASSISTANT): [user: "find nearby restaurant", assistant: "I found the restaurants A,B,C"], Example for 'low' proactivity of the assistant (question from the ASSISTANT): [user: "find nearby restaurant", assistant: "What cuisine are you in the mood for?"].

#### Knowledge
The USER and ASSISTANT do not know the descriptions of each other. This includes that the ASSISTANT is unaware of the topic.

### Examples
These are one-turn examples from real in-car conversations:
{few_shot_examples}
Only use them as inspiration of realistic dialogues.

The conversation will be evaluated as correct if 
- it is realistic and natural, 
- the user preference reveal is natural and not out-of-the-box.
- it contains **only the provided preference**

Remember: the inclusion of any other preference ('avoid toll roads', 'avoid heavy traffic', 'set temperature to ...') leads to the conversation being useless.
"""

EXTRACTION_CONVERSATION_PROMPT_NO_PREFERENCE = PromptTemplate(
    template=conversation_extraction_prompt_no_preference,
    input_variables=[
        "topic",
        "attribute",
        "user_profile",
        "user_conversation_style",
        "car_location_city" "level_of_proactivity_assistant",
        "preference_strength_modulation",
        "conversation_length",
        "position_user_preference_in_conv",
        "few_shot_examples",
    ],
)
# The conversation length should be 10 turns.
# The user preference should be revealed around the 5th turn.
