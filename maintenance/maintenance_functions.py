import sys
from pathlib import Path
# Add the project root to the sys.path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import json
from config.config_loader import config

from document_store.milvus2_preference_store import Milvus2PreferenceStore
from langchain.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.chains.base import Chain
from langchain_core.prompts import BasePromptTemplate, ChatPromptTemplate
from langchain.prompts import MessagesPlaceholder
from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from maintenance.maintenance_function_calling import Append, Pass, Update
from maintenance.incoming_preference import IncomingPreference
from utils.llm import get_llm_gpt4o
from pydantic import BaseModel

from utils.custom_logger import log_info, log_warning

MAINTENANCE_PROMPT_MP = ChatPromptTemplate.from_messages([
    ("system", "You are a client to maintain a database storing user preferences. Your task is to keep the storage up-to-date by performing a database function based on the incoming preference and existing preferences. You must call a tool. There are multiple preferences per category allowed, however not with the same attribute or very similar ones. Examples: 1. (incoming: vegetarian, existing: vegetarian --> results in 'pass_preference'); 2. (incoming: vegetarian, existing: kosher --> results in 'append_preference'); 3. (incoming: vegetarian, existing: no vegetarian --> results in 'update_preference')."),
    ("human", "Existing Preferences: {existing_preferences} ### Incoming (NEW) Preference: {incoming_preference}"),
    ("human", "First output your thought process (category check, text check, attribute check, multiple preferences within category allowed check), then call one function.")
])

MAINTENANCE_PROMPT_MP_RETRY = ChatPromptTemplate.from_messages([
    ("system", "Your last run did not call a tool, make sure to call a tool. You are a client to maintain a database storing user preferences. Your task is to keep the storage up-to-date by performing a database function based on the incoming preference and existing preferences. You must call a tool. There are multiple preferences per category allowed, however not with the same attribute or very similar ones. Examples: 1. (incoming: vegetarian, existing: vegetarian --> results in 'pass_preference'); 2. (incoming: vegetarian, existing: kosher --> results in 'append_preference'); 3. (incoming: vegetarian, existing: no vegetarian --> results in 'update_preference')."),
    ("human", "Existing Preferences: {existing_preferences} ### Incoming (NEW) Preference: {incoming_preference}"),
    ("human", "First output your thought process (category check, text check, attribute check, multiple preferences within category allowed check), then call one function.")
])

MAINTENANCE_PROMPT_MNP = ChatPromptTemplate.from_messages([
    ("system", "You are a client to maintain a database storing user preferences. Your task is to keep the storage up-to-date by performing a database function based on the incoming preference and existing preferences. You must call a tool. There can always only be stored one preference."),
    ("human", "Existing Preferences: {existing_preferences} ### Incoming (NEW) Preference: {incoming_preference}"),
    ("human", "First output your thought process (category check, text check, attribute check, multiple preferences within category allowed check), then call one function.")
])

MAINTENANCE_PROMPT_MNP_RETRY = ChatPromptTemplate.from_messages([
    ("system", "Your last run did not call a tool, make sure to call a tool. You are a client to maintain a database storing user preferences. Your task is to keep the storage up-to-date by performing a database function based on the incoming preference and existing preferences. You must call a tool. There can always only be stored one preference."),
    ("human", "Existing Preferences: {existing_preferences} ### Incoming Preference: {incoming_preference}"),
    ("human", "First output your thought process (category check, text check, attribute check, multiple preferences within category allowed check), then call one function.")
])

class Maintenace():
    """
    class which implements functions performed by preference storage maintenance
    """
    
    def __init__(self):
        self.maintenance_functions_a_p_u: list[BaseTool] = [
            Append(),
            Pass(),
            Update()
        ]
        self.maintenance_functions_p_u: list[BaseTool] = [
            Pass(),
            Update()
        ]
        self.functions_a_p_u = [convert_to_openai_function(t) for t in self.maintenance_functions_a_p_u]
        self.functions_p_u = [convert_to_openai_function(t) for t in self.maintenance_functions_p_u]
        self.maintenance_chain_mp = (
            MAINTENANCE_PROMPT_MP
            | get_llm_gpt4o(temperature=0.0).bind_tools(self.functions_a_p_u)
        )
        self.maintenance_chain_retry_mp = (
            MAINTENANCE_PROMPT_MP_RETRY
            | get_llm_gpt4o(temperature=0.0).bind_tools(self.functions_a_p_u)
        )
        self.maintenance_chain_mnp = (
            MAINTENANCE_PROMPT_MNP
            | get_llm_gpt4o(temperature=0.0).bind_tools(self.functions_p_u)
        )
        self.maintenance_chain_retry_mnp = (
            MAINTENANCE_PROMPT_MNP_RETRY
            | get_llm_gpt4o(temperature=0.0).bind_tools(self.functions_p_u)
        )

    def run_tool(self, tool_name, args, perform_function=True):
        if tool_name=='append_preference':
            append_function = Append()
            run_tool_answer, tool_call = append_function._run(self.incoming_preference, perform_function=perform_function)
        
        elif tool_name=='pass_preference':
            pass_function = Pass()
            pk_of_equal_existing_preference = args['pk_of_equal_existing_preference']
            matching_dict = next((item for item in self.exisiting_preferences if item.get('pk') == pk_of_equal_existing_preference), None)
            if matching_dict:
                run_tool_answer, tool_call = pass_function._run(
                    self.incoming_preference,
                    pk_of_equal_existing_preference=pk_of_equal_existing_preference,
                    equal_existing_preference=matching_dict,
                    perform_function=perform_function
                    )
            else:
                log_warning(f"pk generated by llm does not match existing preferences key")
                run_tool_answer, tool_call = pass_function._run(self.incoming_preference, perform_function=False)
        
        elif tool_name=='update_preference':
            update_function = Update()
            pk_to_delete_existing_preference = args['pk_of_to_delete_existing_preference']
            matching_dict = next((item for item in self.exisiting_preferences if item.get('pk') == pk_to_delete_existing_preference), None)
            if matching_dict:
                run_tool_answer, tool_call =update_function._run(
                incoming_preference=self.incoming_preference, 
                pk_to_delete_existing_preference=args['pk_of_to_delete_existing_preference'],
                to_delete_existing_preference=matching_dict,
                perform_function=perform_function
                )
            else:
                log_warning(f"pk generated by llm does not match existing preferences key")
                pass_function = Pass()
                run_tool_answer, tool_call = pass_function._run(self.incoming_preference, perform_function=False)

        return run_tool_answer, tool_call

    def filter_extracted_preference_mp(self, incoming_preference, existing_preferences, perform_function=True):
        """
        performs one of the functions pass, update, append based on similar stored preferences and the new incoming preference.
        the new incoming preference should be from a mp (multiple possible) category
        params:
        incoming_preference: the currently extracted preference which should be inserted to the storage
        existing_preferences: the existing preferences within the detail_category.
        perform_function: if the function should actually be performed in the database or only simulated
        """
        self.incoming_preference = incoming_preference
        self.exisiting_preferences = existing_preferences
        keys_to_include_incoming_preference = ['detail_category', 'text', 'attribute']
        incoming_preference_prompt_input = {key: incoming_preference[key] for key in keys_to_include_incoming_preference if key in incoming_preference}
        
        existing_preferences_prompt_input = []
        keys_to_include_existing_preferences = ['pk', 'detail_category', 'text', 'attribute']
        for existing_preference in existing_preferences:
            existing_preference_prompt_input = {key: existing_preference[key] for key in keys_to_include_existing_preferences if key in existing_preference}        
            existing_preferences_prompt_input.append(existing_preference_prompt_input)
        try:
            result = self.maintenance_chain_mp.invoke(input={"incoming_preference": incoming_preference_prompt_input, "existing_preferences": existing_preference_prompt_input})
            tool_call = result.additional_kwargs['tool_calls'][0]
            tool_name = tool_call['function']['name']
            tool_args = json.loads(tool_call['function']['arguments'])
        except KeyError:
            try:
                log_info(f"In the first call no tool was called. Retrying now.")
                result = self.maintenance_chain_retry_mp.invoke(input={"incoming_preference": incoming_preference_prompt_input, "existing_preferences": existing_preference_prompt_input})
                tool_call = result.additional_kwargs['tool_calls'][0]
                tool_name = tool_call['function']['name']
                tool_args = json.loads(tool_call['function']['arguments'])
            except KeyError:
                log_info(f"The second call did not call a tool again. Passing the preference as default.")
                tool_name='pass_preference'
                tool_args = None
            
        run_tool_answer, tool_call = self.run_tool(tool_name=tool_name, args=tool_args, perform_function=perform_function)
        print(run_tool_answer)
        log_info(f"{run_tool_answer}")
        return run_tool_answer, tool_call

    def filter_extracted_preference_mnp(self, incoming_preference, existing_preferences, perform_function=True):
        """
        performs one of the functions pass, update, or (if category empty) append based on similar stored preferences and the new incoming preference.
        the new incoming preference should be from a mnp (multiple not possible) category
        params:
        incoming_preference: the currently extracted preference which should be inserted to the storage
        existing_preferences: the existing preferences within the detail_category.
        perform_function: if the function should actually be performed in the database or only simulated
        """
        self.incoming_preference = incoming_preference
        self.exisiting_preferences = existing_preferences
        keys_to_include_incoming_preference = ['detail_category', 'text', 'attribute']
        incoming_preference_prompt_input = {key: incoming_preference[key] for key in keys_to_include_incoming_preference if key in incoming_preference}
        
        existing_preferences_prompt_input = []
        keys_to_include_existing_preferences = ['pk', 'detail_category', 'text', 'attribute']
        for existing_preference in existing_preferences:
            existing_preference_prompt_input = {key: existing_preference[key] for key in keys_to_include_existing_preferences if key in existing_preference}        
            existing_preferences_prompt_input.append(existing_preference_prompt_input)
        try:
            result = self.maintenance_chain_mnp.invoke(input={"incoming_preference": incoming_preference_prompt_input, "existing_preferences": existing_preference_prompt_input})
            tool_call = result.additional_kwargs['tool_calls'][0]
            tool_name = tool_call['function']['name']
            tool_args = json.loads(tool_call['function']['arguments'])
        except KeyError:
            try:
                log_info(f"In the first call no tool was called. Retrying now.")
                result = self.maintenance_chain_retry_mnp.invoke(input={"incoming_preference": incoming_preference_prompt_input, "existing_preferences": existing_preference_prompt_input})
                tool_call = result.additional_kwargs['tool_calls'][0]
                tool_name = tool_call['function']['name']
                tool_args = json.loads(tool_call['function']['arguments'])
            except KeyError:
                log_info(f"The second call did not call a tool again. Passing the preference as default.")
                tool_name='pass_preference'
                tool_args = None
            
        run_tool_answer, tool_call = self.run_tool(tool_name=tool_name, args=tool_args, perform_function=perform_function)
        print(run_tool_answer)
        log_info(f"{run_tool_answer}")

        return run_tool_answer, tool_call