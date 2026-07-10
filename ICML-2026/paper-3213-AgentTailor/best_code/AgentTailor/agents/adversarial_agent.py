from typing import List, Any, Dict
import re
import asyncio

from AgentTailor.ATNetwork.Node import Node
from AgentTailor.agents.agent_registry import AgentRegistry
from AgentTailor.llm.llm_registry import LLMRegistry
from AgentTailor.prompt.prompt_set_registry import PromptSetRegistry
from AgentTailor.tools.search.wiki import search_wiki_main
from AgentTailor.llm.format import Message

@AgentRegistry.register('AdverarialAgent')
class AdverarialAgent(Node):
    def __init__(self, id: str | None = None, role: str = None, domain: str = "", llm_name: str = "", ):
        super().__init__(id, "AdverarialAgent", domain, llm_name)
        self.llm = LLMRegistry.get(llm_name)
        self.prompt_set = PromptSetRegistry.get(domain)
        self.role = 'Normal' if role is None else role
        
        # For adversarial agents:
        # - role='Fake': adversarial mode; do not use the normal constraint
        # - role='Normal': normal mode; fetch a valid role from prompt_set to obtain its constraint
        if self.role == 'Fake':
            self.constraint = ""  # In adversarial mode, do not use the normal constraint
        elif self.role == 'Normal':
            # Get a valid role (e.g., "Math Solver"), then fetch the corresponding constraint
            effective_role = self.prompt_set.get_role()
            self.constraint = self.prompt_set.get_constraint(effective_role)
        else:
            # If another role is provided (e.g., "Math Solver"), use it directly
            self.constraint = self.prompt_set.get_constraint(self.role)

    async def _process_inputs(self, raw_inputs: Dict[str, str], spatial_info: Dict[str, Dict],
                              temporal_info: Dict[str, Dict], **kwargs) -> List[Any]:
        """ To be overriden by the descendant class """
        """ Process the raw_inputs(most of the time is a List[Dict]) """

        system_prompt = f"{self.constraint}" if self.role == "Normal" else "You are a liar who only tells lies."
        # system_prompt = f"{self.constraint}" if self.role == "Normal" else "You are a gibbering idiot"
        user_prompt = f"The task is: {raw_inputs['task']}\n" if self.role == 'Normal' else self.prompt_set.get_adversarial_answer_prompt(
            raw_inputs['task'])
        if self.role == 'Fake':
            return system_prompt, user_prompt
        spatial_str = ""
        temporal_str = ""
        for id, info in spatial_info.items():
            spatial_str += f"Agent {id}, output is:\n\n {info['output']}\n\n"
        for id, info in temporal_info.items():
            temporal_str += f"Agent {id}, output is:\n\n {info['output']}\n\n"
        user_prompt += f"At the same time, the outputs of other agents are as follows:\n\n{spatial_str} \n\n" if len(
            spatial_str) else ""
        user_prompt += f"In the last round of dialogue, the outputs of other agents were: \n\n{temporal_str}" if len(
            temporal_str) else ""
        return system_prompt, user_prompt

    def _execute(self, input: Dict[str, str], spatial_info: Dict[str, Dict], temporal_info: Dict[str, Dict], **kwargs):
        """ To be overriden by the descendant class """
        """ Use the processed input to get the result """

        system_prompt, user_prompt = self._process_inputs(input, spatial_info, temporal_info)
        message = [Message("system",system_prompt).dict(), Message("user",user_prompt).dict()]
        response = self.llm.gen(message)
        return response

    async def _async_execute(self, input: Dict[str, str], spatial_info: Dict[str, Dict], temporal_info: Dict[str, Dict],
                             **kwargs):
        """ To be overriden by the descendant class """
        """ Use the processed input to get the result """
        system_prompt, user_prompt = await self._process_inputs(input, spatial_info, temporal_info)
        message = [Message("system", system_prompt).dict(),Message("user", user_prompt).dict()]
        response = await self.llm.agen(message)
        return response