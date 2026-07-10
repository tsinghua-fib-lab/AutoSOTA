import copy
from typing import Dict, List

from discoverllm.core.generate import generate_and_process, generate_chat
from discoverllm.core.tasks.simulate_user_response import get_goal_status_as_str
from discoverllm.pipeline.base import LLMPipeline
from discoverllm.utils import format_chat_history


class AssistantSimulator(LLMPipeline):
    def __init__(
        self,
        initial_chat_history: List[Dict[str, str]],
        system_prompt: str | None = None,
        user_prompt: str | None = None,
        model_name: str = "gpt-5.2-2025-12-11",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        verbose: bool = False
    ):
        super().__init__(model_name, temperature, max_tokens, verbose)
        self.chat_history = copy.deepcopy(initial_chat_history)
        self.system_prompt = system_prompt
        if system_prompt is None or len(system_prompt) == 0:
            self.system_prompt = None
        self.user_prompt = user_prompt

    def generate_with_user_prompt(self):
        formatted_user_prompt = self.user_prompt.format(
            chat_history=format_chat_history(self.chat_history)
        )

        output, raw_output = generate_and_process(
            model_name=self.model_name,
            system_prompt=self.system_prompt,
            user_prompt=formatted_user_prompt,
            processing_function=lambda x: x.split("# Response")[1].strip(),
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            verbose=self.verbose
        )
        return output

    def __call__(self, user_response: str, criteria_objs=None):
        self.chat_history.append({"role": "user", "content": user_response})
        if criteria_objs is not None:
            try:
                state_str = get_goal_status_as_str(criteria_objs)
                if state_str:
                    sidebar = "[INTENT STATE] " + state_str + " [/INTENT STATE] "
                    self.chat_history[-1]["content"] = sidebar + self.chat_history[-1]["content"]
            except Exception:
                pass

        if self.user_prompt is not None:
            assistant_response = self.generate_with_user_prompt()
        else:
            assistant_response = generate_chat(
                model_name=self.model_name,
                system_prompt=self.system_prompt,
                messages=self.chat_history,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                verbose=self.verbose
            )
        self.chat_history.append({"role": "assistant", "content": assistant_response})

        return assistant_response

