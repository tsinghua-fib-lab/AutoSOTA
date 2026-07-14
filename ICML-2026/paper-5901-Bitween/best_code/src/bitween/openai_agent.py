from strands.agent import Agent
from strands.models.openai import OpenAIModel

from bitween.agent import AgentResponse, BaseAgent
from bitween.config import Config
from bitween.miscs import TimeoutFunction, getLogger

config = Config()
log = getLogger(__name__, config.logger_level)


class OpenAIAgent(BaseAgent):
    def __init__(
        self,
        model_id: str,
        base_url: str,
        api_key: str,
        max_tokens: int,
    ):
        self.name = "OpenAI Agent"

        self.model_id = model_id
        self.base_url = base_url
        self.api_key = api_key
        self.max_tokens = max_tokens

        log.info(f"Initializing {self.name}: {self.model_id} @ {self.base_url}")

    def query(
        self,
        prompt: str,
        tools: list = None,
        timeout_sec: float = 1800.0,
        **kwargs,
    ) -> AgentResponse:
        tools = tools or []

        model_config = {
            "base_url": self.base_url,
            "api_key": self.api_key,
            **kwargs,
        }

        model = OpenAIModel(
            client_args=model_config,
            model_id=self.model_id,
            params={"max_completion_tokens": self.max_tokens},
        )

        agent = Agent(
            model=model,
            system_prompt=self._system_prompt,
            tools=tools,
            callback_handler=None,
        )

        def query_agent(prompt):
            return agent(prompt)

        try:
            agent_result, error_msg = TimeoutFunction.call_for(
                timeout_sec=timeout_sec,
                func=query_agent,
                args=(prompt,),
            )

            if error_msg:
                return AgentResponse.create_empty_response(
                    content=error_msg,
                    agent_name=self.name,
                )
            else:
                return AgentResponse(
                    agent_result=agent_result,
                    session_messages=agent.messages,
                    agent_name=self.name,
                )

        except Exception as e:
            log.error(f"Exception during {self.name} query: {e}")
            return AgentResponse.create_empty_response(
                content=str(e),
                agent_name=self.name,
            )
