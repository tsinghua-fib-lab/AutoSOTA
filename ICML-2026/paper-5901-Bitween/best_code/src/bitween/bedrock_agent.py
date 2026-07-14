from strands.agent import Agent
from strands.models.bedrock import BedrockModel

from bitween.agent import AgentResponse, BaseAgent
from bitween.config import Config
from bitween.miscs import TimeoutFunction, getLogger

config = Config()
log = getLogger(__name__, config.logger_level)


class BedrockAgent(BaseAgent):
    def __init__(
        self,
        model_id: str,
        region_name: str,
        enable_thinking: bool = False,
        max_thinking_tokens: int = 4096,
        max_tokens: int = 64000,
        # 64000: us.anthropic.claude-sonnet-4-20250514-v1:0
        # 32768: us.anthropic.claude-opus-4-20250514-v1:0
        # 32000: us.anthropic.claude-opus-4-1-20250805-v1:0
    ):
        self.name = "Bedrock Agent"

        self.model_id = model_id
        self.region_name = region_name

        if enable_thinking:
            assert max_thinking_tokens < max_tokens, (
                f"Provided `max_thinking_tokens` = {max_thinking_tokens} "
                f">= `max_tokens` = {max_tokens}: "
                "it should be `max_thinking_tokens` < `max_tokens`"
            )

        self.enable_thinking = enable_thinking
        self.max_thinking_tokens = max_thinking_tokens
        self.max_tokens = max_tokens

        log.info(f"Initializing {self.name}: {self.model_id} @ {self.region_name}")

    def query(
        self,
        prompt: str,
        tools: list = None,
        timeout_sec: float = 1800.0,
        **kwargs,
    ) -> AgentResponse:
        tools = tools or []

        model_config = {
            "model_id": self.model_id,
            "region_name": self.region_name,
            "max_tokens": self.max_tokens,
            **kwargs,
        }

        if self.enable_thinking:
            model_config.setdefault("additional_request_fields", {})
            model_config["additional_request_fields"]["thinking"] = {
                "type": "enabled",
                "budget_tokens": self.max_thinking_tokens,
            }

        model = BedrockModel(**model_config)

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
