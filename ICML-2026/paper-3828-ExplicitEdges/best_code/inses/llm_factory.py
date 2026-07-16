from enum import Enum
from typing import Dict, Any
from llama_index.llms.openai import OpenAI
from llama_index.llms.zhipuai import ZhipuAI
from llama_index.llms.deepseek import DeepSeek
from llama_index.core.llms import ChatMessage



class LLMProvider(Enum):
    OPENAI = "openai"
    ZHIPUAI = "zhipuai"
    DEEPSEEK = "deepseek"
    # Add other providers as needed


class LLMFactory:
    """LLM factory class that returns different LLM instances based on input parameters"""

    @staticmethod
    def create_llm(
            provider: str,
            model: str,
            api_key: str,
            **kwargs
    ) -> Any:
        """
        Create and return an LLM instance

        Args:
            provider: LLM provider name (openai, zhipuai, etc.)
            model: Model name
            api_key: API key
            **kwargs: Additional LLM parameters (temperature, max_tokens, etc.)

        Returns:
            Corresponding LLM instance
        """
        provider = provider.lower()

        if provider == LLMProvider.OPENAI.value:
            return OpenAI(
                model=model,
                api_key=api_key,
                **kwargs
            )
        elif provider == LLMProvider.ZHIPUAI.value:
            return ZhipuAI(
                model=model,
                api_key=api_key,
                **kwargs
            )
        elif provider == LLMProvider.DEEPSEEK.value:
            return DeepSeek(
                model=model,
                api_key=api_key,
                **kwargs
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")


if __name__ == "__main__":
    # Create OpenAI LLM instance
    openai_llm = LLMFactory.create_llm(
        provider="openai",
        model="gpt-4o",
        api_key="your key",
        temperature=0.0
    )

    # Create DeepSeek LLM instance
    deepseek_llm = LLMFactory.create_llm(
        provider="deepseek",
        model="deepseek-chat",
        api_key="your key",
        temperature=0.0,
        max_tokens=1000
    )

    # Create ZhipuAI LLM instance
    zhipuai_llm = LLMFactory.create_llm(
        provider="zhipuai",
        model="glm-4",
        api_key="your key",
        temperature=0.0,
        max_tokens=1000
    )
