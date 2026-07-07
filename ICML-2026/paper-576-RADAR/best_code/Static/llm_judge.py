import os
import re
import logging
from typing import Optional

from openai import OpenAI

logger = logging.getLogger('RRAG-main')

judge_prompt = '''Here is the complete answer:

<Start of Answer>
{answer}
<End of Answer>

an LLM provides to the following question:

<Start of Question>
{question}
<End of Question>

Extract the ultimate answer provided by the LLM to the question, without any additional analysis, thinking, internal notes, etc. Provide your answer in the following format:

<Answer>
[Your Answer].
</Answer>

Example: Given the following complete answer:

<Start of Answer>
The first document provides irrelevant information to the question. The second document says Jack wins the prize at 2011 but seems incorrect.
The third document says Jack wins the prize at 2005 and seems more trustworthy. Thus, my answer is 2005.
<End of Answer>

an LLM provides to the following question:

<Start of Question>
When did Jack win the Nobel Prize?
<End of Question>

The answer you should provide is the following:

<ANSWER>
2005.
</ANSWER>

If the answer provided by the LLM is in a multiple-choice format, include the choice as well, e.g.

<Start of Answer>
... Thus, my answer is A. 2005.
<End of Answer>

<ANSWER>
A. 2005.
</ANSWER>
'''


class LLMJudge(object):
    """
    Judge 支持 OpenAI / DeepSeek / Gemini：
    - 三者统一使用 OpenAI SDK（chat.completions），差别只在 api_key/base_url。
    - Gemini 不再使用 google-genai，不再 fallback；行为与 GPT-4o 一致。
    """

    _OPENAI_MODELS = {"gpt-4o", "gpt-4o-mini", "o1-mini"}
    _DEEPSEEK_MODELS = {"deepseek-chat", "deepseek-reasoner"}
    _GROK_MODELS = {"grok-3"}

    def __init__(self, model: Optional[str] = None, provider: Optional[str] = None):
        self.model = model or os.environ.get("LLM_JUDGE_MODEL")
        if not self.model:
            raise RuntimeError("LLMJudge: 未提供 model，且环境变量 LLM_JUDGE_MODEL 也未设置。")

        self.provider = provider or self._infer_provider(self.model)

        # === OpenAI (官方) ===
        if self.provider == "openai":
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("环境变量 OPENAI_API_KEY 未设置。")

            base_url = os.environ.get("OPENAI_BASE_URL")  # 可选
            self.client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)

            logger.info(
                f"LLMJudge initialized: provider=openai model='{self.model}' base_url='{base_url or 'default'}'"
            )

        # === DeepSeek (OpenAI-compatible) ===
        elif self.provider == "deepseek":
            api_key = os.environ.get("DEEPSEEK_API_KEY")
            if not api_key:
                raise RuntimeError("环境变量 DEEPSEEK_API_KEY 未设置。")

            base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
            self.client = OpenAI(api_key=api_key, base_url=base_url)

            logger.info(f"LLMJudge initialized: provider=deepseek model='{self.model}' base_url='{base_url}'")

        # === Grok (OpenAI-compatible; 与 GPT4o 同样走 chat.completions) ===
        elif self.provider == "grok":
            api_key = (
                os.environ.get("XAI_API_KEY")
                or os.environ.get("GROK_API_KEY")
                or os.environ.get("OPENAI_API_KEY")
            )
            if not api_key:
                raise RuntimeError(
                    "环境变量 XAI_API_KEY / GROK_API_KEY / OPENAI_API_KEY 未设置其一（Grok 走 OpenAI-compatible）。"
                )

            # OpenAI-compatible base_url（默认 xAI）
            base_url = os.environ.get("OPENAI_BASE_URL") or "https://api.x.ai/v1"
            self.client = OpenAI(api_key=api_key, base_url=base_url)

            logger.info(f"LLMJudge initialized: provider=grok model='{self.model}' base_url='{base_url}'")


        else:
            raise RuntimeError(
                f"LLMJudge: 当前仅支持 API 模型（OpenAI/DeepSeek/Gemini）。model='{self.model}', provider='{self.provider}'"
            )

    def _infer_provider(self, model_name: str) -> str:
        m = (model_name or "").lower()
        if m in self._GROK_MODELS or m.startswith("grok-"):
            return "grok"
        if m in self._DEEPSEEK_MODELS or m.startswith("deepseek"):
            return "deepseek"
        if m in self._OPENAI_MODELS or m.startswith("gpt-") or m.startswith("o1-"):
            return "openai"
        return "unknown"

    def judge(self, question: str, answer: str) -> str:
        final_prompt = judge_prompt.format(question=question, answer=answer)
        response = self.get_output(final_prompt)
        final_response = self.extract_from_text(response, "ANSWER")
        logger.debug(f"Final response after post-processing by LLM judge: {final_response}")
        return final_response

    def get_output(self, prompt: str, temperature: float = 0.0) -> str:
        # 三家统一：OpenAI SDK chat.completions
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=1000,
                top_p=0.5,
                stream=False,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            logger.warning("LLMJudge error getting output: %s", e)
            return ""

    def extract_from_text(self, text: str, tag: str) -> str:
        try:
            pattern = fr'<{tag}>\s*(.*?)\s*</{tag}>'
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            return match.group(1).strip() if match else ""
        except Exception as e:
            logger.warning("LLM Judge error extracting from text: %s", e)
            return ""