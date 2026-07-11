import os
from dotenv import load_dotenv

load_dotenv()

import torch

# Login to Hugging Face using token from environment
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    from huggingface_hub import login
    login(hf_token)

from vllm import LLM, SamplingParams

class Inference:
    def __init__(self, model_name, max_tokens = 32768, temperature=None):
        self.model_name = model_name

        tp_size = 4 if torch.cuda.device_count() >= 4 else torch.cuda.device_count()
        kwargs = {
                "model": model_name,
                "enforce_eager": True,
                "trust_remote_code": True,
                "tensor_parallel_size": tp_size,
        }
        self.llm = LLM(**kwargs)

        if temperature is not None:
            self.sampling_params = SamplingParams(max_tokens=max_tokens, temperature=temperature)
        else:
            self.sampling_params = SamplingParams(max_tokens=max_tokens)

    def get_response(self, prompts, enable_thinking=True):
        messages = []
        for prompt in prompts:
            messages.append(
                [{"role": "user", "content": prompt}]
            )
        outputs = self.llm.chat(
            messages,
            self.sampling_params,
            chat_template_kwargs={"enable_thinking": enable_thinking}
        )
        
        results = []
        for output in outputs:
            generated_text = output.outputs[0].text
            if enable_thinking:
                if "</think>" in generated_text:
                    content = generated_text.split("</think>")[1].strip()
                    thinking_content = generated_text.split("</think>")[0].strip()
                elif "assistantfinal" in generated_text and self.model_name.startswith("openai/"):
                    content = generated_text.split("assistantfinal")[1].strip()
                    thinking_content = generated_text.split("assistantfinal")[0].strip().replace("analysis", "").strip()
                else:
                    # If the </think> token is not found, it assumes that the entire output is thinking_content.
                    thinking_content = generated_text.strip()
                    content = ""
                results.append((thinking_content, content))
            else:
                results.append(generated_text.strip())
        return results