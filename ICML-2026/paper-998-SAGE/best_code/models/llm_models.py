"""
LLM Generation Models
For open-ended question answering tasks
"""

from typing import Dict, Any
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from .base_model import BaseModel
import os
from openai import OpenAI


class LlamaGenerator(BaseModel):
    """Llama-3.1-8B generation model"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = config.get('model_name', 'meta-llama/Llama-3.1-8B-Instruct')
        self.max_new_tokens = config.get('max_new_tokens', 256)
        self.temperature = config.get('temperature', 0.7)
        self.model = None
        self.tokenizer = None
        
    def load_model(self):
        """Load model"""
        print(f"Loading Llama model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        dtype = torch.float16 if self.device == 'cuda' else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=dtype,
            device_map="auto" if self.device == 'cuda' else None
        )
        if self.device != 'cuda':
            self.model.to(self.device)
        self.model.eval()
        print("Model loaded")
    
    @torch.no_grad()
    def predict(self, question: str) -> Dict[str, Any]:
        if self.model is None:
            self.load_model()

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful and truthful assistant. "
                    "Answer the question in ONE short sentence (no more than 20 words). "
                    "Do NOT give any explanation, background or multiple hypotheses."
                ),
            },
            {"role": "user", "content": question},
        ]

        try:
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            prompt = f"<|system|>\nYou are a helpful assistant.\n<|user|>\n{question}\n<|assistant|>\n"

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,  # Suggest setting to 32/48 in config
            do_sample=False,                     # Don't sample during evaluation
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        # Keep only the newly generated part
        new_tokens = generated_ids[:, inputs.input_ids.shape[1]:]
        raw_answer = self.tokenizer.decode(
            new_tokens[0], skip_special_tokens=True
        ).strip()

        # Simple post-processing: take only the first sentence
        answer = raw_answer.split("\n")[0]
        for sep in [".", "?", "!"]:
            if sep in answer:
                answer = answer.split(sep)[0]
                break
        answer = answer.strip()

        # Truncation detection: if generated length == limit, might be truncated
        is_truncated = new_tokens.shape[1] >= self.max_new_tokens - 1

        return {
            "generated_answer": answer,
            "raw_output": {
                "raw_answer": raw_answer,
                "prompt_length": int(inputs.input_ids.shape[1]),
                "generated_length": int(new_tokens.shape[1]),
                "temperature": float(self.temperature),
                "max_new_tokens": int(self.max_new_tokens),
                "possibly_truncated": bool(is_truncated),
            },
        }

    @torch.no_grad()
    def generate(self, prompt: str) -> str:
        """
        Direct generation: receive complete prompt, return generated text
        For scenarios requiring custom prompt like scoring
        """
        if self.model is None:
            self.load_model()
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        
        # Keep only the newly generated part
        new_tokens = generated_ids[:, inputs.input_ids.shape[1]:]
        response = self.tokenizer.decode(new_tokens[0], skip_special_tokens=True).strip()
        
        return response


class Qwen3Generator(BaseModel):
    """Qwen3-8B generation model"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = config.get('model_name', 'Qwen/Qwen2.5-7B-Instruct')
        self.max_new_tokens = config.get('max_new_tokens', 256)
        self.temperature = config.get('temperature', 0.7)
        self.model = None
        self.tokenizer = None
        
    def load_model(self):
        """Load model"""
        print(f"Loading Qwen3 model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.model.eval()
        print("Model loaded")
    
    @torch.no_grad()
    def predict(self, question: str) -> Dict[str, Any]:
        if self.model is None:
            self.load_model()

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful and truthful assistant. "
                    "Answer the question in ONE short sentence (no more than 20 words). "
                    "Do NOT give any explanation, background or multiple hypotheses."
                ),
            },
            {"role": "user", "content": question},
        ]

        try:
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )
        except Exception:
            prompt = f"<|system|>\nYou are a helpful assistant.\n<|user|>\n{question}\n<|assistant|>\n"

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,  # Suggest setting to 32/48 in config
            do_sample=False,                     # Don't sample during evaluation
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        # Keep only the newly generated part
        new_tokens = generated_ids[:, inputs.input_ids.shape[1]:]
        raw_answer = self.tokenizer.decode(
            new_tokens[0], skip_special_tokens=True
        ).strip()

        # Simple post-processing: take only the first sentence
        answer = raw_answer.split("\n")[0]
        for sep in [".", "?", "!"]:
            if sep in answer:
                answer = answer.split(sep)[0]
                break
        answer = answer.strip()

        # Truncation detection: if generated length == limit, might be truncated
        is_truncated = new_tokens.shape[1] >= self.max_new_tokens - 1

        return {
            "generated_answer": answer,
            "raw_output": {
                "raw_answer": raw_answer,
                "prompt_length": int(inputs.input_ids.shape[1]),
                "generated_length": int(new_tokens.shape[1]),
                "temperature": float(self.temperature),
                "max_new_tokens": int(self.max_new_tokens),
                "possibly_truncated": bool(is_truncated),
            },
        }

    @torch.no_grad()
    def generate(self, prompt: str) -> str:
        """
        Direct generation: receive complete prompt, return generated text
        For scenarios requiring custom prompt like scoring
        """
        if self.model is None:
            self.load_model()
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        
        # Keep only the newly generated part
        new_tokens = generated_ids[:, inputs.input_ids.shape[1]:]
        response = self.tokenizer.decode(new_tokens[0], skip_special_tokens=True).strip()
        
        return response


class OpenAIGenerator(BaseModel):
    """OpenAI GPT-4o-mini generation model"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = config.get('api_key') or os.getenv('OPENAI_API_KEY')
        self.model_name = config.get('model_name', 'gpt-4o-mini')
        self.max_tokens = config.get('max_tokens', 256)
        self.temperature = config.get('temperature', 0.7)
        self.client = OpenAI(api_key=self.api_key)
        
    def load_model(self):
        """OpenAI API doesn't need loading"""
        print(f"Using OpenAI model: {self.model_name}")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found.")
    
    def predict(self, question: str) -> Dict[str, Any]:
        if not self.api_key:
            self.load_model()

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful and truthful assistant. "
                    "Answer the question in ONE short sentence (no more than 20 words). "
                    "Do NOT give any explanation, background or multiple hypotheses."
                ),
            },
            {"role": "user", "content": question},
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=self.max_tokens,   # Suggest 32/48
                temperature=0.0,              # Suggest 0 for evaluation
            )
            choice = response.choices[0]
            answer_raw = choice.message.content.strip()
            finish_reason = choice.finish_reason

            # Also take only the first sentence
            answer = answer_raw.split("\n")[0]
            for sep in [".", "?", "!"]:
                if sep in answer:
                    answer = answer.split(sep)[0]
                    break
            answer = answer.strip()

            is_truncated = (finish_reason == "length")

            return {
                "generated_answer": answer,
                "raw_output": {
                    "model": self.model_name,
                    "temperature": float(self.temperature),
                    "finish_reason": finish_reason,
                    "possibly_truncated": bool(is_truncated),
                    "usage": {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens,
                    },
                    "raw_answer": answer_raw,
                },
            }

        except Exception as e:
            print(f"❌ API Error: {e}")
            return {
                "generated_answer": f"Error: {str(e)}",
                "raw_output": {"error": str(e)},
            }
class MinistralGenerator(BaseModel):
    """Ministral-8B-Instruct-2410 generation model"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = config.get('model_name', 'mistralai/Ministral-8B-Instruct-2410')
        self.max_new_tokens = config.get('max_new_tokens', 256)
        self.temperature = config.get('temperature', 0.7)
        self.model = None
        self.tokenizer = None
        
    def load_model(self):
        """Load model"""
        print(f"Loading Ministral model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Use bfloat16 (recommended) or float16
        if self.device == 'cuda':
            torch_dtype = torch.bfloat16
            device_map = "auto"
        else:
            torch_dtype = torch.float32
            device_map = None
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch_dtype,
            device_map=device_map
        )
        if self.device != 'cuda':
            self.model.to(self.device)
        self.model.eval()
        print("Ministral model loaded")
    
    @torch.no_grad()
    def predict(self, question: str) -> Dict[str, Any]:
        if self.model is None:
            self.load_model()

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful and truthful assistant. "
                    "Answer the question in ONE short sentence (no more than 20 words). "
                    "Do NOT give any explanation, background or multiple hypotheses."
                ),
            },
            {"role": "user", "content": question},
        ]

        try:
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            # Fallback prompt
            system = messages[0]["content"]
            user = messages[1]["content"]
            prompt = f"<|system|>\n{system}\n<|user|>\n{user}\n<|assistant|>\n"

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,  # Use greedy decoding for evaluation
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        # Keep only the newly generated part
        new_tokens = generated_ids[:, inputs.input_ids.shape[1]:]
        raw_answer = self.tokenizer.decode(
            new_tokens[0], skip_special_tokens=True
        ).strip()

        # Simple post-processing: take only the first sentence
        answer = raw_answer.split("\n")[0]
        for sep in [".", "?", "!"]:
            if sep in answer:
                answer = answer.split(sep)[0] + sep
                break
        answer = answer.strip()

        # Truncation detection
        is_truncated = new_tokens.shape[1] >= self.max_new_tokens - 1

        return {
            "generated_answer": answer,
            "raw_output": {
                "raw_answer": raw_answer,
                "prompt_length": int(inputs.input_ids.shape[1]),
                "generated_length": int(new_tokens.shape[1]),
                "temperature": float(self.temperature),
                "max_new_tokens": int(self.max_new_tokens),
                "possibly_truncated": bool(is_truncated),
            },
        }

    @torch.no_grad()
    def generate(self, prompt: str) -> str:
        """
        Direct generation: receive complete prompt, return generated text
        For scenarios requiring custom prompt like scoring
        """
        if self.model is None:
            self.load_model()
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        
        # Keep only the newly generated part
        new_tokens = generated_ids[:, inputs.input_ids.shape[1]:]
        response = self.tokenizer.decode(new_tokens[0], skip_special_tokens=True).strip()
        
        return response