import json
from inference.code.evaluators.evaluator import Evaluator
from time import sleep
import re, os
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig


class LlamaEvaluator(Evaluator):
    def __init__(self, model_name, k=-1, budget=16384, device_map='auto', torch_dtype=torch.bfloat16, **kwargs):
        super(LlamaEvaluator, self).__init__(model_name, k)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, **kwargs)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, low_cpu_mem_usage=True,
                                                          torch_dtype=torch_dtype, device_map=device_map, **kwargs)
        self.model.generation_config = GenerationConfig.from_pretrained(model_name)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.budget = budget

    def make_input(self, prompt, question_content):
        content = prompt + '\n' + question_content + '\n'
        # Adding the prompt recommended in Deepseek-Math's huggingface repository
        if self.is_chinese:
            content += '请通过逐步推理来解答问题，并把最终答案放置于\\boxed{}中。'
        else:
            content += 'Please reason step by step, and put your final answer within \\boxed{}.'
        messages = [{
            'role': 'user',
            'content': content
        }]
        return messages

    def get_answer(self, input):
        # print(input)
        input_tensor = self.tokenizer.apply_chat_template(input, add_generation_prompt=True, return_tensors="pt")
        outputs = self.model.generate(input_tensor.to(self.model.device), max_new_tokens=self.budget)
        result = self.tokenizer.decode(outputs[0][input_tensor.shape[1]:], skip_special_tokens=True)
        # print(result)
        return result
