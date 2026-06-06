import json
from evaluators.evaluator import Evaluator
from time import sleep
import re, os
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig


class Deepseek_Evaluator(Evaluator):
	def __init__(self, model_name, cuda_device_id=0, k=-1):
		super(Deepseek_Evaluator, self).__init__(model_name, k)
		# model_name = "../../deepseek-math-7b-rl"
		self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
		self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map=f'cuda:{cuda_device_id}', trust_remote_code=True)
		self.model.generation_config = GenerationConfig.from_pretrained(model_name)
		self.model.generation_config.pad_token_id = self.model.generation_config.eos_token_id


	def make_input(self, prompt, question_content):
		content = prompt + '\n' + question_content + '\n'
		# Adding the prompt recommended in Deepseek-Math's huggingface repository
		if self.is_chinese:
			content += '请通过逐步推理来解答问题，并把最终答案放置于\\boxed{}中。'
		else:
			content += 'Please reason step by step, and put your final answer after "The final answer is ".'
		messages = [{
			'role': 'user',
			'content': content
		}]
		return messages

	def get_answer(self, input):
		# print(input)
		input_tensor = self.tokenizer.apply_chat_template(input, add_generation_prompt=True, return_tensors="pt")
		outputs = self.model.generate(input_tensor.to(self.model.device), max_new_tokens=2048)
		result = self.tokenizer.decode(outputs[0][input_tensor.shape[1]:], skip_special_tokens=True)
		# print(result)
		return result
