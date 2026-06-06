import os, re
import json
from tqdm import tqdm

chinese_answer_type_dict = {
	'Numerical': '数值',
	'Expression': '表达式',
	'Equation': '方程',
	'Interval': '区间'
}
english_answer_type_dict = {
	'Numerical': 'a numerical value',
	'Expression': 'an expression',
	'Equation': 'an equation',
	'Interval': 'an interval'
}

def get_single_answer_type_text(answer_type, is_chinese):
	if '-' in answer_type:	# No need now
		answer_type = answer_type[:answer_type.find('-')]
	for t in ['Numerical', 'Expression', 'Equation', 'Interval']:
		if t in answer_type:
			if is_chinese:
				return chinese_answer_type_dict[t]
			else:
				return english_answer_type_dict[t]
	exit(f'Error parsing answer type {answer_type}!')

def get_answer_type_text(answer_type, is_chinese, multiple_answer):
	if ('Need_human_evaluate' in answer_type) or ('Tuple' in answer_type):	# 'Tuple' has various meanings in different context, such as position or values of a series of variable, so it may lead to confusion to directly use 'tuple' in the prompt.
		full_answer_text = ''
	else:
		if not multiple_answer:
			answer_text = get_single_answer_type_text(answer_type, is_chinese)
			if is_chinese:
				full_answer_text = f'，答案类型为{answer_text}'
			else:
				full_answer_text = f"The answer of The problem should be {answer_text}. "
		else:
			if ',' not in answer_type:	# Same answer type for all answers
				answer_text = get_single_answer_type_text(answer_type, is_chinese)
				if is_chinese:
					full_answer_text = f'，题目有多个答案，答案类型均为{answer_text}'
				else:
					full_answer_text = f'The problem has multiple answers, each of them should be {answer_text}. '
			else:
				answer_types = answer_type.split(',')
				answer_types = [get_single_answer_type_text(t, is_chinese) for t in answer_types]
				if len(set(answer_types)) == 1:
					answer_text = answer_types[0]
					if is_chinese:
						full_answer_text = f'，题目有多个答案，答案类型均为{answer_text}'
					else:
						full_answer_text = f'The problem has multiple answers, each of them should be {answer_text}. '
				else:
					if is_chinese:
						answer_text = '、'.join(answer_types)
						full_answer_text = f'，题目有多个答案，答案类型分别为{answer_text}'
					else:
						answer_text = ', '.join(answer_types)
						full_answer_text = f'The problem has multiple answers, with the answers in order being {answer_text}. '
	return full_answer_text

class Evaluator:
	def __init__(self, model_name, k=-1):
		self.model_name = model_name
		self.k = k
		self.json_dataset_path = ''
		self.image_parent_dir = ''

	def make_prompt(self, question):
		if self.is_chinese:
			subject_content = '数学' if self.is_math else '物理'
			if self.is_theorem_proving:
				prompt = f'以下是中国{subject_content}竞赛中的证明题。请根据题目的要求，运用逻辑推理及常用定理证明题目中的命题。证明过程中使用的变量和公式请使用LaTeX格式表示。'
			else:
				answer_type_text = get_answer_type_text(question['answer_type'], is_chinese=True, multiple_answer=question['is_multiple_answer'])
				if question['is_multiple_answer']:
					multiple_answer_text = '\\boxed{用英文逗号连接的多个答案}'
				else:
					multiple_answer_text = '\\boxed{答案}'
				unit_text = ''
				if question['unit']:
					multiple_answer_text += '(单位)'
					unit_text = '，注意答案的单位不要放在\\boxed{}中'
				prompt = f'以下是中国{subject_content}竞赛中的解答题{answer_type_text}。请根据题目的要求和所提供的信息计算得出答案。解答过程和结果中使用的变量和公式请使用LaTeX格式表示。请在最后以“所以最终答案是{multiple_answer_text}。”显式给出结果{unit_text}。'
		else:
			subject_content = 'Math' if self.is_math else 'Physics'
			if self.is_theorem_proving:
				prompt = f'The following is a theorem proving problem from an International {subject_content} competition. Please use logical reasoning and common theorems to prove the proposition in the problem according to the given requirements. Please use LaTeX format to represent the variables and formulas used in the proof.'
			else:
				if question['is_multiple_answer']:
					multiple_answer_text = '\\boxed{multiple answers connected with commas}'
				else:
					multiple_answer_text = '\\boxed{answer}'
				unit_text = ''
				if question['unit']:
					multiple_answer_text += '(unit)'
					unit_text = ', note that the unit of the answer should not be included in \\boxed{}'
				answer_type_text = get_answer_type_text(question['answer_type'], is_chinese=False, multiple_answer=question['is_multiple_answer'])
				prompt = f'The following is an open-ended problem from an International {subject_content} competition. {answer_type_text} Please calculate the answer according to the given requirements and the information provided. Please use LaTeX format to represent the variables and formulas used in the solution process and results. Please end your solution with "The final answer is {multiple_answer_text}." and give the result explicitly {unit_text}.'
		return prompt

	def make_input(self, prompt, question_content):
		input = prompt + '\n' + question_content
		return input
	
	def get_answer(self, input):
		pass

	def get_image_mapping_dict(self):
		print(self.json_dataset_path)
		# self.image_parent_dir = os.path.join(os.path.dirname(self.json_dataset_path), 'images')
		self.image_parent_dir = os.path.join(os.path.dirname(os.path.dirname(self.json_dataset_path)), 'images')
		if not os.path.exists(self.image_parent_dir):
			print('Cannot find image directory!')
			exit()

	def eval_dataset(self, json_dataset_path, json_dataset, save_result_dir):
		self.json_dataset_path = json_dataset_path
		self.get_image_mapping_dict() # Confirm if there is an image folder
		self.is_theorem_proving = 'TP' in json_dataset_path
		self.is_math = 'math' in json_dataset_path
		self.is_chinese = 'zh' in json_dataset_path

		model_name = self.model_name.split("/")[-1].strip() # For paths like deepseek, extract the model name.

		os.makedirs(save_result_dir, exist_ok=True)
		temp_result = []

		for id in tqdm(range(min(40, len(json_dataset)))):#, len(json_dataset))):
			question = json_dataset[id]
			prompt = self.make_prompt(question)
			if self.is_math:
				input = self.make_input(prompt, question['question'])
			else:
				if 'context' in question.keys() and question['context']: # cannot be null
					input = self.make_input(prompt, question['context']+'\n'+question['question'])
				else:
					input = self.make_input(prompt, question['question'])
			answer = self.get_answer(input)
			if 'model_output' not in question.keys():
				question['model_output'] = {model_name:{'raw_output':answer}}
			else:
				question['model_output'][model_name] = {'raw_output':answer}
			temp_result.append(question)
			#if id % 100 == 99:
			save_start_id = id #- 99
			with open(os.path.join(save_result_dir, f'{model_name}_{save_start_id}_to_{id}.json'), 'w', encoding='utf-8') as f:
				json.dump(temp_result, f, ensure_ascii=False, indent=4)
			temp_result = []
		if temp_result:
			save_start_id = 100 * int(id / 100)
			with open(os.path.join(save_result_dir, f'{model_name}_{100*(int(id/100))}_to_{id}.json'), 'w', encoding='utf-8') as f:
				json.dump(temp_result, f, ensure_ascii=False, indent=4)
		print(f'Evaluation finished for {json_dataset_path}.')