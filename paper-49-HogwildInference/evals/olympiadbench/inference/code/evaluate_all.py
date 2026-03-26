import os
import json
import argparse
from evaluators.deepseek_evaluator import Deepseek_Evaluator
from evaluators.llama_evaluator import LlamaEvaluator
from evaluators.llama_parallel_evaluator import ParallelLlamaEvaluator

# from evaluators.yi_vl import YI_VL_Evaluator
# from evaluators.yi_chat import YI_Chat_Evaluator


def main(args):
	if args.make_parallel:
		assert 'TO' in args.dataset_name	# text-only model
		evaluator = ParallelLlamaEvaluator(
			model_name=args.model_name,
			cuda_device_id=args.cuda_device
		)
	elif 'deepseek' in args.model_name:
		assert 'TO' in args.dataset_name	# text-only model
		evaluator = Deepseek_Evaluator(
			model_name=args.model_name,
			cuda_device_id=args.cuda_device
		)
	elif 'Llama' in args.model_name or 'QwQ' in args.model_name:
		assert 'TO' in args.dataset_name	# text-only model
		evaluator = LlamaEvaluator(
			model_name=args.model_name,
			cuda_device_id=args.cuda_device
		)
	# elif 'Yi-VL' in args.model_name:
	# 	evaluator = YI_VL_Evaluator(
	# 		model_name=args.model_name
	# 	)
	# 	args.model_name = 'Yi-VL'
	# elif 'Yi-34B-Chat' in args.model_name:
	# 	evaluator = YI_Chat_Evaluator(
	# 		model_name=args.model_name
	# 	)
	# 	args.model_name = 'Yi-34B-Chat'
	# elif 'Nous-Hermes-2-Yi-34B' in args.model_name:
	# 	evaluator = YI_Chat_Evaluator(
	# 		model_name=args.model_name
	# 	)
	# 	args.model_name = 'Nous-Hermes-2-Yi-34B'
	else:
		print("Unknown model name")
		exit()
	if not os.path.exists(args.save_dir):
		os.mkdir(args.save_dir)

	dataset_save_dir = os.path.join(args.save_dir, args.dataset_name)
	if not os.path.exists(dataset_save_dir):
		os.mkdir(dataset_save_dir)
	# dataset_path = os.path.join('data_0301', args.dataset_name)
	if not args.saving_name:
		save_result_dir = os.path.join(dataset_save_dir, args.model_name)
	else:
		save_result_dir=os.path.join(dataset_save_dir, args.saving_name)

	print("dataset_path: ", args.dataset_path)
	print("save_result_dir: ", save_result_dir)
	with open(args.dataset_path, 'r', encoding='utf-8') as f:
		json_dataset = json.load(f)
		evaluator.eval_dataset(
			json_dataset_path=args.dataset_path,
			json_dataset=json_dataset,
			save_result_dir=save_result_dir
		)
			

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--model_name", type=str, required=True)
	parser.add_argument("--dataset_path", type=str, required=True)
	parser.add_argument("--save_dir", type=str, default='../generated')
	parser.add_argument("--saving_name", type=str)
	parser.add_argument("--cuda_device", type=int)    
	parser.add_argument("--make_parallel", action='store_true') 
	args = parser.parse_args()
	args.dataset_name = args.dataset_path.split("/")[-1].strip()[:-5]
	main(args)