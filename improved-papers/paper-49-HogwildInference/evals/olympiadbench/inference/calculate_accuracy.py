import os, sys, re
import json
import argparse


def calculate_merged_accuracy(reference_dir, text_only, model_output):
	if text_only:
		print('*'*20 + ' evaluating in text-only mode ' + '*'*20)

	for model in os.listdir(model_output):
		print('-'*10 + model + '-'*10)
		total_scored_num = 0
		total_correct_num = 0
		for dataset_fn in os.listdir(os.path.join(model_output, model)):
			if dataset_fn[-5:] != '.json':
				continue
			dataset = dataset_fn[:-5]
			if 'TP' in dataset:	# 只统计可机评的
				continue
			full_num = 0
			correct_num = 0
			
			with open(os.path.join(reference_dir, dataset_fn), 'r', encoding='utf-8') as f:
				ref_json = json.load(f)
				ref_id_list = [d['id'] for d in ref_json]
				
			generated_id_list = []

			single_result_path = os.path.join(model_output, model, dataset_fn)
			with open(single_result_path, 'r', encoding='utf-8') as f:
				single_result_dict = json.load(f)
				
				available_single_result_dict = [d for d in single_result_dict if (len(d['model_output'][model]['raw_output'])>0 and d['model_output'][model]['raw_output'] != '<Inappropriate content in response>' and d['model_output'][model]['raw_output']!='<No response>' and ('code:' not in d['model_output'][model]['raw_output'] or 'message:' not in d['model_output'][model]['raw_output']))]
				generated_id_list += [d['id'] for d in available_single_result_dict if (d['id'] in ref_id_list)]
				correct_num += len([1 for d in available_single_result_dict if (d['model_output'][model]['correctness'] and d['id'] in ref_id_list)])

			full_num = len(generated_id_list)
			missing_id_list = [id for id in ref_id_list if id not in generated_id_list]
			if missing_id_list:
				id_path = os.path.join(model_output, model, f'{dataset}_missing_id.txt')
				with open(id_path, 'w', encoding='utf-8') as id_f:
					id_f.write(str(missing_id_list))
			accuracy = -1
			if full_num > 0:
				accuracy = (correct_num + 0.0) / full_num * 100
			print('-'*3 + dataset)
			print(f'{full_num} finished\t|{full_num} scored\t|accuracy={accuracy:.2f}%\t|{len(missing_id_list)} missing\t|{correct_num} correct')
			total_scored_num += full_num
			total_correct_num += correct_num
		if total_scored_num == 0:
			total_scored_num += 1
		print('Average accuracy: ', (total_correct_num + 0.0) / total_scored_num * 100)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--ref_dir", required=True) # Path to origin dataset
	parser.add_argument("--text_only", action='store_true')
	parser.add_argument("--model_output", type=str, default='merged_result')
	args = parser.parse_args()
	calculate_merged_accuracy(
		reference_dir=args.ref_dir,
		text_only=args.text_only,
		model_output=args.model_output
	)