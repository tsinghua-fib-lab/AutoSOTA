import os
import sys
import openai
import json
from tqdm import tqdm


openai.api_key = os.getenv("OPENAI_API_KEY")
dataset = sys.argv[1]	# omniobject3d, objaverse_lvis
llm_name = sys.argv[2]	# gpt-3.5-turbo, gpt-4 

# NOTE replace the category_list with the desired categories
category_list = []
if dataset == "omniobject3d":
    for cls in os.listdir('data/omniobject3d/1024'):
        if os.path.isdir(os.path.join('data/omniobject3d/1024', cls)):
            category_list.append(cls)
            
elif dataset == "objaverse_lvis":
	with open('data/objaverse_lvis/classnames.txt') as fin:
		lines = fin.readlines()
		category_list = [line.strip() for line in lines]
  
else:
	print("Invalid dataset")
	sys.exit(1)

all_responses = {}

vowel_list = ['A', 'E', 'I', 'O', 'U']

for category in tqdm(category_list):

	if category[0].upper() in vowel_list:
		article = "an"
	else:
		article = "a"

	prompts = []
	
	prompts.append(f"What does {article} {category} point cloud look like?")
	prompts.append(f"What are the identifying characteristics of {article} {category} point cloud?")
	prompts.append(f"Please describe {article} {category} point cloud with details.")
	prompts.append(f"Make a complete and meaningful sentence with the following words: {category}, point cloud.")
	with open('prompts.json', 'w') as f:
		json.dump(prompts, f, indent=4)
  
	all_result = []
	for curr_prompt in prompts:
		response = openai.ChatCompletion.create(
			model=llm_name,
			messages=[
				{"role": "system", "content": "You are a helpful assistant."},
				{"role": "user", "content": curr_prompt}
			],
			max_tokens=70,
			n=10,
			stop="."
		)

		for r in range(len(response["choices"])):
			result = response["choices"][r]["message"]["content"]
			all_result.append(result.replace("\n\n", "") + ".")

	all_responses[category] = all_result

with open(f'{dataset}-{llm_name}.json', 'w') as f:
	json.dump(all_responses, f, indent=4)