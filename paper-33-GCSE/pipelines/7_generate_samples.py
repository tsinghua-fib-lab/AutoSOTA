# %%
import os
import json
import random
from tqdm import tqdm
from argparse import ArgumentParser

import sys
sys.path.append("../")
cmd_args = True
#
parser = ArgumentParser()
parser.add_argument('--n_gpu', default=1, help='n_gpu')
parser.add_argument('--skip', default=-1, help='skip the first n lines, the skip index is count from the start index of n-th chunks')
parser.add_argument('--file_dir', default='../dataset/mix', help='file name')
parser.add_argument('--file_name', default='__combine.txt', help='file name of the dataset')
parser.add_argument('--llm_name', default='', help='the prefix name of save dir (usually is the LLM name)')
parser.add_argument('--save_type_name', default='GLM4', help='the prefix name of save dir (usually is the LLM name)')
parser.add_argument('--model_from_pretrained', default='/home/lpc/models/glm-4-9b-chat/', help='model from pretrained')
parser.add_argument('--vllm', default='0', help='whether use vllm')
parser.add_argument('--tensor_parallel_size', default=1, help='tensor_parallel_size (TP) for vLLM')
parser.add_argument('--max_new_tokens', default=512, help='max new tokens')
parser.add_argument('--do_sample', default='1', help='do_sample')
parser.add_argument('--temperature', default=0.6, help='temperature')
parser.add_argument('--top_p', default=0.95, help='top_p')
parser.add_argument('--skip_thinking', default='0', help='skip deep thinking in RL model with <think>\n</think>')
parser.add_argument('--batch_size', default=20, help='batch size')

if not cmd_args:
    args = parser.parse_args([]) # You can directly set above parameters in the default.
else:
    args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.n_gpu)

API_MODELS = ['gpt-4o-mini', 'deepseek-chat', 'deepseek-reasoner']
API_CONFIGS = [('OpenAI', None), ('Deepseek', 'https://api.deepseek.com'), ('Deepseek', 'https://api.deepseek.com')]

USE_VLLM = str(args.vllm) == '1'

llm_name = args.llm_name if args.llm_name != '' else args.save_type_name
if llm_name == 'GLM3':
    from llm.predictor.chatglm import Predictor
elif llm_name in API_MODELS:
    from llm.predictor.openai import Predictor
elif USE_VLLM:
    from llm.predictor.vllm import Predictor
else:
    from llm.predictor.llm import Predictor

if llm_name not in API_MODELS:
    pred = Predictor(model_from_pretrained=args.model_from_pretrained, tensor_parallel_size=int(args.tensor_parallel_size))
else:
    CONFIG_INDEX = API_MODELS.index(llm_name)
    with open('api_key.txt') as f:
        api_keys = f.readlines()
    for key_item in api_keys:
        key_item = key_item.strip().split(' ')
        if len(key_item) == 1:
            api_key = key_item
            break
        else:
            if key_item[0] == API_CONFIGS[CONFIG_INDEX][0]:
                api_key = key_item[1]
                break
    pred = Predictor(api_key=api_key, base_url=API_CONFIGS[CONFIG_INDEX][1])

# %%
SOURCE_FILE = os.path.join(args.file_dir, args.file_name)
basename = os.path.basename(SOURCE_FILE)
SAVE_DIR = os.path.join(os.path.dirname(SOURCE_FILE), basename.split('.')[0] + f'_{args.save_type_name}_DA')
SAVE_FILE = os.path.join(SAVE_DIR,
                         basename.split('.')[0]+'_syn_samples.tsv')
BATCH_SIZE = int(args.batch_size)
MAX_NEW_TOKENS = int(args.max_new_tokens)

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# 
with open(SOURCE_FILE, encoding='utf-8') as f:
    ori_data = f.readlines()

CUR_ENTITY_FILE = os.path.join(SAVE_DIR, f"{basename.split('.')[0]}_entities.jsonl")
CUR_MAIN_PART_FILE = os.path.join(SAVE_DIR, f"{basename.split('.')[0]}_main_parts.jsonl")
ENTITY_KG_FILE = os.path.join(SAVE_DIR, 'entity_kg.jsonl')
MAIN_PART_KG_FILE = os.path.join(SAVE_DIR, 'main_part_kg.jsonl')

with open(CUR_ENTITY_FILE) as f:
    entity_list = f.readlines()
entity_list = [json.loads(item) for item in entity_list]

with open(CUR_MAIN_PART_FILE) as f:
    main_part_list = f.readlines()
main_part_list = [json.loads(item) for item in main_part_list]

with open(ENTITY_KG_FILE) as f:
    ekg_data = f.readlines()
ekg_data = [json.loads(item) for item in ekg_data]
entity_kg = {}
entity_kg_dict = {}
for item in tqdm(ekg_data):
    for key in item:
        entity_kg[key] = item[key]
        for entity_item in entity_kg[key]:
            entity_item['type'] = key
            if entity_item['text'] not in entity_kg_dict:
                entity_kg_dict[entity_item['text']] = entity_item

with open(MAIN_PART_KG_FILE) as f:
    mpkg_data = f.readlines()
mpkg_data = [json.loads(item) for item in mpkg_data]
main_part_kg = {
    'subject': [],
    'action': [],
    'state': []
}
main_part_kg_dict = {}
for item in tqdm(mpkg_data):
    for key in item:
        main_part_kg_dict[key] = item[key]
        data_type = item[key]['data_type']
        main_part_kg[data_type].append(item[key])

pos_prompts = {}
neg_prompts = {}
quantity_revise_prompt = ''
replace_entity_prompt = ''

with open('../prompts/rewrite_prompts.json') as f:
    pos_prompts = f.read()
    pos_prompts = json.loads(pos_prompts)

with open('../prompts/neg_prompts.json') as f:
    neg_prompts = f.read()
    neg_prompts = json.loads(neg_prompts)

with open('../prompts/quantity_revise.txt') as f:
    quantity_revise_prompt = f.read()

with open('../prompts/replace_entity.txt') as f:
    replace_entity_prompt = f.read()

# %%
pos_list = []
neg_list = []

if int(args.skip) > -1:
    ori_data = ori_data[int(args.skip):]

def get_related(entity, data_type=''):
    if data_type == '':
        kg_dict = entity_kg_dict
    else:
        kg_dict = main_part_kg_dict
    if entity not in kg_dict:
        return entity
    entity_item = kg_dict[entity]
    try:
        return entity_item['related'][-1][0]
    except:
        return entity

def get_random(entity, entity_type, data_type=''):
    if data_type == '':
        kg_dict = entity_kg
        target_type = entity_type
    else:
        kg_dict = main_part_kg
        target_type = data_type
    if target_type not in kg_dict:
        return entity
    try:
        entity_item = random.choice(kg_dict[target_type])
        return entity_item['text']
    except:
        return entity

def rint():
    return random.randint(1, 10)

for idx, tp in tqdm(enumerate(zip(ori_data, main_part_list, entity_list)), total=len(ori_data)):
    item, item_main_part, item_entities = tp
    new_item = item.strip()
    
    if rint() <= 7:
        prompt = random.choice(pos_prompts['styles'])['text'] + pos_prompts['suffix']
        ask_content = prompt.format(
            input_text=new_item.strip()
        )
        pos_list.append((ask_content, idx))
    else:
        prompt = replace_entity_prompt
        if rint() <= 5:
            data_type = random.choice(['subject', 'action', 'state'])
            try:
                entity = item_main_part[data_type][0]['text']
            except:
                entity = 'A man'
            replace_entity = get_related(entity, data_type)
        else:
            if len(item_entities) == 0:
                entity_item = {'entity': 'A man', 'type': 'person'}
            else:
                entity_item = random.choice(item_entities)
            if type(entity_item) != dict:
                entity_item = {'entity': 'A man', 'type': 'person'}
            if 'entity' not in entity_item:
                entity_item['entity'] = 'A man'
            if 'type' not in entity_item:
                entity_item['type'] = 'person'
            entity = entity_item['entity']
            replace_entity = get_related(entity_item['entity'])
        ask_content = prompt.format(
            ori_entities=f'"{entity}"',
            replace_entities=f'"{replace_entity}"',
            input_text=new_item.strip()
        )
        pos_list.append((ask_content, idx))
    
    if rint() <= 7:
        prompt = random.choice(neg_prompts['styles'])['text'] + neg_prompts['suffix']
        ask_content = prompt.format(
            input_text=new_item.strip()
        )
        neg_list.append((ask_content, idx))
    elif rint() <= 5:
        prompt = replace_entity_prompt
        if rint() <= 5:
            data_type = random.choice(['subject', 'action', 'state'])
            try:
                entity = item_main_part[data_type][0]['text']
            except:
                entity = 'A man'
            replace_entity = get_random(entity, '', data_type)
        else:
            if len(item_entities) == 0:
                entity_item = {'entity': 'A man', 'type': 'person'}
            else:
                entity_item = random.choice(item_entities)
            if type(entity_item) != dict:
                entity_item = {'entity': 'A man', 'type': 'person'}
            if 'entity' not in entity_item:
                entity_item['entity'] = 'A man'
            if 'type' not in entity_item:
                entity_item['type'] = 'person'
            entity = entity_item['entity']
            replace_entity = get_random(entity_item['entity'], entity_item['type'])
        ask_content = prompt.format(
            ori_entities=f'"{entity}"',
            replace_entities=f'"{replace_entity}"',
            input_text=new_item.strip()
        )
        neg_list.append((ask_content, idx))
    else:
        prompt = quantity_revise_prompt
        entity = item_main_part['subject']
        if len(entity) == 0:
            entity = 'A man'
        else:
            entity = entity[0]['text']
        replace_num = random.randint(0, 51)
        if rint() >= 5:
            replace_num = random.randint(51, 10000)
        ask_content = prompt.format(
            subject_text=entity,
            new_number=str(replace_num),
            input_text=new_item.strip()
        )
        neg_list.append((ask_content, idx))
   
def build_chat_custom(content):
    content = f'<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
    return content

if args.skip_thinking == '1':
    for idx, tp in enumerate(pos_list):
        pos_list[idx] = (build_chat_custom(tp[0]), tp[1])
    for idx, tp in enumerate(neg_list):
        neg_list[idx] = (build_chat_custom(tp[0]), tp[1])
    
#%%
# Calculate num batches.
if llm_name not in API_MODELS:
    num_batches = len(pos_list) // BATCH_SIZE + (1 if len(pos_list) % BATCH_SIZE != 0 else 0)

    # predict per batch and save results.
    for i in tqdm(range(num_batches)):
        pos_batch = pos_list[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
        pos_prompts = [item[0] for item in pos_batch]
        ids = [item[1] for item in pos_batch]

        neg_batch = neg_list[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
        neg_prompts = [item[0] for item in neg_batch]
        
        prompts = pos_prompts + neg_prompts
        outputs = pred(prompts, max_new_tokens=MAX_NEW_TOKENS, build_message=args.skip_thinking != '1', do_sample=args.do_sample == '1', temperature=float(args.temperature), top_p=float(args.top_p))
        pos_outputs = outputs[:len(outputs) // 2]
        neg_outputs = outputs[len(outputs) // 2:]
        with open(SAVE_FILE, encoding='utf-8', mode='a+') as f:
            for id, pos, neg in zip(ids, pos_outputs, neg_outputs):
                f.write(str(id) + '\t' + json.dumps(pos, ensure_ascii=False) + '\t' + json.dumps(neg, ensure_ascii=False) + '\n')
else:
    for pos_item, neg_item in tqdm(zip(pos_list, neg_list)):
        id = pos_item[1]
        pos_content = pos_item[0]
        neg_content = neg_item[0]
        pos_res = pred(pos_content, model=llm_name)
        pos_res = pos_res[0]
        pos_res = pos_res.replace('\n', '')
        pos_res = pos_res.replace(' ', '')

        neg_res = pred(pos_content, model=llm_name)
        neg_res = neg_res[0]
        neg_res = neg_res.replace('\n', '')
        neg_res = neg_res.replace(' ', '')
        with open(SAVE_FILE, encoding='utf-8', mode='a+') as f:
            f.write(str(id) + '\t' + json.dumps(pos_res, ensure_ascii=False) + '\t' + json.dumps(neg_res, ensure_ascii=False) + '\n')


#%%
