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
                         basename.split('.')[0]+'_ext_ori.tsv')
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
for item in tqdm(ekg_data):
    for key in item:
        entity_kg[key] = item[key]

with open(MAIN_PART_KG_FILE) as f:
    mpkg_data = f.readlines()
mpkg_data = [json.loads(item) for item in mpkg_data]
main_part_kg_dict = {}
main_part_type_dict = {}
for item in tqdm(mpkg_data):
    for key in item:
        main_part_kg_dict[key] = item[key]
        for cls_name in item[key]['cls']:
            if cls_name not in main_part_type_dict:
                main_part_type_dict[cls_name] = {
                    'subject': [],
                    'action': [],
                    'state': []
                }
            main_part_type_dict[cls_name][item[key]['data_type']].append(item[key])

# %%
prompt_prefix = '''**Instruction**: Fill in the bracketed [] sections of this news-themed sentence (one word per []), then evaluate and polish if the result sounds unnatural. If already fluent, keep the original filled version.

**Evaluation Criteria**:
1. Each [] = exactly one English word
2. News-style vocabulary preferred
3. Polish if:
   - Grammar/syntax errors exist
   - Logical inconsistencies appear
   - News tone is violated

**Example**:  
Input: *"I [] you."*  
Output:
{{
  "ori": "I love you.", 
  "polish": "I love you."
}}

**Input**: {input_text}
**Output**:  
'''

# %%
ask_list = []

if int(args.skip) > -1:
    ori_data = ori_data[int(args.skip):]

def valid_main_part(item):
    if type(item) != list:
        return False
    if len(item) <= 0:
        return False
    if type(item[0]) != dict:
        return False
    if 'text' not in item[0]:
        return False
    if len(item[0]['text']) <= 0:
        return False
    if item[0]['text'].lower() in 'replace_subject':
        return False
    if item[0]['text'].lower() in 'replace_action':
        return False
    if item[0]['text'].lower() in 'replace_state':
        return False
    return True

def get_random_mask(text):
    text = text.split(' ')
    for i, _ in enumerate(text):
        if random.randint(0, 10) > 7:
            text[i] = '[]'
    return ' '.join(text)

mode_list = ['r_main_part', 'r_entity']
mode_weight = [0.7, 0.3]
for idx, tp in tqdm(enumerate(zip(ori_data, main_part_list, entity_list)), total=len(ori_data)):
    item, item_main_part, item_entities = tp
    new_item = item.strip()
    mode = random.choices(mode_list, weights=mode_weight)[0]
    if mode == 'r_main_part' or len(item_entities) == 0:
        subject, action, state = item_main_part['subject'], item_main_part['action'], item_main_part['state']
        cls_name = item_main_part['cls'][0] if len(item_main_part['cls']) > 0 else 'news'
        if cls_name not in main_part_type_dict or len(main_part_type_dict[cls_name]) == 0:
            cls_name = 'news'
        if valid_main_part(subject):
            # new_item = new_item.lower().replace(subject[0]['text'].lower(), '{replace_subject}')
            # replace_subject = get_random_mask(subject[0]['text'].lower())
            try:
                new_item = new_item.lower().replace(subject[0]['text'].lower(), '{replace_subject}')
                replace_subject = random.choice(main_part_type_dict[cls_name]['subject'])['text']
            except:
                replace_subject = get_random_mask(subject[0]['text'])
        else:
            replace_subject = ''
        if valid_main_part(action):
            try:
                new_item = new_item.lower().replace(action[0]['text'].lower(), '{replace_action}')
                replace_action = random.choice(main_part_type_dict[cls_name]['action'])['text']
            except:
                replace_action = action[0]['text']
        else:
            replace_action = ''
        if valid_main_part(state):
            try:
                new_item = new_item.lower().replace(state[0]['text'].lower(), '{replace_state}')
                replace_state = random.choice(main_part_type_dict[cls_name]['state'])['text']
            except:
                replace_state = state[0]['text']
        else:
            replace_state = ''
        
        new_item = new_item.split(' ')
        for i, word in enumerate(new_item):
            if word.startswith('{replace'):
                continue
            new_item[i] = '[]'
        new_item = ' '.join(new_item)
        new_item = new_item.format(replace_subject=replace_subject, replace_action=replace_action, replace_state=replace_state)
    elif mode == 'r_entity':
        for i, entity_item in enumerate(item_entities):
            if type(entity_item) != dict:
                continue
            if i > 3:
                break
            entity, entity_type = entity_item['entity'], entity_item['type']
            if entity_type not in entity_kg:
                replace_entity = 'man'
            else:
                replace_entity = random.choice(entity_kg[entity_type])['text']
            new_item = new_item.lower().replace(entity.lower(), f"$${replace_entity.replace(' ', '@')}")
        new_item = new_item.split(' ')
        for i, word in enumerate(new_item):
            if word.startswith('$$'):
                continue
            new_item[i] = '[]'
        new_item = ' '.join(new_item)
        new_item = new_item.replace('$$', '')
        new_item = new_item.replace('@', ' ')
    
    random_suffix = random.randint(0, 100)
    if random_suffix >= 95:
        random_suffix = random_suffix % 5
        random_suffix = ['[]' for _ in range(random_suffix)]
        new_item+= ' '.join(random_suffix)
    
    ask_content = prompt_prefix.format(
        input_text=new_item.strip()
    )
    ask_list.append((ask_content, idx))

def build_chat_custom(content):
    content = f'<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
    return content

if args.skip_thinking == '1':
    for idx, tp in enumerate(ask_list):
        ask_list[idx] = (build_chat_custom(tp[0]), tp[1])
    
#%%
# Calculate num batches.
if llm_name not in API_MODELS:
    num_batches = len(ask_list) // BATCH_SIZE + (1 if len(ask_list) % BATCH_SIZE != 0 else 0)

    # predict per batch and save results.
    for i in tqdm(range(num_batches)):
        batch = ask_list[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
        prompts = [item[0] for item in batch]
        ids = [item[1] for item in batch]
        
        outputs = pred(prompts, max_new_tokens=MAX_NEW_TOKENS, build_message=args.skip_thinking != '1', do_sample=args.do_sample == '1', temperature=float(args.temperature), top_p=float(args.top_p))
        with open(SAVE_FILE, mode='a+') as f:
            for item in outputs:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
else:
    for ask_content, id in tqdm(ask_list):
        res = pred(ask_content, model=llm_name)
        res = res[0]
        res = res.replace('\n', '')
        res = res.replace(' ', '')
        with open(SAVE_FILE, mode='a+') as f:
            f.write(json.dumps(res, ensure_ascii=False) + '\n')


#%%
