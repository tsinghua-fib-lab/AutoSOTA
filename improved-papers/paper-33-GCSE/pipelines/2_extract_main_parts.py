# %%
import os
import json
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
parser.add_argument('--do_sample', default='0', help='do_sample')
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
                         basename.split('.')[0]+'_main_parts.tsv')
BATCH_SIZE = int(args.batch_size)
MAX_NEW_TOKENS = int(args.max_new_tokens)

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# 
with open(SOURCE_FILE, encoding='utf-8') as f:
    ori_data = f.readlines()

# %%
prompt_prefix = '''**Instruction**: Extract the theme category, subject part, action part, state part, and subject count from the given text. Output in JSON format.  

**Example**:  
Input: *"A man playing with a black dog on a white blanket."*  
Output:
{{
    cls: 'leisure activity',
    suject: [{{text: "A man", type: "person", quantity: 1}}],
    action: [{{text: "playing with a black dog"}}]
    state: [{{text: "on a white blanket"}}]
}}

**Output Format**:  
{{
    cls: [category],
    suject: [{{"text": "subject_text", "type": "subject_entity_type", "quantity": subject_count}}],
    action: [{{"text": "action_text"}}]
    state: [{{"text": "state_text"}}]
}}

**Input**: {input_text}
**Output**:
'''

# %%
ask_list = []

if int(args.skip) > -1:
    ori_data = ori_data[int(args.skip):]

for idx, item in tqdm(enumerate(ori_data)):
    ask_content = prompt_prefix.format(
        input_text=item.strip()
    )
    ask_list.append((ask_content, id))

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
