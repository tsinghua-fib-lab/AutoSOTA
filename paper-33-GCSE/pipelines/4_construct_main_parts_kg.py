# %%
import os
import json
import json_repair
from tqdm import tqdm
from copy import deepcopy
from argparse import ArgumentParser
from sentence_transformers import SentenceTransformer
import faiss

import sys
sys.path.append("../")
cmd_args = True
#
parser = ArgumentParser()
parser.add_argument('--n_gpu', default=1, help='n_gpu')
parser.add_argument('--file_dir', default='../dataset/mix', help='file name')
parser.add_argument('--file_name', default='__combine.txt', help='file name of the dataset')
parser.add_argument('--save_type_name', default='GLM4', help='the prefix name of save dir (usually is the LLM name)')
parser.add_argument('--model_from_pretrained', default='<PATH>', help='the frozen embedding model pretrained path')
parser.add_argument('--batch_size', default=1024, help='batch size')

if not cmd_args:
    args = parser.parse_args([]) # You can directly set above parameters in the default.
else:
    args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.n_gpu)

model = SentenceTransformer(args.model_from_pretrained).to('cuda:0')

# %%
SOURCE_FILE = os.path.join(args.file_dir, args.file_name)
basename = os.path.basename(SOURCE_FILE)
SAVE_DIR = os.path.join(os.path.dirname(SOURCE_FILE), basename.split('.')[0] + f'_{args.save_type_name}_DA')
SAVE_FILE = os.path.join(SAVE_DIR,
                         basename.split('.')[0]+'_main_parts.tsv')
BATCH_SIZE = int(args.batch_size)

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# 
with open(SAVE_FILE, encoding='utf-8') as f:
    ori_data = f.readlines()

main_parts_kg = {}
results = []
documents = []
for item in tqdm(ori_data):
    item = json.loads(item)
    item = json_repair.loads(item)
    if type(item) != dict:
        results.append({"cls": ["news"], "subject": [], "action": [], "state": []})
        continue
    if 'cls' not in item or item['cls'] is None:
        item['cls'] = ['news']
    else:
        if type(item['cls']) == str:
            item['cls'] = [item['cls'].strip()]
    if 'subject' not in item:
        item['subject'] = []
    if 'action' not in item:
        item['action'] = []
    if 'state' not in item:
        item['state'] = []
    results.append(item)
    for subject in item['subject']:
        if type(subject) != dict or 'text' not in subject or subject['text'] in main_parts_kg:
            continue
        copy_subject = deepcopy(subject)
        copy_subject['data_type'] = 'subject'
        copy_subject['cls'] = item['cls']
        main_parts_kg[subject['text']] = copy_subject
        documents.append(subject['text'])
    for action in item['action']:
        if type(action) != dict or 'text' not in action or action['text'] in main_parts_kg:
            continue
        copy_action = deepcopy(action)
        copy_action['data_type'] = 'action'
        copy_action['cls'] = item['cls']
        main_parts_kg[action['text']] = copy_action
        documents.append(action['text'])
    for state in item['state']:
        if type(state) != dict or 'text' not in state or state['text'] in main_parts_kg:
            continue
        copy_state = deepcopy(state)
        copy_state['data_type'] = 'state'
        copy_state['cls'] = item['cls']
        main_parts_kg[state['text']] = copy_state
        documents.append(state['text'])

embeddings = model.encode(documents, batch_size=BATCH_SIZE, show_progress_bar=True)
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(embeddings)

def search_similar(query, k=5):
    if type(query) == str:
        query = [query]
    query_embedding = model.encode(query, batch_size=BATCH_SIZE)
    distances, indices = index.search(query_embedding, k)
    query_results = []
    for idxes, dises in zip(indices, distances):
        per_result = []
        for i, d in zip(idxes, dises):
            per_result.append((documents[i], round(float(d) * 0.05, 4)))
        query_results.append(per_result)
    return query_results

num_batches = len(main_parts_kg) // BATCH_SIZE + 1 if len(main_parts_kg) % BATCH_SIZE == 0 else len(main_parts_kg) // BATCH_SIZE
values = list(main_parts_kg.values())
for i in tqdm(range(num_batches)):
    batches = values[BATCH_SIZE * i : BATCH_SIZE * (i + 1)]
    queries = [item['text'] for item in batches]
    outputs = search_similar(queries)
    for j, out in enumerate(outputs):
        batches[j]['related'] = out
    

with open(os.path.join(SAVE_DIR,
                         basename.split('.')[0]+'_main_parts.jsonl'), mode='w+') as f:
    for item in results:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

with open(os.path.join(SAVE_DIR,
                         'main_part_kg.jsonl'), mode='w+') as f:
    for key in main_parts_kg:
        f.write(json.dumps({key: main_parts_kg[key]}, ensure_ascii=False) + '\n')

# %%
