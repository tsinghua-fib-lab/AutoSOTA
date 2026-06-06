# %%
import os
import json
import json_repair
from tqdm import tqdm
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
                         basename.split('.')[0]+'_entities.tsv')
BATCH_SIZE = int(args.batch_size)

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# 
with open(SAVE_FILE, encoding='utf-8') as f:
    ori_data = f.readlines()

entities = {}
results = []
documents = {}
for item in tqdm(ori_data):
    item = json.loads(item)
    item = json_repair.loads(item)
    if type(item) != list:
        results.append([])
        continue
    results.append(item)
    for entity_item in item:
        if type(entity_item) != dict or 'entity' not in entity_item or 'type' not in entity_item:
            continue
        entity, entity_type = entity_item['entity'], entity_item['type']
        entity = entity.strip()
        entity_type = entity_type.strip()
        if entity_type not in entities:
            entities[entity_type] = []
            documents[entity_type] = {}
        if entity not in documents[entity_type]:
            documents[entity_type][entity] = 1
            entities[entity_type].append({'text': entity})

def search_similar(query, documents, index, k=10):
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

total = 0
for key in entities:
    total += len(entities[key])
progress = tqdm(total=total)
for key in entities:
    docs = list(documents[key])
    embeddings = model.encode(docs, batch_size=BATCH_SIZE, show_progress_bar=True)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    cur_entities = entities[key]
    num_batches = len(cur_entities) // BATCH_SIZE + 1 if len(cur_entities) % BATCH_SIZE == 0 else len(cur_entities) // BATCH_SIZE
    for i in range(num_batches):
        batches = cur_entities[BATCH_SIZE * i : BATCH_SIZE * (i + 1)]
        queries = [item['text'] for item in batches]
        outputs = search_similar(queries, docs, index)
        for j, out in enumerate(outputs):
            batches[j]['related'] = out
        progress.update(len(batches))
progress.close()

with open(os.path.join(SAVE_DIR,
                         basename.split('.')[0]+'_entities.jsonl'), mode='w+') as f:
    for item in results:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

with open(os.path.join(os.path.dirname(SAVE_FILE), 'entity_kg.jsonl'), mode='w+') as f:
    for key in entities:
        f.write(json.dumps({key: entities[key]}, ensure_ascii=False) + '\n')

# %%
