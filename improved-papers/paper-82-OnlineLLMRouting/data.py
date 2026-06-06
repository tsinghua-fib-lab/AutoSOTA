import os
import pickle
import requests
import random, json
import numpy as np
import yaml
from datasets import load_dataset, Dataset, load_from_disk, concatenate_datasets
from transformers import AutoTokenizer
import pandas as pd

datsets =["sprout", "routerbench", "leaderboard"]


input_costs = [3.0, 0.5, 2.5, 0.15, 0.1, 0.2, 0.9, 0.2, 0.06, 0.06, 0.9, 3.5,0.6]
output_costs = [15.0, 1.5, 10.0, 0.6, 0.1, 0.2, 0.9, 0.2, 0.06, 0.06, 0.9, 3.5, 0.6] 
input_costs = [x / 1000000 for x in input_costs]
output_costs = [x / 1000000 for x in output_costs]

leader_cost = [
    0.8, 0.6, 1.2, 0.9, 1.2, 0.3,1.2, 0.9, 0.8, 0.3, 0.1, 0.3, 0.9, 0.2, 0.2, 0.2, 0.6, 0.9
]
leader_cost = [x / 1000000 for x in leader_cost]

class data():
    def __init__(self, name="",  models=None):
        if name not in datsets:
            raise Exception("dataset not found.")
        self.dir = "./data"
        self.models = models
        if name == datsets[0]:
            local_path = "./sprout_raw"
            if not os.path.exists(local_path):
                self.dataset = load_dataset("CARROT-LLM-Routing/SPROUT")
                self.dataset.save_to_disk(local_path)
            else:
                self.dataset = load_from_disk(local_path)
            self.n_train = len(self.dataset['train'])
            self.n_test = len(self.dataset['validation']) + len(self.dataset['test'])
            self.dataset = concatenate_datasets([self.dataset['train'], self.dataset['validation'], self.dataset['test']])
            self.op = 0
        elif name == datsets[1]:
            filename ="routerbench_0shot.pkl"
            local_path = f"{self.dir}/{filename}"

            with open(local_path, "rb") as f:
                self.dataset = pickle.load(f)

            self.dataset = Dataset.from_pandas(self.dataset)
            self.op = 1
        elif name == datsets[2]:
            if not os.path.exists("leaderboard"):
                local_path = "leaderboard_raw/data_QA.json"
                with open(local_path, "r") as f:
                    data_QA = json.load(f)
                with open("config.yaml", "r") as f:
                    ans = yaml.safe_load(f)
                benchmark = ans['OPEN_BENCHMARKS']
                raw_models = ans['RAW_MODELS']
                raw_models = [m.replace("open-llm-leaderboard/","").replace("__","/").replace("-details","") for m in raw_models]
                tokenizer_dict = {}
                for model in raw_models:
                    tokenizer_dict[model] = AutoTokenizer.from_pretrained(model, use_fast=False)

                for sce in data_QA.keys():
                    print(f"************** {sce} **************")
                    data_QA[sce]['tokens'] = np.array([[len(tokenizer_dict[model](q)['input_ids']) for q in data_QA[sce]['Ps']] for model in raw_models]).T
                
                bench = 'all'
                benchmark['all'] = benchmark['bbh']+benchmark['gpqa']+benchmark['math']+benchmark['mmlu_pro']+benchmark['musr']

                from functools import reduce
                data_Y = np.load("leaderboard_raw/new_leaderboard_processed_20241205.pickle", allow_pickle=True)
                M = [data_Y[k]['models'] for k in benchmark[bench]]
                M = np.sort(list(reduce(set.intersection, map(set, M)))).tolist()
                Y = [data_Y[k]['correctness'][[int(np.argmax(np.array(data_Y[k]['models'])==m)) for m in M]] for k in benchmark[bench]]
                Y = np.hstack(Y)
                data_Y[bench] = {}
                data_Y[bench]['correctness'] = Y.T
                data_Y[bench]['models'] = [m.replace("open-llm-leaderboard/","").replace("__","/").replace("-details","") for m in M]

                def flatten(xss):
                    return [x for xs in xss for x in xs]
                data_QA[bench] = {}
                data_QA[bench]['Qs'] = flatten([data_QA[k]['Qs'] for k in benchmark[bench]])
                data_QA[bench]['tokens'] = np.vstack([data_QA[k]['tokens'] for k in benchmark[bench]])

                records = []
                for q, token_row, correct_row in zip(
                    data_QA[bench]['Qs'],
                    data_QA[bench]['tokens'],
                    data_Y[bench]['correctness']
                ):
                    entry = {'prompt': q}
                    for idx, model in enumerate(data_Y[bench]['models']):
                        entry[f"{model}"] = correct_row[idx]  # correctness
                        entry[f"{model}|total_cost"] = token_row[idx] * leader_cost[idx] # cost (token length)
                    records.append(entry)
    
                df = pd.DataFrame(records)
                self.dataset = Dataset.from_pandas(df)
                indices = list(range(len(self.dataset)))
                self.dataset = self.dataset.add_column("index", indices)
                self.dataset.save_to_disk("leaderboard")
            else:
                self.dataset = load_from_disk("leaderboard")
            self.op = 2

    def split(self, e_num=10000, b_num=26497):
        if self.op == 0:
            if not os.path.exists(f"./sprout_data/"):
                def compute_field(example):
                    for idx, model in enumerate(self.models):
                        example[f"{model}|total_cost"] = input_costs[idx] * example[model]["num_input_tokens"] + output_costs[idx] * example[model]["num_output_tokens"]
                        example[model] = (example[model]["score"])
                    return example
                dataset = self.dataset.map(compute_field) 
                indices = list(range(len(dataset)))
                dataset = dataset.add_column("index", indices)
                dataset.save_to_disk(f"./sprout_data/")
            else:
                dataset = load_from_disk(f"./sprout_data/")

            # split 
            train = dataset.select(range(0, self.n_train))
            self.test = dataset.select(range(self.n_train, self.n_train + self.n_test))

            base_indices = list(range(len(train)))
            test_indices = list(range(len(self.test)))
            random.seed(42)
            sample_indices = random.sample(test_indices, k=e_num)
            rest_indices = random.sample(base_indices, k=b_num)

            sampled_dataset = self.test.select(sample_indices)
            rest_dataset = train.select(rest_indices)

            sample_indices = []
            rest_indices = []
            for item in sampled_dataset:
                sample_indices.append(item["index"])

            for item in rest_dataset:
                rest_indices.append(item["index"])
            
            sample_indices = np.array(sample_indices)
            rest_indices = np.array(rest_indices)
            
            cost = []
            for model in self.models:
                cost.append(np.sum(sampled_dataset[f"{model}|total_cost"]))
            min_cost = np.array(cost).min()
            
        elif self.op == 1:
            if not os.path.exists(f"./routerbench_data/"):
                def compute_field(example):
                    example["dataset"] = example["eval_name"]
                    return example
                dataset = self.dataset.map(compute_field) 
                indices = list(range(len(dataset)))
                dataset = dataset.add_column("index", indices)
                dataset.save_to_disk(f"./routerbench_data/")
            else:
                dataset = load_from_disk(f"./routerbench_data/")

            random.seed(42)
            all_indices = list(range(len(dataset)))
            sample_indices = random.sample(all_indices, k=10000)
            
            rest_indices = list(set(all_indices) - set(sample_indices))
            self.test = dataset.select(rest_indices)
            if e_num < 10000:
                sample_indices = random.sample(sample_indices, k=e_num)
            if b_num < 26497:
                rest_indices = random.sample(rest_indices, k=b_num)


            sampled_dataset = dataset.select(sample_indices)
            rest_dataset = dataset.select(rest_indices)
            
            cost = []
            for model in self.models:
                cost.append(np.sum(sampled_dataset[f"{model}|total_cost"]))
            min_cost = np.array(cost).min()

        elif self.op == 2:
            dataset = self.dataset
            random.seed(42)
            all_indices = list(range(len(dataset)))
            sample_indices = random.sample(all_indices, k=10000)
            
            rest_indices = list(set(all_indices) - set(sample_indices))
            self.test = dataset.select(rest_indices)
            if e_num < 10000:
                sample_indices = random.sample(sample_indices, k=e_num)
            if b_num < 11065:
                rest_indices = random.sample(rest_indices, k=b_num)

            sampled_dataset = dataset.select(sample_indices)
            rest_dataset = dataset.select(rest_indices)

            cost = []
            for model in self.models:
                cost.append(np.sum(sampled_dataset[f"{model}|total_cost"]))
            min_cost = np.array(cost).min()
        
        return sampled_dataset, rest_dataset, sample_indices, rest_indices, min_cost




