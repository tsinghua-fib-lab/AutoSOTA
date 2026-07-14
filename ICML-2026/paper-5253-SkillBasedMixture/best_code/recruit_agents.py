import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ.pop("HF_ENDPOINT", None)
os.environ["HF_HOME"] = "/autosota_cache/hf"

from utils import *
from agent import agent_map
from collections import Counter
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
import glob
from tqdm import tqdm
import random
import math
import pandas as pd
from transformers import AutoConfig

def seed_everything(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
def find_similar_keywords(emb_model, target_keywords, available_keywords, top_k=1):
    
    target_embeddings = emb_model.encode(target_keywords)
    available_embeddings = emb_model.encode(available_keywords)

    results = []
    for idx, target in enumerate(target_keywords):
        # Calculate cosine similarity between target and all available keywords
        similarities = cosine_similarity(
            target_embeddings[idx].reshape(1, -1), 
            available_embeddings
        )[0]
        
        # Get indices of top_k most similar keywords
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # Store results with similarity scores
        keyword = available_keywords[top_indices[0]]
        results.append(keyword)
    return results

def normalize_to_probability(numbers, temperature=1.0):
    min_val = min(numbers)
    if min_val < 0:
        shifted = [x - min_val + 1 for x in numbers]
    else:
        shifted = numbers
    
    scaled = [x ** (1/temperature) for x in shifted]
    
    total = sum(scaled)
    probabilities = [x/total for x in scaled]
    
    return probabilities
    
def create_skill_rankings(all_agent_profiles, available_keywords):
    
    # Create dictionary with skills as keys
    skill_rankings = {}
    
    for skill in available_keywords:
        agent_scores = []
        for agent, skills in all_agent_profiles.items():
            if skill in skills:
                agent_scores.append((agent, skills[skill]))
        
        # Sort by score in descending order
        agent_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Store only the agent names in order
        skill_rankings[skill] = [(agent, score) for agent, score in agent_scores]
    
    return skill_rankings

def recruit_agents(models_scores, normalized_agent_strength, k=3):
    model_dict = dict(models_scores)
    positive_models = {model: score for model, score in model_dict.items() if score > 0}
    
    if positive_models:
        models = list(positive_models.keys())
        weights = [score * normalized_agent_strength[model] for model, score in positive_models.items()]
    else:
        min_score = min(model_dict.values())
        shifted_dict = {model: score - min_score + 1 for model, score in model_dict.items()}
        models = list(shifted_dict.keys())
        weights = [score * normalized_agent_strength[model] for model, score in shifted_dict.items()]
    
    total_weight = sum(weights)
    probabilities = [w/total_weight for w in weights]
    
    return random.choices(models, weights=probabilities, k=k)

def get_model_replacement(model_to_replace: str, model_counts: Counter, threshold: float = 0.05) -> str:
    
    total_count = sum(model_counts.values())
    
    # Calculate frequencies and identify frequent models
    frequencies = {model: count/total_count for model, count in model_counts.items()}
    frequent_models = {model: count for model, count in model_counts.items() 
                      if frequencies[model] >= threshold}
    
    if not frequent_models:
        raise ValueError("No models meet the frequency threshold!")

    frequent_models_total = sum(frequent_models.values())
    probabilities = [count/frequent_models_total for count in frequent_models.values()]
    frequent_model_names = list(frequent_models.keys())

    if model_to_replace in frequent_model_names:
        return model_to_replace
        
    # Sample replacement model
    replacement = random.choices(
        frequent_model_names,
        weights=probabilities,
        k=1
    )[0]
    
    return replacement
    
def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--task',
        type=str,
        default="MMLU"
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=0
    )
    return parser.parse_args()
    
if __name__ == "__main__":

    args = parse_args()
    seed_everything(args.seed)
    
    emb_model = SentenceTransformer('/autosota_cache/hf/models/sentence-transformers--all-MiniLM-L6-v2/snapshots/master')
    train_samples = read_json(f"./skills/{args.task}/train_samples_with_keywords_Qwen.json")
    
    mapped_file = f"./skills/{args.task}/test_samples_with_keywords_Qwen_mapped.json"
    original_file = f"./skills/{args.task}/test_samples_with_keywords_Qwen.json"
    need_mapping = False
    if os.path.exists(mapped_file):
        test_samples = read_json(mapped_file)
    else:
        test_samples = read_json(original_file)
        need_mapping = True
        
    print(f"\n\n=== recruiting agents for {args.task} ===\n\n")

    available_keywords = []
    for i in train_samples:
        available_keywords.extend(i['keywords'])
    keyword_freq = Counter(available_keywords)
    available_keywords = list(set(available_keywords))
    
    # map the keywords in the test sample into the keywords in model profiles
    if need_mapping:
        print("the keywords in the test samples have not been mapped, processing them now...")
        for sample in tqdm(test_samples):
            sample['keywords'] = find_similar_keywords(emb_model, sample['keywords'], available_keywords)
        write_json(test_samples, mapped_file)
        
    # create agent profiles
    all_agent_profiles = {}
    profiles = glob.glob(f"./skills/{args.task}/*_profile_*.json")
    model_names = [path.split('_profile')[0].split(f'./skills/{args.task}/')[-1] for path in profiles]
    
    for pf, mn in zip(profiles, model_names):
        all_agent_profiles[mn] = read_json(pf)
    
    global_agent_strength = [sum(list(all_agent_profiles[k].values())) for k in all_agent_profiles.keys()]
    result = normalize_to_probability(global_agent_strength, temperature=0.5)

    normalized_agent_strength = {}
    for agent, p in zip(all_agent_profiles.keys(), result):
        normalized_agent_strength[agent] = p

    skill_rankings = create_skill_rankings(all_agent_profiles, available_keywords)
    for i in range(len(test_samples)):
        solvers = []
        model_rank = {}
        test_samples[i]['solvers'] = []
        for kw in test_samples[i]['keywords']:
            for model, score in skill_rankings[kw]:
                if model not in model_rank:
                    model_rank[model] = score
                else:
                    model_rank[model] += score
        if not model_rank:
            print("Warning: empty keywords")
            test_samples[i]['solvers'].extend(random.choices(list(all_agent_profiles.keys()), k=3))
            test_samples[i]['aggregator'] = random.choices(list(all_agent_profiles.keys()), k=1)[0]
        else:
            model_rank = sorted(model_rank.items(), key=lambda x: x[1], reverse=True)
            test_samples[i]['solvers'].extend(recruit_agents(model_rank, normalized_agent_strength, k=3))

    test_samples_df = pd.DataFrame(test_samples)
    test_samples_df.to_csv(f"./skills/{args.task}/test_samples_with_keywords_and_solvers_seed{args.seed}.csv", index=False)
    print("done agent recruitment!")