import argparse, json, os
import pandas as pd
from nltk.tokenize import sent_tokenize
from tqdm import tqdm
import random

# Inline probing attacks (no model dependencies)
random.seed(42)

def probing_insert(text, rate):
    sentences = sent_tokenize(text)
    num_inserts = int(rate * len(sentences))
    if num_inserts <= 0:
        return text
    insert_positions = random.sample(range(1, len(sentences)), min(num_inserts, len(sentences)-1))
    new_sentences = []
    for i in range(len(sentences)):
        new_sentences.append(sentences[i])
        if i in insert_positions:
            new_sentences.append(random.choice([' . ', ' ? ', ' ! ']))
    return ' '.join(new_sentences)

def probing_delete(text, rate):
    sentences = sent_tokenize(text)
    num_deletes = int(rate * len(sentences))
    if num_deletes >= len(sentences):
        num_deletes = len(sentences) - 1
    if num_deletes <= 0:
        return text
    delete_positions = set(random.sample(range(len(sentences)), num_deletes))
    new_sentences = [s for i, s in enumerate(sentences) if i not in delete_positions]
    return ' '.join(new_sentences) if new_sentences else sentences[-1]

def probing_reorder(text, rate):
    sentences = sent_tokenize(text)
    num_reorders = int(rate * len(sentences))
    if num_reorders >= len(sentences):
        num_reorders = len(sentences) - 1
    if num_reorders <= 0:
        return text
    reorder_positions = random.sample(range(len(sentences)), num_reorders)
    reorder_indices = list(range(len(sentences)))
    for pos in reorder_positions:
        if pos + 1 < len(reorder_indices):
            reorder_indices[pos], reorder_indices[pos+1] = reorder_indices[pos+1], reorder_indices[pos]
    return ' '.join([sentences[i] for i in reorder_indices])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--watermark_block_size', type=int, default=8)
    parser.add_argument('--dataset_name', type=str, default='c4')
    args = parser.parse_args()
    
    B = args.watermark_block_size
    D = args.dataset_name
    gen_dir = f'_result/generation/block_size_{B}'
    attack_dir = f'_result/attack/block_size_{B}'
    os.makedirs(attack_dir, exist_ok=True)
    
    gen_file = os.path.join(gen_dir, f'{D}_AliMark_facebook_opt-1.3b.json')
    attack_file = os.path.join(attack_dir, f'{D}_AliMark_facebook_opt-1.3b.json')
    
    df_gen = pd.read_json(gen_file, orient='index')
    print(f'Loaded {len(df_gen)} generation results')
    
    if os.path.exists(attack_file):
        df_attack = pd.read_json(attack_file, orient='index')
        print(f'Resuming from {len(df_attack)} attack results')
    else:
        df_attack = df_gen.copy()
    
    ATTACKS = {
        'insert_10': lambda t: probing_insert(t, 0.1),
        'insert_50': lambda t: probing_insert(t, 0.5),
        'delete_10': lambda t: probing_delete(t, 0.1),
        'delete_50': lambda t: probing_delete(t, 0.5),
        'reorder_30': lambda t: probing_reorder(t, 0.3),
        'reorder_70': lambda t: probing_reorder(t, 0.7),
    }
    
    for atk_name in ATTACKS:
        col = f'{atk_name}_result'
        if col not in df_attack.columns:
            df_attack[col] = None
    
    for idx, row in tqdm(df_attack.iterrows(), total=len(df_attack), desc='Probing attacks'):
        wm_text = row['watermarked_result']['text']
        for atk_name, atk_func in ATTACKS.items():
            col = f'{atk_name}_result'
            if row[col] is not None and not pd.isna(row[col]):
                continue
            attacked = atk_func(wm_text)
            df_attack.at[idx, col] = {'text': attacked}
            df_attack.to_json(attack_file, orient='index', indent=4)
    
    print(f'Done! Results saved to {attack_file}')
