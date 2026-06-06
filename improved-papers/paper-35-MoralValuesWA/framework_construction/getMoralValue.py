import json
import torch

tt = torch.cuda

def get_mfd(pair_path):
    with open(pair_path, 'r', encoding='utf-8') as file:
        word_pairs = json.load(file)
        cues = set(word_pairs.keys())

    dimension = ['care', 'fairness', 'loyalty', 'authority', 'sanctity']
    
    label_map = {
        1: ('care', 1),   # care.virtue
        2: ('care', -1),  # care.vice
        3: ('fairness', 1),   # fairness.virtue
        4: ('fairness', -1),  # fairness.vice
        5: ('loyalty', 1),   # loyalty.virtue
        6: ('loyalty', -1),  # loyalty.vice
        7: ('authority', 1),   # authority.virtue
        8: ('authority', -1),  # authority.vice
        9: ('sanctity', 1),   # sanctity.virtue
        10: ('sanctity', -1)  # sanctity.vice
    }
    
    word_dict = {}

    with open('./data/mfd2.txt', 'r') as file:
        lines = file.readlines()
    
    for line in lines[12:]:

        parts = line.strip().rsplit(maxsplit=1)
    
        word = parts[0]
        if len(word.split()) > 1:
            continue 
        if word not in cues:
            continue
        label_index = int(parts[1])

        # Get the dimension name and value from label_map
        dimension_name, value = label_map[label_index]
        
        # If the word already exists in the dictionary, update its dimension value
        if word in word_dict:
            word_dict[word][dimension_name] = value
        else:
            # If the word is not in the dictionary, create a new entry with the dimension dictionary
            word_dict[word] = {dim: 0 for dim in dimension}
            word_dict[word][dimension_name] = value
    
    return word_dict

def build_graph(pair_path):
    moral_scores = get_mfd(pair_path)
    with open(pair_path, 'r', encoding='utf-8') as file:
        word_pairs = json.load(file)
        cues = set(word_pairs.keys())
    for cue in cues:
        responses = list(word_pairs[cue])
        for response in responses:
            if response not in cues:
                del word_pairs[cue][response]
    
    vocab_dict = {word: idx for idx, word in enumerate(cues)}
    initial_matrix = [[0, 0, 0, 0, 0] for _ in range(len(vocab_dict))]
    for key, value in moral_scores.items():
        moral_word_id = vocab_dict[key]
        initial_matrix[moral_word_id] = [value['care'], value['fairness'], value['loyalty'], value['authority'], value['sanctity']]
    
    adjacency_matrix = [[0 for __ in range(len(vocab_dict))] for _ in range(len(vocab_dict))]

    for key, responses in word_pairs.items():
        cue_id = vocab_dict[key]
        for response in responses:
            response_id = vocab_dict[response]
            adjacency_matrix[cue_id][response_id] += word_pairs[key][response]
            adjacency_matrix[response_id][cue_id] += word_pairs[key][response]

    adjacency_matrix = tt.FloatTensor(adjacency_matrix)
    # Apply triple log1p transformation to edge weights for smoother weight distribution
    adjacency_matrix = torch.log1p(torch.log1p(torch.log1p(adjacency_matrix)))
    initial_matrix = tt.FloatTensor(initial_matrix)

    return vocab_dict, initial_matrix, adjacency_matrix

def normalize_trans_matrix(adjacency_matrix):
    out_degree_matrix = torch.diag(1 / torch.sqrt(adjacency_matrix.sum(dim=1)))
    in_degree_matrix = torch.diag(1 / torch.sqrt(adjacency_matrix.sum(dim=0)))
    trans_matrix = torch.chain_matmul(out_degree_matrix, adjacency_matrix, in_degree_matrix)
    return trans_matrix

def stereotype_propagation(pair_path, alpha = 0.4):
    vocab_dict, initial_matrix, adjacency_matrix = build_graph(pair_path)
    trans_matrix = normalize_trans_matrix(adjacency_matrix)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    identity_matrix = torch.eye(trans_matrix.size(0), device=device)
    
    # Six-step multi-scale propagation with optimized alpha schedule
    # information = prod_i (I - alphai * T)^{-1} * initial
    # Applied from innermost (alpha6=0.25) to outermost (alpha1=alpha)
    alpha_schedule = [0.82, 0.80, 0.68, 0.49, 0.25]  # a2..a6
    
    # Apply in reverse order (innermost first)
    info = initial_matrix.clone()
    for a in reversed(alpha_schedule):
        info = torch.matmul(torch.inverse(identity_matrix - a * trans_matrix), info)
    # Apply outermost (alpha = a1)
    info = torch.matmul(torch.inverse(identity_matrix - alpha * trans_matrix), info)
    return vocab_dict, info

def process_and_save_moral_lexicon(input_path, output_path, given_alpha):
    vocab_dict, information_matrix = stereotype_propagation(input_path, given_alpha)
    moral_lexicon = {
        word: information_matrix[idx].tolist() for word, idx in vocab_dict.items()
    }

    with open(output_path, 'w', encoding='utf-8') as f_moral_lexicon:
        json.dump(moral_lexicon, f_moral_lexicon, ensure_ascii=False, indent=4)
    print('Saved moral lexicon to: ' + output_path)

if __name__ == '__main__':
    # Paths to input and output files
    paths = [
        {
            'input': './data/human_association.json',
            'output': './data/human_moral.json',
            'alpha': 0.75
        },
        {
            'input': './data/llama_2.1_association.json',
            'output': './data/llama_2.1_moral.json',
            'alpha': 0.84
        }
    ]

    # Process and save moral lexicons for both human and LLM data
    for path_info in paths:
        process_and_save_moral_lexicon(
            input_path=path_info['input'],
            output_path=path_info['output'],
            given_alpha=path_info['alpha']
        )
