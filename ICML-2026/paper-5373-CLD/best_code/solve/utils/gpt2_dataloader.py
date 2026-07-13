'''
dataloader for gpt2 extracted positive-negative features
if adding a new model, just add a new model_path
this version dynamically gets num_batches
'''


import numpy as np
import jax.numpy as jnp
from os.path import dirname, join, abspath, basename
import os
import re
import glob

def load_data(model_name, data_seed, caller_script=None):
    np.random.seed(data_seed) # seed for train-test split
    
    # Define a mapping of model names to their paths
    model_paths = {
        # 'gpt2_imdb_trained': "/home/--/CVXDPO/extracted_features_attn_NEG_POS_checkpoint_gpt2_e1_imdb",
        # 'gpt2_attn_ultra': "/home/--/CVXDPO/extracted_features_attn_NEG_POS_SFT_openai-community_gpt2_ultra",
        # 'gpt2_attn_edu': "/home/--/CVXDPO/extracted_features/extracted_features_attn_NEG_POS_SFT_openai-community_gpt2_edu",
        # 'gpt2_lmhead_commune': join(dirname(abspath('content')), 'datasets', 'gpt2lmhead_commu'),

        # SFT base trained model paths - batchsize64
        'dolphin_imdb_sft': "/home/--/CVXDPO/extracted_features/batchsize64/extracted_features_attn_NEG_POS_SFT_dolphin-2.1-7b_imdb/",
        'dolphin_edu_sft': "/home/--/CVXDPO/extracted_features/batchsize64/extracted_features_attn_NEG_POS_SFT_dolphin2.1-7B_edu",
        'dolphin_ultra_sft': "/home/--/CVXDPO/extracted_features/batchsize64/extracted_features_attn_NEG_POS_SFT_dolphin2.1-7B_ultra",
        'llama_edu_sft': "/home/--/CVXDPO/extracted_features/batchsize64/extracted_features_attn_NEG_POS_SFT_meta-llama_Llama-3.1-8B_edu",
        'llama_imdb_sft': "/home/--/CVXDPO/extracted_features/batchsize64/extracted_features_attn_NEG_POS_SFT_meta-llama_Llama-3.1-8B_imdb",
        'llama_ultra_sft': "/home/--/CVXDPO/extracted_features/batchsize64/extracted_features_attn_NEG_POS_SFT_meta-llama_Llama-3.1-8B_ultra",
        'mistral_edu_sft': "/home/--/CVXDPO/extracted_features/batchsize64/extracted_features_attn_NEG_POS_SFT_mistralai_Mistral-7B-v0.1_edu",
        'mistral_imdb_sft': "/home/--/CVXDPO/extracted_features/batchsize64/extracted_features_attn_NEG_POS_SFT_mistralai_Mistral-7B-v0.1_imdb",
        'mistral_ultra_sft': "/home/--/CVXDPO/extracted_features/batchsize64/extracted_features_attn_NEG_POS_SFT_mistralai_Mistral-7B-v0.1_ultra",

        # batchsize150
        'distilgpt2_edu_sft': "/home/--/CVXDPO/extracted_features/batchsize150/extracted_features_attn_NEG_POS_SFT_distilbert_distilgpt2_edu_edu/",
        'distilgpt2_imdb_sft': "/home/--/CVXDPO/extracted_features/batchsize150/extracted_features_attn_NEG_POS_SFT_distilbert_distilgpt2_imdb_all/",
        'distilgpt2_ultra_sft': "/home/--/CVXDPO/extracted_features/batchsize150/extracted_features_attn_NEG_POS_SFT_distilbert_distilgpt2_ultra",
        'gpt2_edu_sft': "/home/--/CVXDPO/extracted_features/batchsize150/extracted_features_attn_NEG_POS_SFT_openai-community_gpt2_edu",
        'gpt2_imdb_sft': "/home/--/CVXDPO/extracted_features/batchsize150/extracted_features_attn_NEG_POS_SFT_openai-community_gpt2_imdb_all/",
        'gpt2_ultra_sft': "/home/--/CVXDPO/extracted_features/batchsize150/extracted_features_attn_NEG_POS_SFT_openai-community_gpt2_ultra",


        # No SFT base trained model paths - batchsize64 or batchsize8
        'dolphin_imdb': "/home/--/CVXDPO/extracted_features0504/batchsize64/extracted_features_attn_NEG_POS_dolphin-2.1-mistral-7b_imdb",
        'dolphin_edu': "/home/--/CVXDPO/extracted_features0504/batchsize64/extracted_features_attn_NEG_POS_dolphin-2.1-mistral-7b_edu",
        'dolphin_ultra': "/home/--/CVXDPO/extracted_features0504/batchsize64/extracted_features_attn_NEG_POS_dolphin-2.1-mistral-7b_ultra",
        'llama_edu': "/home/--/CVXDPO/extracted_features0504/batchsize8/extracted_features_attn_NEG_POS_Llama-3.1-8B_edu",
        'llama_imdb': "/home/--/CVXDPO/extracted_features0504/batchsize8/extracted_features_attn_NEG_POS_Llama-3.1-8B_imdb/",
        'llama_ultra': "/home/--/CVXDPO/extracted_features0504/batchsize8/extracted_features_attn_NEG_POS_Llama-3.1-8B_ultra",
        'mistral_edu': "/home/--/CVXDPO/extracted_features0504/batchsize64/extracted_features_attn_NEG_POS_Mistral-7B-v0.1_edu",
        'mistral_imdb': "/home/--/CVXDPO/extracted_features0504/batchsize64/extracted_features_attn_NEG_POS_Mistral-7B-v0.1_imdb",
        'mistral_ultra': "/home/--/CVXDPO/extracted_features0504/batchsize64/extracted_features_attn_NEG_POS_Mistral-7B-v0.1_ultra",

        # batchsize150
        'distilgpt2_edu': "/home/--/CVXDPO/extracted_features0504/batchsize150/extracted_features_attn_NEG_POS_distilgpt2_edu/",
        'distilgpt2_imdb': "/home/--/CVXDPO/extracted_features0504/batchsize150/extracted_features_attn_NEG_POS_distilgpt2_all",
        'distilgpt2_ultra': "/home/--/CVXDPO/extracted_features0504/batchsize150/extracted_features_attn_NEG_POS_distilgpt2_ultra",
        'gpt2_edu': "/home/--/CVXDPO/extracted_features0504/batchsize150/extracted_features_attn_NEG_POS_gpt2_edu/",
        'gpt2_imdb': "/home/--/CVXDPO/extracted_features0504/batchsize150/extracted_features_attn_NEG_POS_gpt2_all",
        'gpt2_ultra': "/home/--/CVXDPO/extracted_features0504/batchsize150/extracted_features_attn_NEG_POS_gpt2_ultra/",
    }
    
    # Define file name patterns for different models
    file_patterns = {
        'gpt2_lmhead_commune': ('POSlast_hidden_states_gpt2commu_lmhead_', 'NEGlast_hidden_states_gpt2commu_lmhead_'),
        # Default pattern for most models
        'default': ('POSlast_hidden_states_', 'NEGlast_hidden_states_')
    }
    
    if model_name not in model_paths:
        raise ValueError(f"Unknown model name: {model_name}")
    
    path = model_paths[model_name]
    print(f'---Loading dataset for model "{model_name}" from path: {path}---')
    
    # Check if this path is under the batchsize64 directory
    is_batchsize64 = 'batchsize64' in path
    is_batchsize150 = 'batchsize150' in path
    is_batchsize8 = 'batchsize8' in path

    # get file name patterns for this model
    pos_pattern, neg_pattern = file_patterns.get(model_name, file_patterns['default'])
    
    # Determine if we're dealing with pooled data (no sequence dimension)
    is_pooled = any(pooled_name in model_name for pooled_name in 
                   ['meanpooled', 'attn_', 'lmhead'])
    
    # Load data
    dats_training = []
    labels_training = []
    print('Loading data')
    
    if is_batchsize64 or is_batchsize150 or is_batchsize8:

        # For batchsize64, batchsize150, and batchsize8, files typically have proc IDs
        print("Detected batchsize64 path format - looking for files with processor IDs")
        
        # Find all positive files using glob pattern
        pos_files = glob.glob(join(path, f"{pos_pattern}*_proc*.npy"))
        neg_files = glob.glob(join(path, f"{neg_pattern}*_proc*.npy"))
        
        print(f"Found {len(pos_files)} positive files and {len(neg_files)} negative files")
        
        # Load positive files
        for pos_file in pos_files:
            try:
                with open(pos_file, 'rb') as ftrain:
                    X = np.load(ftrain)
                    dats_training.append(X)
                    labels_training.append(np.ones(X.shape[0]))
            except Exception as e:
                print(f"Error loading {pos_file}: {e}")
        
        # Load negative files
        for neg_file in neg_files:
            try:
                with open(neg_file, 'rb') as ftrain:
                    X = np.load(ftrain)
                    dats_training.append(X)
                    labels_training.append(-1 * np.ones(X.shape[0]))
            except Exception as e:
                print(f"Error loading {neg_file}: {e}")
    
    else:
        # Standard file naming without processor IDs OLD
        # Dynamically determine the number of batches by counting POS files
        print("----------------------No Processor IDs found, are you sure?----------------------")
        all_files = os.listdir(path)
        pos_files = [f for f in all_files if f.startswith(pos_pattern)]
        
        # Get the number of batches
        num_batches = len(pos_files) + 1  # +1 because batch numbering starts at 1
        
        print(f"Found {len(pos_files)} positive batch files, setting num_batches to {num_batches}")
        
        for i in range(1, num_batches):
            training_file_name_pos = f'{pos_pattern}{i}.npy'
            training_file_name_neg = f'{neg_pattern}{i}.npy'
            
            try:
                with open(join(path, training_file_name_pos), 'rb') as ftrain:
                    X = np.load(ftrain)
                    dats_training.append(X)
                    labels_training.append(np.ones(X.shape[0]))
                    
                with open(join(path, training_file_name_neg), 'rb') as ftrain:
                    X = np.load(ftrain)
                    dats_training.append(X)
                    labels_training.append(-1 * np.ones(X.shape[0]))
            except FileNotFoundError as e:
                print(f"Warning: Could not find file at {i}: {e}")
    
    if not dats_training:
        raise ValueError(f"No data files loaded for model {model_name}")
    
    # Concatenate all data
    A = jnp.concatenate([jnp.array(dat_training) for dat_training in dats_training], axis=0)
    y = jnp.concatenate([jnp.array(labels) for labels in labels_training], axis=0)
    print('Finished loading data!')
    
    # Reshape based on the actual number of dimensions in the data
    shape_A = A.shape
    print(f"Original data shape: {shape_A}")
    
    # Check if data is 3D (batch_size, seq_len, hidden_dim) or already 2D (batch_size, hidden_dim)
    # This is due to pooling
    if len(shape_A) == 3:
        # Non-pooled data is 3D, so flatten seq_len and hidden_dim
        A = A.reshape(shape_A[0], shape_A[1] * shape_A[2])
        print(f"Data is 3D (non-pooled), reshaping to {A.shape}")
    else:
        # Data is already 2D, no need to reshape
        print(f"Data is already 2D (pooled), shape: {A.shape}")
    
    # Shuffle and split 
    n = shape_A[0]
    J = np.random.permutation(n)
    A = A[J]
    y = y[J]

    
    if caller_script == "defrun":
        # Only return top 70% of the data for training the convex model (Stage 1)
        split_idx = int(0.9 * n)
        print(f"Returning top 70% ({split_idx} samples) for cronos_trainer")
        
        # Shuffle and split for defrun
        J = np.random.permutation(n)
        A = A[J]
        y = y[J]
        
        # For training phase
        ntr = int(0.8 * split_idx)  # 80% of the 70% for training
        ntst = split_idx - ntr      # 20% of the 70% for testing
        
        Atr = A[:ntr]
        Atst = A[ntr:split_idx]
        ytr = y[:ntr]
        ytst = y[ntr:split_idx]
    
    elif caller_script == "finetune":
        # FIXED: Return 30% of each class separately to maintain balance
        pos_indices = np.where(y == 1)[0]
        neg_indices = np.where(y == -1)[0]
        
        print(f"Found {len(pos_indices)} positive samples and {len(neg_indices)} negative samples")
        
        # Take 30% from each class
        pos_30_percent = int(len(pos_indices) * 0.3)
        neg_30_percent = int(len(neg_indices) * 0.3)
        
        print(f"Selecting 30% from each class: {pos_30_percent} positive, {neg_30_percent} negative")
        
        # Randomly select 30% from each class
        pos_selected_indices = np.random.choice(pos_indices, pos_30_percent, replace=False)
        neg_selected_indices = np.random.choice(neg_indices, neg_30_percent, replace=False)
        
        # Combine the selected indices
        selected_indices = np.concatenate([pos_selected_indices, neg_selected_indices])
        
        # Extract the balanced subset
        finetune_data = A[selected_indices]
        finetune_labels = y[selected_indices]
        
        total_finetune_samples = len(selected_indices)
        print(f"Returning balanced 30% ({total_finetune_samples} samples: {pos_30_percent} pos + {neg_30_percent} neg) for finetuning cvx")
        
        # Shuffle the selected data
        finetune_permutation = np.random.permutation(total_finetune_samples)
        finetune_data = finetune_data[finetune_permutation]
        finetune_labels = finetune_labels[finetune_permutation]
        
        # For fine-tuning phase: split into 80% train, 20% test
        ntr = int(0.8 * total_finetune_samples)
        ntst = total_finetune_samples - ntr
        
        Atr = finetune_data[:ntr]
        Atst = finetune_data[ntr:]
        ytr = finetune_labels[:ntr]
        ytst = finetune_labels[ntr:]
        
        # Verify balance is maintained
        pos_train = np.sum(ytr == 1)
        neg_train = np.sum(ytr == -1)
        pos_test = np.sum(ytst == 1)
        neg_test = np.sum(ytst == -1)
        print(f"Train split: {pos_train} positive, {neg_train} negative")
        print(f"Test split: {pos_test} positive, {neg_test} negative")
    
    else:
        # Default behavior (standard 80/20 split of all data)
        print("You've chosen default 80/20 behavior for splitting data.")
        
        # Shuffle for default case
        J = np.random.permutation(n)
        A = A[J]
        y = y[J]
        
        ntr = int(0.8 * n)
        ntst = n - ntr
        Atr = A[:ntr]
        Atst = A[ntr:]
        ytr = y[:ntr]
        ytst = y[ntr:]
    
    del A, y
    
    return Atr, ytr, Atst, ytst, ntr, ntst
    
    #  # Split based on caller script
    # if caller_script == "defrun":
    #     # Only return top 70% of the data for training the convex model (Stage 1)
    #     split_idx = int(0.9 * n)
    #     print(f"Returning top 70% ({split_idx} samples) for cronos_trainer")
        
    #     # For training phase
    #     ntr = int(0.8 * split_idx)  # 80% of the 70% for training
    #     ntst = split_idx - ntr      # 20% of the 70% for testing
        
    #     Atr = A[:ntr]
    #     Atst = A[ntr:split_idx]
    #     ytr = y[:ntr]
    #     ytst = y[ntr:split_idx]
    
    # elif caller_script == "finetune":
    #     # Only return bottom 30% of the data for fine-tuning (Stage 2)
    #     split_idx = int(0.9 * n)
    #     finetune_data = A[split_idx:]
    #     finetune_labels = y[split_idx:]
    #     print(f"Returning bottom 30% ({n - split_idx} samples) for finetuning cvx")
        
    #     # For fine-tuning phase
    #     ntr = int(0.8 * (n - split_idx))  # 80% of the 30% for training
    #     ntst = (n - split_idx) - ntr       # 20% of the 30% for testing
        
    #     Atr = finetune_data[:ntr]
    #     Atst = finetune_data[ntr:]
    #     ytr = finetune_labels[:ntr]
    #     ytst = finetune_labels[ntr:]
    
    # else:
    #     # Default behavior (standard 80/20 split of all data)
    #     print("You've chosen default 80/20 behavior for splitting data.")
    #     ntr = int(0.8 * n)
    #     ntst = n - ntr
    #     Atr = A[:ntr]
    #     Atst = A[ntr:]
    #     ytr = y[:ntr]
    #     ytst = y[ntr:]
    
    # del A, y
    
    # return Atr, ytr, Atst, ytst, ntr, ntst


# import numpy as np
# import jax.numpy as jnp
# from os.path import dirname, join, abspath
# import os

# def load_data(model_name, data_seed):
#     np.random.seed(data_seed) # seed for train-test split
    
#     # Define a mapping of model names to their paths
#     model_paths = {
#         'gpt2_imdb_trained': "/home/--/CVXDPO/extracted_features_attn_NEG_POS_checkpoint_gpt2_e1_imdb",
#         'gpt2_attn_ultra': "/home/--/CVXDPO/extracted_features_attn_NEG_POS_SFT_openai-community_gpt2_ultra",
#         'gpt2_attn_edu': "/home/--/CVXDPO/extracted_features/extracted_features_attn_NEG_POS_SFT_openai-community_gpt2_edu",
#         'gpt2_lmhead_commune': join(dirname(abspath('content')), 'datasets', 'gpt2lmhead_commu'),

#         # New model paths — batchsize24
#         'dolphin_imdb': "/home/--/CVXDPO/extracted_features/batchsize24/extracted_features_attn_NEG_POS_SFT_dolphin-2.1-7b_imdb",
#         'dolphin_edu': "/home/--/CVXDPO/extracted_features/batchsize24/extracted_features_attn_NEG_POS_SFT_dolphin2.1-7B_edu",
#         'dolphin_ultra': "/home/--/CVXDPO/extracted_features/batchsize24/extracted_features_attn_NEG_POS_SFT_dolphin2.1-7B_ultra",
#         'llama_edu': "/home/--/CVXDPO/extracted_features/batchsize24/extracted_features_attn_NEG_POS_SFT_meta-llama_Llama-3.1-8B_edu",
#         'llama_imdb': "/home/--/CVXDPO/extracted_features/batchsize24/extracted_features_attn_NEG_POS_SFT_meta-llama_Llama-3.1-8B_imdb",
#         'llama_ultra': "/home/--/CVXDPO/extracted_features/batchsize24/extracted_features_attn_NEG_POS_SFT_meta-llama_Llama-3.1-8B_ultra",
#         'mistral_edu': "/home/--/CVXDPO/extracted_features/batchsize24/extracted_features_attn_NEG_POS_SFT_mistralai_Mistral-7B-v0.1_edu",
#         'mistral_imdb': "/home/--/CVXDPO/extracted_features/batchsize24/extracted_features_attn_NEG_POS_SFT_mistralai_Mistral-7B-v0.1_imdb",
#         'mistral_ultra': "/home/--/CVXDPO/extracted_features/batchsize24/extracted_features_attn_NEG_POS_SFT_mistralai_Mistral-7B-v0.1_ultra",

#         # New model paths — batchsize150
#         'distilgpt2_edu': "/home/--/CVXDPO/extracted_features/batchsize150/extracted_features_attn_NEG_POS_SFT_distilbert_distilgpt2_edu",
#         'distilgpt2_imdb': "/home/--/CVXDPO/extracted_features/batchsize150/extracted_features_attn_NEG_POS_SFT_distilbert_distilgpt2_imdb",
#         'distilgpt2_ultra': "/home/--/CVXDPO/extracted_features/batchsize150/extracted_features_attn_NEG_POS_SFT_distilbert_distilgpt2_ultra",
#         'gpt2_edu': "/home/--/CVXDPO/extracted_features/batchsize150/extracted_features_attn_NEG_POS_SFT_openai-community_gpt2_edu",
#         'gpt2_imdb': "/home/--/CVXDPO/extracted_features/batchsize150/extracted_features_attn_NEG_POS_SFT_openai-community_gpt2_imdb",
#         'gpt2_ultra': "/home/--/CVXDPO/extracted_features/batchsize150/extracted_features_attn_NEG_POS_SFT_openai-community_gpt2_ultra",
#     }

    
#     # Define file name patterns for different models
#     file_patterns = {
#         #'gpt2_no_train': ('Poslast_hidden_states_gpt2_', 'NEGlast_hidden_states_gpt2_'),
#         #'gpt2_notrain_medium': ('Poslast_hidden_states_gpt2_large', 'NEGlast_hidden_states_gpt2_large'),
#         'gpt2_lmhead_commune': ('POSlast_hidden_states_gpt2commu_lmhead_', 'NEGlast_hidden_states_gpt2commu_lmhead_'),
#         # Default pattern for most models
#         'default': ('POSlast_hidden_states_', 'NEGlast_hidden_states_')
#     }
    
#     if model_name not in model_paths:
#         raise ValueError(f"Unknown model name: {model_name}")
    
#     path = model_paths[model_name]
#     print(f'---Loading dataset for model "{model_name}" from path: {path}---')
    
#     # get file name patterns for this model
#     pos_pattern, neg_pattern = file_patterns.get(model_name, file_patterns['default'])
    
#     # Determine if we're dealing with pooled data (no sequence dimension)
#     is_pooled = any(pooled_name in model_name for pooled_name in 
#                    ['meanpooled', 'attn_', 'lmhead'])
    
#     # Dynamically determine the number of batches by counting POS files
#     all_files = os.listdir(path)
#     pos_files = [f for f in all_files if f.startswith(pos_pattern)]
    
#     # Get the number of batches - assume batch numbering starts at 1
#     num_batches = len(pos_files) + 1  # +1 because batch numbering starts at 1
    
#     print(f"Found {len(pos_files)} positive batch files, setting num_batches to {num_batches}")
    

#     # Load data
#     dats_training = []
#     labels_training = []
#     print('Loading data')
    
#     for i in range(1, num_batches):
#         training_file_name_pos = f'{pos_pattern}{i}.npy'
#         training_file_name_neg = f'{neg_pattern}{i}.npy'
        
#         try:
#             with open(join(path, training_file_name_pos), 'rb') as ftrain:
#                 X = np.load(ftrain)
#                 dats_training += [X]
#                 labels_training += [np.ones(X.shape[0])]
                
#             with open(join(path, training_file_name_neg), 'rb') as ftrain:
#                 X = np.load(ftrain)
#                 dats_training += [X]
#                 labels_training += [-1*np.ones(X.shape[0])]
#         except FileNotFoundError as e:
#             print(f"Warning: Could not find file at {i}: {e}")
    

#     A = jnp.concatenate([jnp.array(dat_training) for dat_training in dats_training], axis=0)
#     y = jnp.concatenate([jnp.array(labels) for labels in labels_training], axis=0)
#     print('Finished loading data!')
    
#     # Reshape based on the actual number of dimensions in the data
#     shape_A = A.shape
#     print(f"Original data shape: {shape_A}")
    
#     # Check if data is 3D (batch_size, seq_len, hidden_dim) or already 2D (batch_size, hidden_dim)
#     # This is due to pooling
#     if len(shape_A) == 3:
#         # Non-pooled data is 3D, so flatten seq_len and hidden_dim
#         A = A.reshape(shape_A[0], shape_A[1] * shape_A[2])
#         print(f"Data is 3D (non-pooled), reshaping to {A.shape}")
#     else:
#         # Data is already 2D, no need to reshape
#         print(f"Data is already 2D (pooled), shape: {A.shape}")
    
#     # Shuffle and split 
#     n = shape_A[0]
#     J = np.random.permutation(n)
#     A = A[J]
#     y = y[J]
    
#     ntr = np.int64(0.8*n)
#     ntst = n-ntr
#     Atr = A[:ntr]
#     Atst = A[ntr+1:]
#     ytr = y[:ntr]
#     ytst = y[ntr+1:]
    
#     del A, y
    
#     return Atr, ytr, Atst, ytst, ntr, ntst