import torch

def get_preferred_dtype_per_model(model_name):
    model_lower = model_name.lower()
    if 'llama' in model_lower:  # tested with llama3-8B, llama3.1-8B, llama3.2-1/2B
        return torch.float16
    elif 'mistral' in model_lower and '-v0.' in model_lower:
        return torch.float16
        # return torch.bfloat16
    elif 'qwen2.5' in model_lower:
        return torch.bfloat16
    else:
        raise ValueError(f'Can not find appropiate dtype for model name: {model_name} (suggest to use default option, but unsupported yet)')
    

def get_partiotion_ngram(base_model, partiotion_ngram=" ###"):
    base_model_lower = base_model.lower()
    # partiotion_ngram = " ###"  # works for gpt2, llama3,3.1,3.2, qwen2.5
    if 'mistral' in base_model_lower and '-v0.' in base_model_lower:  # tested Mistral-7B-v0.1
        # partiotion_ngram = "###"
        partiotion_ngram = partiotion_ngram.strip()
    
    return partiotion_ngram