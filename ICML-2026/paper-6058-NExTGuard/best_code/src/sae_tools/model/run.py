from tqdm import tqdm
from typing import Dict, Any, List, Tuple
import torch
from .formatting import format_with_tokenizer

def process_batch(
    tokenizer, 
    raw_batch: Dict[str, List[Any]], 
    data_type: str = "prompt"
) -> Tuple[List[str], List[Tuple[int, int]]]:
    """
    Convert the original batch data to a list of formatted text and start indices that can be input to the model.
    """
    assert data_type in ["prompt", "response"], "Invalid data type"
    
    prompts = raw_batch["prompt"]
    responses = raw_batch["response"] if (data_type == "response") else None

    batch_texts: List[str] = []
    valid_token_indices: List[Tuple[int, int]] = []
    
    for idx, prompt in enumerate(prompts):
        response = responses[idx] if responses is not None else None
        full_text, _, valid_idx = format_with_tokenizer(
            tokenizer,
            prompt,
            response
        )
        batch_texts.append(full_text)
        valid_token_indices.append(valid_idx)
    
    return batch_texts, valid_token_indices

def filter_activations_by_norm(
    acts_BLD: torch.Tensor,
    encoded_acts_BLF: torch.Tensor
) -> torch.Tensor:
    norms_BL = acts_BLD.norm(dim=-1)
    median_norm = norms_BL.median()
    norm_mask_BL = norms_BL > (median_norm * 10)
    encoded_acts_BLF = encoded_acts_BLF * ~norm_mask_BL[:, :, None]
    encoded_acts_BLF[:,0,:] = 0
    return encoded_acts_BLF
    
def generate_activations_batch(
    tokenizer,
    model,
    sae,
    layer: int,
    data: Dict[str, Any],
    data_type: str = "prompt"
) -> Tuple[torch.sparse_coo_tensor, torch.Tensor]:
    """
    Full inference, return the activations in sparse matrix format.
    """
    hook_name = f"blocks.{layer}.hook_resid_post"
    
    with torch.no_grad():
        batch_texts, valid_token_indices = process_batch(
            tokenizer=tokenizer,
            raw_batch=data,
            data_type=data_type
        )
        _, cache = model.run_with_cache(batch_texts, names_filter=hook_name)

        # acts_BLD Shape: [Batch, Length, d_model]
        acts_BLD = cache[hook_name]

        # encoded_acts_BLF Shape: [Batch, Length, num_features]
        encoded_acts_BLF = sae.encode(acts_BLD)
        encoded_acts_BLF = filter_activations_by_norm(acts_BLD, encoded_acts_BLF)

    valid_token_idx = torch.tensor(
        valid_token_indices,
        dtype=torch.int64,
        device=encoded_acts_BLF.device
    )

    # Shape: [Batch, Length, num_features], [Batch, 2]
    return encoded_acts_BLF, valid_token_idx

def generate_activations(
    tokenizer,
    model,
    sae,
    layer: int,
    dataset,
    data_type: str = "prompt",
    batch_size: int = 1
) -> Dict[str, Any]:
    """
    Batch generate sparse activations, and concatenate the results into a huge sparse tensor.
    
    Args:
        dataset: dataset object (supports slicing, e.g. HuggingFace Dataset)
        data_type: data type, "prompt" or "response"
        batch_size: batch size
    
    Returns:
        Dict[str, Any]: dictionary containing the following keys:
            - "sparse_acts": torch.Tensor (Sparse COO)
            Shape: [Total_Samples, Length, num_features]
            - "valid_token_idx": torch.Tensor (Int) or None
            Shape: [Total_Samples]
            - "seq_lens": torch.Tensor (Int)
            Shape: [Total_Samples]
            - "shape": torch.Size
    """
    total_samples = len(dataset)
    all_sparse_acts = []
    all_valid_token_idx = []
    all_seq_lens = []
    
    for i in tqdm(range(0, total_samples, batch_size), total=total_samples // batch_size):
        
        current_batch_indices = range(i, min(i + batch_size, total_samples))
        
        # [Batch, Length, num_features], [Batch, 2]
        encoded_acts_BLF, valid_token_idx = generate_activations_batch(
            tokenizer=tokenizer,
            model=model,
            sae=sae,
            layer=layer,
            data=dataset.select(current_batch_indices), 
            data_type=data_type
        )
        B, L, F = encoded_acts_BLF.shape
        for _ in range(B):
            all_seq_lens.append(L)
        # [Batch, Length, Feature] -> [Batch*Length, Feature]
        flat_act = encoded_acts_BLF.squeeze(0)
        sparse_acts = flat_act.to_sparse()

        all_sparse_acts.append(sparse_acts.cpu())
        all_valid_token_idx.append(valid_token_idx.cpu())
    
    print("Concatenating batches...")

    final_sparse_acts = torch.cat(all_sparse_acts, dim=0).coalesce()
    final_valid_token_idx = torch.cat(all_valid_token_idx, dim=0)

    return {
        "sparse_acts": final_sparse_acts,
        "valid_token_idx": final_valid_token_idx,
        "seq_lens": torch.tensor(all_seq_lens, dtype=torch.long),
        "shape": final_sparse_acts.shape
    }