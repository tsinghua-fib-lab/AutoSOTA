import gc
import logging
from typing import List

import ray
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM

logger = logging.getLogger(__name__)

MAX_NUM_SEQS = 256


def get_prompt_embeds(
        input_prompt: str,
        tokenizer: AutoTokenizer,
        embed_layer: torch.nn.Module,
    ) -> torch.Tensor:
    token_ids = tokenizer(input_prompt)['input_ids']
    token_ids = torch.tensor(token_ids).unsqueeze(0)
    prompt_embeds = embed_layer(token_ids).squeeze(0).detach()

    return prompt_embeds


def init_model(model_args: dict) -> tuple[AutoTokenizer, torch.nn.Module, LLM]:
    embed_layer = AutoModelForCausalLM.from_pretrained(model_args['model']).get_input_embeddings()
    llm = LLM(**model_args, max_num_seqs=MAX_NUM_SEQS)
    tokenizer = llm.get_tokenizer()

    return tokenizer, embed_layer, llm


def cleanup_vllm_and_cuda(llm=None, embed_layer=None, timeout_seconds=30):
    """
    Clean up vLLM engine and CUDA cache to prevent GPU memory leaks.
    
    Follows the recommended cleanup order from Google:
    1. Delete LLM model and embedding layer
    2. Clear CUDA cache
    3. Trigger garbage collection
    4. Shutdown Ray distributed environment
    
    Args:
        llm: vLLM LLM instance to destroy (optional)
        embed_layer: PyTorch embedding layer to cleanup (optional)  
        timeout_seconds: Not used, kept for API compatibility
    """
    logger.info("Starting GPU cleanup...")
    
    # Step 1: Delete the LLM model and embedding layer
    if llm is not None:
        try:
            del llm
            logger.info("✓ LLM model deleted")
        except Exception as e:
            logger.warning(f"✗ Error deleting LLM model: {e}")
    else:
        logger.info("⊘ No LLM model to delete")
    
    if embed_layer is not None:
        try:
            del embed_layer
            logger.info("✓ Embedding layer deleted")
        except Exception as e:
            logger.warning(f"✗ Error deleting embedding layer: {e}")
    else:
        logger.info("⊘ No embedding layer to delete")
    
    # Step 2: Clear CUDA cache
    try:
        torch.cuda.empty_cache()
        logger.info("✓ CUDA cache cleared")
    except Exception as e:
        logger.warning(f"✗ Error clearing CUDA cache: {e}")
    
    # Step 3: Trigger garbage collection
    try:
        gc.collect()
        logger.info("✓ Garbage collection completed")
    except Exception as e:
        logger.warning(f"✗ Error in garbage collection: {e}")
    
    # Step 4: Shutdown Ray distributed environment
    try:
        ray.shutdown()
        logger.info("✓ Ray shutdown completed")
    except Exception as e:
        logger.warning(f"✗ Error shutting down Ray: {e}")
    
    logger.info("GPU cleanup completed successfully")



def get_vllm_text_output(vllm_output: str | List[str]) -> List[str]:
    if isinstance(vllm_output, str):
        vllm_output = [vllm_output]

    return [output.outputs[0].text for output in vllm_output]


def decode_embedding(
    embeddings: torch.Tensor,
    embed_layer: torch.nn.Module,
    tokenizer: AutoTokenizer,
    metric: str = 'cosine'
) -> str:
    """
    Decode embeddings back to text by finding nearest token embeddings.
    
    Args:
        embeddings: Input embeddings of shape [T, p] where T is sequence length, p is embedding dimension
        embed_layer: The embedding layer from the model
        tokenizer: Tokenizer to convert token IDs to text
        metric: Distance metric to use ('cosine' or 'l2')
    
    Returns:
        decoded_text: The decoded text string
    """
    T, p = embeddings.shape
    device = embeddings.device
    
    # Get all token embeddings from the vocabulary [vocab_size, p]
    vocab_embeddings = embed_layer.weight.data  # Shape: [vocab_size, p]
    vocab_embeddings = vocab_embeddings.to(device=device, dtype=embeddings.dtype)
    vocab_size = vocab_embeddings.shape[0]
    
    # Normalize embeddings if using cosine similarity
    if metric == 'cosine':
        embeddings_norm = F.normalize(embeddings, p=2, dim=1)  # [T, p]
        vocab_norm = F.normalize(vocab_embeddings, p=2, dim=1)  # [vocab_size, p]
        # Compute cosine similarity: [T, vocab_size]
        similarities = torch.mm(embeddings_norm, vocab_norm.t())
        # Get token IDs with highest similarity
        nearest_token_ids = torch.argmax(similarities, dim=1)  # [T]
    else:  # L2 distance
        # Compute L2 distance for each position
        nearest_token_ids = []
        for i in range(T):
            emb = embeddings[i:i+1]  # [1, p]
            # Compute L2 distance to all vocab embeddings
            distances = torch.norm(vocab_embeddings - emb, dim=1)  # [vocab_size]
            nearest_idx = torch.argmin(distances)
            nearest_token_ids.append(nearest_idx)
        nearest_token_ids = torch.stack(nearest_token_ids)  # [T]
    
    # Decode token IDs to text
    decoded_text = tokenizer.decode(nearest_token_ids.cpu().tolist())
    
    return decoded_text
