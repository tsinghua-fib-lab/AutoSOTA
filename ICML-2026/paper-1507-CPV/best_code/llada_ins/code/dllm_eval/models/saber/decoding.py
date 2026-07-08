import torch
import torch.nn.functional as F
import numpy as np
import time
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

def margin_function(probabilities):
    if probabilities.dim() != 3:
        raise ValueError("Input tensor 'probabilities' must be a 3D tensor with shape [batch_size, sequence_len, vocab_size]")
    sorted_probs, _ = torch.sort(probabilities, dim=-1, descending=True)
    top1_probs = sorted_probs[:, :, 0]
    top2_probs = sorted_probs[:, :, 1]
    confidence = top1_probs - top2_probs
    return confidence

def entropy_function(probabilities):
    if probabilities.dim() != 3:
        raise ValueError("Input tensor 'probabilities' must be a 3D tensor with shape [batch_size, sequence_len, vocab_size]")
    epsilon = 1e-12
    probs_safe = probabilities.clone() + epsilon
    entropy = torch.sum(probabilities.clone() * torch.log(probs_safe), dim=-1)
    return entropy


@torch.no_grad()
def generate_with_saber(
    model,
    prompt,
    n=2,
    mu=8,
    gen_length=256,
    block_length=256,
    temperature=0.0,
    mask_id=126336,
    track_flip_flop: bool = False,
):

    step = 0
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()
    prompt_index = (x != mask_id)
    mask_index = (x == mask_id) 
    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    global_transfer_index = torch.zeros_like(x, dtype=torch.bool, device=x.device)
    initial_confidence = torch.zeros_like(x, dtype=torch.float32, device=x.device)
    final_confidence = torch.full_like(x, fill_value=-np.inf,dtype=torch.float32)
    last_confidence = torch.zeros_like(x, dtype=torch.float32, device=x.device)

    # Flip-flop tracking: a position is unmasked with token T -> remasked -> unmasked again with SAME token T.
    if track_flip_flop:
        position_history = [{} for _ in range(int(x.shape[0]))]  # {pos: {'last_token': int, 'was_remasked': bool}}
        flip_flop_count = [0 for _ in range(int(x.shape[0]))]
        total_unmask_count = [0 for _ in range(int(x.shape[0]))]
        total_remask_count = [0 for _ in range(int(x.shape[0]))]

    for num_block in range(num_blocks):
        block_end = prompt.shape[1] + (num_block + 1) * block_length
        block_start = prompt.shape[1] + num_block * block_length
        for i in range(1024):

            step += 1
            logits = model(x).logits
            
            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1) 

            p = F.softmax(logits, dim=-1)
            x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l

            x0_p[:, block_end:] = -np.inf

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)
            confidence_delta = confidence - last_confidence
            mask_finite = torch.isfinite(confidence_delta)
            confidence_delta = torch.where(mask_finite, confidence_delta, torch.full_like(confidence_delta, float('inf')))
            last_confidence = confidence
            confidence = torch.where(global_transfer_index, -np.inf, confidence)
            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            fix_list = []
            for j in range(confidence.shape[0]):
                block_final_confidence = final_confidence[j, :block_end]
                #block_final_confidence = final_confidence[j, block_start:block_end] 
                generated_probs = block_final_confidence[block_final_confidence > -np.inf]
                if generated_probs.numel() > 0:
                    threshold = generated_probs.mean()
                else:
                    threshold = 1
                select_index = torch.where(confidence[j] > threshold)[0]
                if select_index.numel() < n:
                    _, select_index = torch.topk(confidence[j], k=n)

                fix_list.append(max(n, select_index.numel()))

                transfer_index[j, select_index] = True
                global_transfer_index[j, select_index] = True

            # ========== Track flip-flops on UNMASK ==========
            if track_flip_flop:
                for j in range(int(x.shape[0])):
                    newly_unmasked_positions = torch.where(transfer_index[j])[0].tolist()
                    total_unmask_count[j] += len(newly_unmasked_positions)
                    for pos in newly_unmasked_positions:
                        current_token = x0[j, pos].item()
                        hist = position_history[j].get(pos)
                        if hist is not None and hist.get("was_remasked") and hist.get("last_token") == current_token:
                            flip_flop_count[j] += 1
                        position_history[j][pos] = {"last_token": current_token, "was_remasked": False}
            x[transfer_index] = x0[transfer_index]
            
            new_generated_mask = transfer_index & (initial_confidence == 0.0)
            if new_generated_mask.any():
                initial_confidence = torch.where(new_generated_mask, x0_p, initial_confidence)
            
            final_confidence = torch.where(transfer_index, x0_p, final_confidence)

            all_masked_transferred = torch.all(global_transfer_index[mask_index]).item()
            
            if all_masked_transferred:
                if track_flip_flop:
                    flip_flop_stats = {
                        "flip_flop_count": flip_flop_count,
                        "total_unmask_count": total_unmask_count,
                        "total_remask_count": total_remask_count,
                        "steps": step,
                    }
                    return x, step, flip_flop_stats
                return x, step   

            if torch.all(x[:, :block_end] != mask_id):
                break  
            delta_for_removal = confidence_delta.clone()

            delta_for_removal = torch.where(mask_index, delta_for_removal, torch.full_like(delta_for_removal, float('inf')))
            if block_end < delta_for_removal.shape[1]:
                delta_for_removal[:, block_end:] = float('inf')
            positions_mask_now = (x == mask_id)
            delta_for_removal[positions_mask_now] = float('inf')
            delta_for_removal[prompt_index] = float('inf')

            for j in range(delta_for_removal.shape[0]):
                num_to_remask = max(int(n/2),(fix_list[j] + mu-1) // mu)

                if num_to_remask > fix_list[j]-1:
                    num_to_remask = fix_list[j]-1

                _, remove_index = torch.topk(delta_for_removal[j], k=num_to_remask, largest=False)

                # ========== Track flip-flops on REMASK ==========
                if track_flip_flop:
                    remasked_positions = remove_index.tolist()
                    total_remask_count[j] += len(remasked_positions)
                    for pos in remasked_positions:
                        hist = position_history[j].get(pos)
                        if hist is not None:
                            hist["was_remasked"] = True
                x[j, remove_index] = mask_id  
                initial_confidence[j, remove_index] = 0.0
                global_transfer_index[j, remove_index] = False
    if track_flip_flop:
        flip_flop_stats = {
            "flip_flop_count": flip_flop_count,
            "total_unmask_count": total_unmask_count,
            "total_remask_count": total_remask_count,
            "steps": step,
        }
        return x, step, flip_flop_stats
    return x, step

def get_num_transfer_tokens(mask_index, steps):
    '''
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    '''
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens

def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise
        
def get_num_transfer_tokens_ours(mask_index, steps):
    '''
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    '''
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps
    
    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base*2

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens

def get_transfer_index(logits, temperature, remasking, mask_index, x, num_transfer_tokens, threshold=None):
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1) # b, l

    if remasking == 'low_confidence':
        p = F.softmax(logits.to(torch.float64), dim=-1)
        x0_p = torch.squeeze(
            torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
    elif remasking == 'random':
        x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
    else:
        raise NotImplementedError(remasking)
    
    x0 = torch.where(mask_index, x0, x)
    confidence = torch.where(mask_index, x0_p, -np.inf)

    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
    if threshold is not None:
        num_transfer_tokens = mask_index.sum(dim=1, keepdim=True)
    for j in range(confidence.shape[0]):
        _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j])
        transfer_index[j, select_index] = True
        if threshold is not None:
            for k in range(1, num_transfer_tokens[j]):
                if confidence[j, select_index[k]] < threshold:
                    transfer_index[j, select_index[k]] = False
    return x0, transfer_index

def get_transfer_index_dynamic(logits, temperature, remasking, mask_index, x, num_transfer_tokens, factor=1):
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1) # b, l
    if remasking == 'low_confidence':
        p = F.softmax(logits.to(torch.float64), dim=-1)
        x0_p = torch.squeeze(
            torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
    elif remasking == 'random':
        x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
    else:
        raise NotImplementedError(remasking)
    
    x0 = torch.where(mask_index, x0, x)
    confidence = torch.where(mask_index, x0_p, -np.inf)

    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
    num_transfer_tokens = mask_index.sum(dim=1, keepdim=True)
    
    for j in range(confidence.shape[0]):
        ns=list(range(1,num_transfer_tokens[j]+1))
        es=[factor/(n+1) for n in ns]
        threshs=[1-e for e in es]

        # at least one token is transferred
        threshs[0]=-1
        sorted_confidence=torch.sort(confidence[j][mask_index[j]],dim=-1,descending=True)[0]
        assert len(sorted_confidence)==len(threshs)
        for top_i in range(len(threshs)):
            if sorted_confidence[top_i]<threshs[top_i]:
                break

        if top_i == 0 or top_i == len(threshs)-1:
            top_i+=1

        _, select_index = torch.topk(confidence[j], k=top_i)
        transfer_index[j, select_index] = True

    return x0, transfer_index
