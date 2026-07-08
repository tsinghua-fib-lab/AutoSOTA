import torch
import numpy as np
import time
import torch.nn.functional as F
from model.modeling_llada import LLaDAModelLM
from transformers import AutoTokenizer


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


@ torch.no_grad()
def generate(model: LLaDAModelLM, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
             cfg_scale=0., remasking='low_confidence', mask_id=126336):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
    '''
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    prompt_index = (x != mask_id)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    for num_block in range(num_blocks):
        block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length:] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        for i in range(steps):
            mask_index = (x == mask_id)
            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                logits = model(x_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x).logits

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1) 

            if remasking == 'low_confidence':
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) 
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)

            x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]

    return x


def get_top1_info(logits: torch.Tensor):
    probs = F.softmax(logits, dim=-1)
    top1_probs, top1_indices = torch.max(probs, dim=-1)
    return top1_indices, top1_probs

def verify_and_commit(
    region_logits: torch.Tensor,
    region_x: torch.Tensor,
    mask_id: int,
    commit_thres: float,
    left_boundary_known: bool
):
    top1_indices, top1_probs = get_top1_info(region_logits)

    is_confident =  (top1_probs > commit_thres)

    if left_boundary_known:
        contiguity_mask = torch.cumprod(is_confident.int(), dim=0).bool()
    else:
        contiguity_mask = torch.zeros_like(is_confident, dtype=torch.bool)

    is_mask = (region_x == mask_id)
    final_commit_mask = contiguity_mask & is_mask
    
    return top1_indices, final_commit_mask

@torch.no_grad()
def generate_with_dc_leap(
    model: "LLaDAModelLM",
    prompt: torch.Tensor,
    steps: int,
    commit_thres: float, 
    draft_thres: float,
    gen_length: int,
    block_length: int,
    max_window_size: int,
    cfg_scale: float,
    temperature: float,
    remasking='low_confidence', 
    mask_id: int = 126336,
     
) -> torch.Tensor:
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        commit_thres: Confidence threshold for Dynamic Contiguous Verification (DCV). Only the longest contiguous prefix within the decoding window where tokens exceed this threshold will be committed. 
        draft_thres: Confidence threshold  for the Draft Mechanism. High-confidence tokens predicted outside the decoding window are cached to provide look-ahead context for bidirectional attention. 
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        max_window_size: Maximum size (L) of the dynamic decoding window. 
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
    '''
    device = model.device
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long, device=device)
    x[:, :prompt.shape[1]] = prompt
    prompt_len = prompt.shape[1]
    draft_bank = torch.full((gen_length,), mask_id, dtype=torch.long, device=device)

    verified_end = 0 
    while verified_end < gen_length:
        
        l2r_len = max_window_size
        future_len = max_window_size
        win_s = verified_end
        win_e = min(verified_end + l2r_len + future_len, gen_length)

        abs_win_s = prompt_len + win_s
        abs_win_e = prompt_len + win_e

        if abs_win_s >= abs_win_e: break

        x_for_prediction = x.clone()

        drafts = draft_bank[win_s:win_e]
        target_slice = x_for_prediction[0, abs_win_s:abs_win_e]
        mask_locs = (target_slice == mask_id)
        valid_drafts = (drafts != mask_id)
        fill_locs = mask_locs & valid_drafts
        if fill_locs.any():
            x_for_prediction[0, abs_win_s:abs_win_e][fill_locs] = drafts[fill_locs]

        leader_end = min(abs_win_s + max_window_size, abs_win_e)
        x_for_prediction[0, abs_win_s:leader_end] = mask_id
        
        if cfg_scale > 0.:
            pred_model_out = model(x_for_prediction, output_hidden_states=False)
            prediction_logits = pred_model_out.logits
            pred_conditional, pred_unconditional = prediction_logits.chunk(2, dim=0)
            prediction_logits = pred_unconditional + (cfg_scale + 1) * (pred_conditional - pred_unconditional)
        else:
            pred_model_out = model(x_for_prediction, output_hidden_states=False)
            prediction_logits = pred_model_out.logits

        l2r_abs_end = min(prompt_len + verified_end + l2r_len, abs_win_e)
        
        if l2r_abs_end > abs_win_s:
            region_logits = prediction_logits[0, abs_win_s:l2r_abs_end]
            region_x = x[0, abs_win_s:l2r_abs_end]

            left_boundary_known = True
            if verified_end > 0:
                left_boundary_known = (x[0, abs_win_s - 1].item() != mask_id)
            
            top1_tokens, commit_mask = verify_and_commit(
                region_logits, region_x, mask_id, 
                commit_thres, left_boundary_known
            )

            if not commit_mask.any():
                is_mask = (region_x == mask_id)
                if is_mask.any():
                    first_idx = is_mask.nonzero(as_tuple=True)[0][0]
                    commit_mask[first_idx] = True

            if commit_mask.any():
                update_idx = commit_mask.nonzero(as_tuple=True)[0]
                x[0, abs_win_s + update_idx] = top1_tokens[update_idx]
            
                next_mask = (x[0, abs_win_s:l2r_abs_end] == mask_id).nonzero(as_tuple=True)
                if next_mask[0].numel() > 0:
                    verified_end += next_mask[0][0].item()
                else:
                    verified_end += (l2r_abs_end - abs_win_s)

        draft_logits = prediction_logits[0, abs_win_s:abs_win_e]
        r_idx, r_probs = get_top1_info(draft_logits)

        draft_candidates_mask = r_probs > draft_thres
        if draft_candidates_mask.any():
            indices = draft_candidates_mask.nonzero(as_tuple=True)[0]
            bank_indices = win_s + indices

            valid = bank_indices < gen_length
            if valid.all():
                draft_bank[bank_indices] = r_idx[indices]
            elif valid.any():
                draft_bank[bank_indices[valid]] = r_idx[indices][valid]
    x = x[:, :prompt.shape[1] + gen_length]
    return x



def main():
    device = 'cuda'
    model = LLaDAModelLM.from_pretrained('/path/to/your/LLaDA-8B-Instruct', trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained('/path/to/your/LLaDA-8B-Instruct', trust_remote_code=True)

    prompt = "Albert has 2 apples. He buys 3 more boxes of apples. Each box contains 4 apples. He then gives half of his total apples to his friend. How many apples does Albert have left?"
    m = [{"role": "user", "content": prompt}, ]
    
    prompt_str = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)

    inputs = tokenizer(prompt_str, return_tensors="pt")
    input_ids = inputs.input_ids.to(device)

    start = time.time()
    #out = generate(model, input_ids, steps=256, gen_length=256, block_length=32, temperature=0., cfg_scale=0., remasking='low_confidence')
    out = generate_with_dc_leap(model, input_ids, steps=256, commit_thres=0.7, draft_thres=0.98, gen_length=256, block_length=32, max_window_size=128, temperature=0., cfg_scale=0., remasking='low_confidence')
    print(f'{time.time() - start}')
    print(tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)[0])

if __name__ == '__main__':
    main()



