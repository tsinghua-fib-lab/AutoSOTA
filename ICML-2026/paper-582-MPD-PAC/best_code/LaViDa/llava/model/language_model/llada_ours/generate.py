import torch
import numpy as np
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel


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

def get_num_transfer_tokens_sch(mask_index, steps,schedule=None,schedule_kwargs=None):
    '''
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    '''
    if schedule is None:
        return get_num_transfer_tokens(mask_index,steps)
    if schedule_kwargs is None:
        schedule_kwargs = {}
   
    mask_num = mask_index.sum(dim=1, keepdim=True)
    steps = int(min(steps,mask_num[0]))
    t = torch.linspace(0, 1, steps+1)
    # at least one sample per step
    if schedule =='logit_normal':
      sigmas = sigmoid_normal_cdf(t)
    elif schedule =='shift':
      sigmas = logit_normal_schedule(schedule_kwargs.get('shift',3),t)
    elif schedule == 'cosine':
        sigmas = cosine_schedule(t)
    else:
      sigmas = t
    sigmas = sigmas.to(mask_num.device)
    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64)
    
    for i in range(mask_num.size(0)):
      # print(sigmas.shape)
      sigmas_sample = (sigmas*mask_num[i]).to(torch.int64)
      # print(sigmas_sample)
      sigmas_sample = sigmas_sample[1:]-sigmas_sample[:-1]
      # print(sigmas_sample)
      # fix detal
      sigmas_sample = torch.clamp(sigmas_sample,1,None) # should only increase
      delta = sigmas_sample.sum() - mask_num[i]
    #   breakpoint()
      assert delta>=0
      j = 0
      
      while delta > 0:
        j = j % len(sigmas_sample) 
        if sigmas_sample[j] == 1:
          j += 1
          continue
        
        delta -= 1
        sigmas_sample[j] -= 1
        j += 1
    #   breakpoint()
      assert sigmas_sample.sum()==mask_num[i]
      num_transfer_tokens[i] = sigmas_sample#.to(torch.int64)
    return num_transfer_tokens.flip(-1)

def linear(y):
    return y

def cosine_schedule(x):
    """
    Cosine schedule mapping [0, 1] -> [1, 0]
    """
    x = np.clip(x, 0, 1)
    return 1-0.5 * (1 + np.cos(np.pi * x))

def sigmoid_normal_cdf(y):
    # y must be in (0, 1)
    logit_y = torch.log(y / (1 - y))
    return 0.5 * (1 + torch.erf(logit_y / torch.sqrt(torch.tensor(2.0))))
def logit_normal_schedule(shift,sigmas):
    # shift = 1 / shift
    sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
    return sigmas

@torch.no_grad()
def init_prior_lastlayer_subspace_from_32layers(model, k=3):
    """
    Build a low-rank prior subspace using PCA over
    layer-wise hidden states of the uncontextualized prior token.

    Cache:
    self.prior_mu : (D,)
    self.prior_Vt : (D, k)
    """
    if hasattr(model, "prior_mu") and hasattr(model, "prior_Vt"):
        return

    E = model.transformer.wte.weight.detach()
    E_mean = E.mean(dim=0, keepdim=True)
    dummy = torch.tensor([[0]], device=model.device)
    attention_mask = torch.ones((1, 1), device=model.device)
    labels = torch.full_like(dummy, -100)
    inputs_embeds = E_mean.unsqueeze(0).to(model.device)   # (1, 1, D)

    with torch.no_grad():
        out = model(
            input_ids=dummy,                         
            input_embeddings=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

    # (3) collect per-layer hidden states
    # 일부 레이어가 서로 다른 GPU(cuda:0, cuda:1)에 있을 수 있으므로
    # PCA 계산은 안전하게 CPU에서 수행한다.
    pca_device = torch.device("cpu")
    H = []
    for h in out.hidden_states[1:]:               # skip embedding layer
        H.append(h[0, 0].detach().to(pca_device))  # (D) on CPU
    H = torch.stack(H, dim=0)                      # (L, D) on CPU

    # (4) PCA
    mu = H.mean(dim=0).float()
    Hc = H - mu.unsqueeze(0)

    U, S, Vh = torch.linalg.svd(Hc, full_matrices=False)
    Vt = Vh[:k].T                                 # (D, k)

    h_prior_last = out.hidden_states[-1][0,0].detach().to(pca_device).float()  # (D,)
    z_prior_last= (h_prior_last-mu)@Vt
    u=torch.nn.functional.normalize(z_prior_last,dim=-1)

    model.prior_mu = mu                             # (D,) on CPU
    model.prior_Vt = Vt                             # (D, k) on CPU
    model.prior_u = u                               # (k,) on CPU
    

import os
DEBUG_PRINT_OUTPUT = os.environ.get('DEBUG_PRINT_OUTPUT',False)
@ torch.no_grad()
def generate(model, prompt=None, steps=None, max_new_tokens=128, block_length=128, temperature=0.,
                         cfg_scale=0., remasking='low_confidence', mask_id=126336,inputs_embeds=None, position_ids=None,attention_mask=None,
                            tokenizer=None,
                                verbose=False,
                                step_per_block=None,
                                prefix_lm=False,
                                schedule=None,
                                schedule_kwargs=None,
                                draft_tokens=None,
                                step_ratio=None,
                                rope=None,
                                prior=None,
                                mode=None,
                                hs=False,
                                attn=False,
                                slope=None,
                                img_id=None,
                                center=None,
                                k=None,
                         **kwargs):
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
    prior = 0.0 if prior is None else float(prior)
    # We don't forward HF-style `output_attentions` into the model, but if
    # the caller passed it, remove it from kwargs so it doesn't leak.
    # _ = kwargs.pop("output_attentions", None)
    # collect_attn = bool(attn)

    # When attn=True, collect per-step attention weights from the model.

    # breakpoint()
    # remasking = 
    # step_ratio = 0.5
    # block_length = 1024
    # steps = 1024
    steps = max_new_tokens # min(steps,max_new_tokens)
    # if step_ratio:
    #     steps = int(max_new_tokens*step_ratio)
    gen_length = max_new_tokens
    assert position_ids is None
    if prompt is None:
        assert inputs_embeds is not None
        bsz, seq_len = inputs_embeds.shape[:2]
        prompt = torch.full((bsz, seq_len), 0, dtype=torch.long).to(model.device)
    past_key_values = None
    if prefix_lm:
        past_key_values = model(None,input_embeddings=inputs_embeds,use_cache=True).attn_key_values
        # breakpoint()
        x = torch.full((bsz, gen_length), mask_id, dtype=torch.long).to(model.device)
        prompt = torch.full((bsz, 0), 0, dtype=torch.long).to(model.device)
        # x[:, :prompt.shape[1]] = prompt.clone()
    else:
        x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
        x[:, :prompt.shape[1]] = prompt.clone()

    prompt_index = (x != mask_id)
    # assert prompt.shape[0] == 1
    if draft_tokens is not None:
        assert draft_tokens.shape[1] <= gen_length
        x[:, prompt.shape[1]:prompt.shape[1]+draft_tokens.shape[1]] = draft_tokens.clone()

    # if block_length < gen_length:
    #    block_length = gen_length
    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert ( steps % num_blocks == 0) or step_per_block is not None
    steps = steps // num_blocks
    if step_per_block:
        steps = min(step_per_block,block_length)
        assert step_ratio is None, 'Please do not pass both step_ratio and step_per_block'

    if step_ratio:
        steps = int(steps*step_ratio)


    if verbose:
        history = []
    st = 0
    transfer_ids=[]

    if hs:
        hs_list = []
        transfer_ids=[]

    for num_block in range(num_blocks):
        block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length:] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens_sch(block_mask_index, steps,schedule=schedule,schedule_kwargs=schedule_kwargs)
        if DEBUG_PRINT_OUTPUT:
            print(f"Block: {num_block + 1}/{num_blocks}, Steps per Block: {steps}, Block Length: {block_length}")
            print(f"Tokens generated per step {num_transfer_tokens[0]}")
        for i in range(steps):
            # print(i)
            mask_index = (x == mask_id)
            block_mask_index = mask_index[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length:]
            # print(mask_index.sum())
            if block_mask_index.sum() == 0:
                continue
            # NFE += 2
            if cfg_scale > 0.:
                assert NotImplementedError('cfg_scale > 0. is not supported.')
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                #
                logits = model(x_,input_embeds_inference=[inputs_embeds,None]).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                inputs_embeds_curr = model.transformer.wte(x)
                if prefix_lm:
                    outputs = model(
                        None,
                        input_embeddings=inputs_embeds_curr,
                        past_key_values=past_key_values,
                        rope=rope,
                        mode=mode,
                        slope=slope,
                        center=center,
                        output_hidden_states=True,
                    )
                else:
                    if inputs_embeds is not None:
                        inputs_embeds_curr[:,:inputs_embeds.shape[1]] = inputs_embeds
                    outputs = model(
                        None,
                        input_embeddings=inputs_embeds_curr,
                        rope=rope,
                        mode=mode,
                        slope=slope,
                        attn=attn,
                        center=center,
                        output_hidden_states=True,
                        # return_dict=True,
                    )
        
                init_prior_lastlayer_subspace_from_32layers(model,k=k)


                H_full = outputs.hidden_states[-1]  
                mu = model.prior_mu.to(H_full.device)   
                Vt = model.prior_Vt.to(H_full.device)   
                u  = model.prior_u.to(H_full.device)   
                

                masked_positions = mask_index[0].to(H_full.device)
                if masked_positions.any():
                    H_masked = H_full[0,masked_positions]             
                    delta = (H_masked - mu.unsqueeze(0)).float()     
                    z = delta @ Vt                                   

                    proj_scalar = (z * u.unsqueeze(0)).sum(dim=-1, keepdim=True)   
                    proj_vec = proj_scalar * u.unsqueeze(0)                       

                    z_norm = torch.norm(z, dim=-1, keepdim=True) + 1e-6
                    cos_zu = (proj_scalar / z_norm).squeeze(-1)                   
                    cos_zu = torch.clamp(cos_zu, min=0.0, max=1.0)                 

                    alpha = prior * cos_zu                                        

                    z_new = z - alpha.unsqueeze(-1) * proj_vec                      

                    delta_sub_old = z @ Vt.T
                    delta_sub_new = z_new @ Vt.T
                    delta_new = delta_sub_new + (delta - delta_sub_old)       

                    H_masked_new = (mu.unsqueeze(0) + delta_new).to(H_full.dtype)     
                    H_full[0,masked_positions] = H_masked_new
                    if hs:
                        hs_list.append(H_full[:,-gen_length:,:].clone().cpu())

                ff_weight = model.transformer.ff_out.weight.to(H_full.device)
                logits = H_full @ ff_weight.t()
              
            # logits = logits.cpu()
            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1) # b, l
            # torch.cuda.empty_cache()
            # torch.cuda.synchronize()
            if remasking == 'low_confidence':
                p = F.softmax(logits.to(torch.float64), dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            elif remasking == 'entrophy':
                epsilon = 1e-10
                probs = F.softmax(logits.to(torch.float64), dim=-1)
                log_probs = torch.log(probs + epsilon)
                x0_p = torch.sum(probs * log_probs, dim=-1)
            elif remasking == 'margin':
                ## similar to margin algo in Dream
                p = F.softmax(logits.to(torch.float64), dim=-1)
                sorted_probs, _ = torch.sort(p, dim=-1, descending=True)
                top1_probs = sorted_probs[:, :, 0] 
                top2_probs = sorted_probs[:, :, 1] 
                x0_p = top1_probs - top2_probs 
            else:
                raise NotImplementedError(remasking)

            x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                try:
                    _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                    if hs:
                        transfer_ids.append(select_index.cpu())
                except:
                    breakpoint()
                transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]
            if verbose:
                history.append(x.clone().cpu())

    if verbose:
        return x, history
    if hs:
        return x, transfer_ids, inputs_embeds.shape[1], hs_list
    return x 

def main():
    device = 'cuda'

    model = AutoModel.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)

    prompt = "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour. How many kilometers can she run in 8 hours?"

    m = [{"role": "user", "content": prompt}, ]
    prompt = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)

    input_ids = tokenizer(prompt)['input_ids']
    input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)

    out = generate(model, input_ids, steps=128, gen_length=128, block_length=32, temperature=0., cfg_scale=0., remasking='low_confidence')
    print(tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)[0])
    generate(model, input_ids, steps=128, gen_length=128, block_length=32, temperature=0., cfg_scale=0., remasking='low_confidence')
   

if __name__ == '__main__':
    main()