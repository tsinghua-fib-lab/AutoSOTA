import torch
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel


def forward_process(batch, prompt_index, mask_id):
    b, l = batch.shape

    target_len = (l - prompt_index.sum()).item()
    k = torch.randint(1, target_len + 1, (), device=batch.device)

    x = torch.round(torch.linspace(float(k), k + (b - 1) * (target_len / b), steps=b, device=batch.device)).long()
    x = ((x - 1) % target_len) + 1
    assert x.min() >= 1 and x.max() <= target_len

    indices = torch.arange(target_len, device=batch.device).repeat(b, 1)
    is_mask = indices < x.unsqueeze(1)
    for i in range(b):
        is_mask[i] = is_mask[i][torch.randperm(target_len)]

    is_mask = torch.cat((torch.zeros(b, prompt_index.sum(), dtype=torch.bool, device=batch.device), is_mask), dim=1)
    noisy_batch = torch.where(is_mask, mask_id, batch)

    # Return the masked batch and the mask ratio
    return noisy_batch, (x / target_len).unsqueeze(1).repeat(1, l)


def get_logits(model, batch, prompt_index, cfg_scale, mask_id,inputs_embeds=None):
    bbsz = batch.shape[0]
    if cfg_scale > 0.:
        assert len(prompt_index) == batch.shape[1]
        prompt_index = prompt_index.unsqueeze(0).repeat(batch.shape[0], 1)
        un_batch = batch.clone()
        un_batch[prompt_index] = mask_id
        batch = torch.cat([batch, un_batch])
        

    input = batch
    inputs_embeds_curr = model.transformer.wte(input)
    if inputs_embeds is not None:
        inputs_embeds_curr[:bbsz,:inputs_embeds.shape[1]] = inputs_embeds
        # print(bbsz,inputs_embeds_curr.shape[0])
    # assert inputs_embeds_curr.shape[0] in [1,2]
    #
    logits = model(None,input_embeddings=inputs_embeds_curr).logits

    if cfg_scale > 0.:
        logits, un_logits = torch.chunk(logits, 2, dim=0)
        logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
    return logits


@ torch.no_grad()
def get_log_likelihood(model, prompt, answer, mc_num=128, batch_size=16, cfg_scale=0., mask_id=126336,
                       inputs_embeds=None,position_ids=None,attention_mask=None,
                        tokenizer=None,
                        verbose=False,
                         **kwargs
                       ):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (l1).
        answer: A tensor of shape (l2).
        mc_num: Monte Carlo estimation times.
                As detailed in Appendix B.5. Since MMLU, CMMLU, and C-EVAL only require the likelihood of a single token, a
                single Monte Carlo estimate is sufficient for these benchmarks. For all other benchmarks, we find that 128
                Monte Carlo samples are adequate to produce stable results.
        batch_size: Mini batch size.
        cfg_scale: Unsupervised classifier-free guidance scale.
        mask_id: The toke id of [MASK] is 126336.
    '''
    if prompt is None:
        assert inputs_embeds is not None
        bsz, seq_len = inputs_embeds.shape[:2]
        prompt = torch.full((bsz, seq_len), 0, dtype=torch.long).to(model.device)
    
    #seq = torch.concatenate([prompt, answer])[None, :]
    seq = torch.concatenate([prompt, answer],dim=-1)
    seq = seq.repeat((batch_size, 1)).to(model.device)
    prompt_index = torch.arange(seq.shape[1], device=model.device) < prompt.shape[-1]

    loss_ = []
    for _ in range(mc_num // batch_size):
        perturbed_seq, p_mask = forward_process(seq, prompt_index, mask_id)
        mask_index = perturbed_seq == mask_id

        logits = get_logits(model, perturbed_seq, prompt_index, cfg_scale, mask_id,inputs_embeds)

        loss = F.cross_entropy(logits[mask_index], seq[mask_index], reduction='none') / p_mask[mask_index]
        loss = loss.sum() / batch_size

        loss_.append(loss.item())

    return - sum(loss_) / len(loss_)


def main():
    device = 'cuda'

    model = AutoModel.from_pretrained('GSAI-ML/LLaDA-8B-Base', trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Base', trust_remote_code=True)

    # this prompt and answer is from Hellaswag dataset
    prompt = 'Roof shingle removal: A man is sitting on a roof. He'
    answer = ' is using wrap to wrap a pair of skis.'

    prompt = torch.tensor(tokenizer(prompt)['input_ids']).to(device)
    answer = torch.tensor(tokenizer(answer)['input_ids']).to(device)
    print(get_log_likelihood(model, prompt, answer, mc_num=128))


if __name__ == '__main__':
    main()