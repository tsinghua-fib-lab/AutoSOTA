import os
import torch
from torch import nn

from data_utils import get_testdata

@torch.no_grad()
def evaluate(model, args):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ppl_results = {}
    for testset in ["wikitext2", 'c4-new']:
        testdata = get_testdata(testset, args)
        ppl = eval(model, testdata, dev)
        print(f'{testset} : {ppl :.3f}')
        ppl_results[testset] = round(ppl, 3)

    return ppl_results

def eval(model, testenc, dev):
    try:
        testenc = testenc.input_ids
    except AttributeError:
        pass
    seqlen = model.seqlen
    nsamples = testenc.numel() // seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False
    
    if not hasattr(model, 'model'):  # BLOOM
        model_type = 'bloom'
        layers = model.transformer.h
        model.transformer.word_embeddings = model.transformer.word_embeddings.to(dev)
        model.transformer.word_embeddings_layernorm = model.transformer.word_embeddings_layernorm.to(dev)

    else:
        if hasattr(model.model, 'decoder'): #OPT
            model_type = "opt"
            layers = model.model.decoder.layers #OPTQ
            model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
            model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
            if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
                model.model.decoder.project_out = model.model.decoder.project_out.to(dev) 
            if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
                model.model.decoder.project_in = model.model.decoder.project_in.to(dev) 
        else:
            model_type = "llama"
            layers = model.model.layers # LLaMA
            model.model.embed_tokens = model.model.embed_tokens.to(dev)
    
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            if kwargs.get('position_ids', None) is not None:
                cache['position_ids'] = kwargs['position_ids']  # LLaMA
            if kwargs.get('alibi', None) is not None:
                cache['alibi'] = kwargs['alibi']  # Bloom
            raise ValueError
    layers[0] = Catcher(layers[0])
    for i in range(nsamples):
        batch = testenc[:, (i * seqlen):((i + 1) * seqlen)].to(dev)
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    
    if model_type == "opt":
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
        if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
            model.model.decoder.project_out = model.model.decoder.project_out.cpu()
        if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
            model.model.decoder.project_in = model.model.decoder.project_in.cpu()
    elif model_type == "llama":
        model.model.embed_tokens = model.model.embed_tokens.cpu()
    else:
        model.transformer.word_embeddings = model.transformer.word_embeddings.to("cpu")
        model.transformer.word_embeddings_layernorm = model.transformer.word_embeddings_layernorm.to("cpu")
    
    torch.cuda.empty_cache()

    kwargs = {'attention_mask': cache['attention_mask']}
    if cache.get('position_ids', None) is not None:
        kwargs['position_ids'] = cache['position_ids']
    if cache.get('alibi', None) is not None:
        kwargs['alibi'] = cache['alibi']

    for i in range(len(layers)):
        layer = layers[i].to(dev)
        for j in range(nsamples):
            inps[j] = layer(inps[j].unsqueeze(0), **kwargs)[0]
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()

    if model_type == "opt":
        if model.model.decoder.final_layer_norm is not None:
            model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.to(dev)
        if model.model.decoder.project_out is not None:
            model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
    elif model_type == "llama":
        if model.model.norm is not None:
            model.model.norm = model.model.norm.to(dev)
    else:
        model.transformer.ln_f = model.transformer.ln_f.to(dev)
    model.lm_head = model.lm_head.to(dev)

    testenc = testenc.to(dev)
    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        if model_type == "opt":
            if model.model.decoder.final_layer_norm is not None:
                hidden_states = model.model.decoder.final_layer_norm(hidden_states)
            if model.model.decoder.project_out is not None:
                hidden_states = model.model.decoder.project_out(hidden_states)
        elif model_type == "llama":
            if model.model.norm is not None:
                hidden_states = model.model.norm(hidden_states)
        else:
            hidden_states = model.transformer.ln_f(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[:, (i * seqlen):((i + 1) * seqlen)][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * seqlen))
    # print(f"\tResult: {ppl.item():.3f}")

    model.config.use_cache = use_cache
    return ppl.item()