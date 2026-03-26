import torch
import spacy
nlp = spacy.load("en_core_web_sm")

def get_seq_acts_and_out(model, submodules, prompt, num_tokens=25):
    if not isinstance(submodules, list):
        submodules = [submodules]
    with torch.no_grad():                
        with model.trace(prompt, invoker_args={"truncation": True, "max_length": 2048, "add_special_tokens": False}):
            # print("LENGTHENED MAX TOKENS FOR SEQUENCE SPARSITY")
            if hasattr(model, "model"):
                inputs = model.model.embed_tokens.input.save() # model.model.embed_tokens
            elif hasattr(model, "gpt_neox"):
                inputs = model.gpt_neox.embed_in.input.save()
            else:
                print("model doesnt have .model or .gpt_neox attr, check how to access layers")
                exit(1)
            
            hidden_states = []
            for submodule in submodules:
                hidden_states.append(submodule.output.save())
    pos = []
    
    if num_tokens == -1:
        num_tokens = hidden_states[0][0].shape[1]
    # indices = torch.randperm(hidden_states[0].shape[1]-2)[:25]+2
    indices = torch.arange(-num_tokens,0)

    for ind in indices:
        doc = nlp(model.tokenizer.decode(inputs[0, ind]))
        pos.append(doc[-1].pos_)

    if len(submodules) == 1:
        return hidden_states[0][0][0][indices], inputs[0][indices], pos
    else:
        return torch.stack([h[0][0][indices] for h in hidden_states],dim=0), inputs[0][indices], pos