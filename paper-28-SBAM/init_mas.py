#%%
from transformers import AutoTokenizer
from mas_llama_impl.modeling_mas_llama import MasLlamaForCausalLM

# this scripts currently only support llama 3.2 that works with eager mode
# please double check the MAS template (mode) you want to use
for llama_model in [
    'meta-llama/Llama-3.2-1B',
    # 'meta-llama/Llama-3.2-1B-Instruct',
    # 'meta-llama/Llama-3.2-3B',
    # 'meta-llama/Llama-3.2-3B-Instruct',
    # 'meta-llama/Llama-3.1-8B',
    # 'meta-llama/Meta-Llama-3-8B'
    ]:
    print(f'\nInit MAS for {llama_model}')
    tokenizer = AutoTokenizer.from_pretrained(llama_model)
    model = MasLlamaForCausalLM.from_pretrained(llama_model, attn_implementation="eager")

    new_mas_llama_model = llama_model.split('/')[-1] + '_MAS'
    # save to mas_llama folder
    tokenizer.save_pretrained(new_mas_llama_model)
    model.save_pretrained(new_mas_llama_model)

    print(f'Successfully saved into {new_mas_llama_model}')

    del model
    del tokenizer

#%%

# from transformers import AutoModelForCausalLM
# model = AutoModelForCausalLM.from_pretrained('Llama-3.2-1B_MAS', attn_implementation="eager", MAS_template='v1')
# model = AutoModelForCausalLM.from_pretrained('Llama-3.2-1B_MAS', attn_implementation="eager", MAS_template='alpaca')
# %%
