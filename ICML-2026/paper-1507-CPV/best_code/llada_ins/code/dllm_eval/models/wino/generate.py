import torch
import numpy as np
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel
from modeling_llada import LLaDAModelLM
import time

from decoding import decoding_default, decoding_wino

def main():
    device = 'cuda:0'
    model_path = '/PATH/TO/LLaDA-8B-Instruct'
    gen_length = 256
    block_length = 128
    steps = 256
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = LLaDAModelLM.from_pretrained(model_path,torch_dtype=torch.bfloat16).to(device).eval()
    prompt = 'Hector purchased a container of gumballs. He gave 4 to Todd, then he gave twice as many as he had given Todd to Alisha, and then he gave 5 less than four times as many to Bobby as he had given to Alisha. If Hector had 6 gumballs remaining, what is the total number of gumballs that Hector purchased?'
    cot_prompt = 'Please think step by step.'
    prompt = prompt + '\n' + cot_prompt
    m = [{"role": "user", "content": prompt}, ]
    prompt = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)

    input_ids = tokenizer(prompt)['input_ids']
    input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)

    out, step = decoding_default(model, input_ids, steps=steps, gen_length=gen_length, block_length=block_length, temperature=0., cfg_scale=0., remasking='low_confidence')
    print(f'Default: {tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)[0]}')
    print(f'\nSteps: {step}\n\n')

    out, step = decoding_wino(model, input_ids, gen_length=gen_length, block_length=block_length, temperature=0., threshold=0.5, threshold_back=0.9)
    print(f'WINO: {tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)[0]}')
    print(f'\nSteps: {step}\n\n')

if __name__ == '__main__':
    main()