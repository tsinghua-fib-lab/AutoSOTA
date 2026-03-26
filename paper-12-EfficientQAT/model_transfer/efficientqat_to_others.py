import torch
from datautils_block import test_ppl
from transformers import AutoTokenizer
from gptqmodel import GPTQModel, QuantizeConfig, get_backend
from pathlib import Path
import time

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, type=str, help="direction for saving quantization model")
    parser.add_argument("--wbits", type=int, default=4, help="quantization bits")
    parser.add_argument("--group_size", type=int, default=128, help="quantization group size")
    parser.add_argument("--target_format", default='gptq', type=str, help="target checkpoint format")
    parser.add_argument("--eval_ppl", action="store_true")
    parser.add_argument("--test_speed", action="store_true")
    parser.add_argument("--save_dir", default=None, type=str, help="direction for saving quantization model")

    


    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False,legacy=False)
    quant_config = QuantizeConfig(
    bits=args.wbits,  
    group_size=args.group_size,
    sym=False,
    desc_act=False,
    format='gptq_v2',
    )
    if args.target_format == 'gptq':
        # EXLLAMA_V2 is faster in 4-bit, and can inference correctly. However, it has some bug in saving models.
        # Therefore, we choose triton backend as default. Note that the saving model can also be loaded by exllama too.
        model = GPTQModel.from_quantized(args.model, device_map='auto',torch_dtype=torch.float16, quantize_config=quant_config,backend=get_backend('TRITON'))

    elif args.target_format == 'bitblas':
        # take a lone time for the first time runing
        try:
            model = GPTQModel.from_quantized(args.model, device_map='auto',torch_dtype=torch.float16, quantize_config=quant_config,backend=get_backend('BITBLAS'))
            args.eval_ppl = False # BitBLAS have bug, which should re-load model for evaluation otherwise would cause wrong outputs
        except:
            model = GPTQModel.from_quantized(args.model, device_map='auto',torch_dtype=torch.float16, backend=get_backend('BITBLAS'))
    else:
        raise NotImplementedError

    if args.save_dir:
        Path(args.save_dir).mkdir(parents=True, exist_ok=True)
        print("start saving model")
        model.quantize_config.model_file_base_name=None # trick to avoid one saving bug in GPTQModel
        model.save_quantized(args.save_dir,max_shard_size='8GB')  
        tokenizer.save_pretrained(args.save_dir) 
        print(f"save model to {args.save_dir} success")

    model.model.cuda()
    
    if args.eval_ppl:
        datasets = ["wikitext2"]
        ppl_results = test_ppl(model, tokenizer, datasets, 2048)
        for dataset in ppl_results:
            print(f'{dataset} perplexity after transfering: {ppl_results[dataset]:.2f}')
    if args.test_speed:
        prompt = "Write a poem about large language model:"
        input_ids = tokenizer(prompt, return_tensors='pt').input_ids.cuda()
        start_time = time.time()
        output = model.generate(inputs=input_ids, do_sample=True, top_k=10, max_new_tokens=256)
        end_time = time.time()
        speed = len(output[0])/(end_time-start_time)
        print(tokenizer.decode(output[0]))
        print(f"generation speed:{speed:.1f}token/s")
        

if __name__ =='__main__':
    main()