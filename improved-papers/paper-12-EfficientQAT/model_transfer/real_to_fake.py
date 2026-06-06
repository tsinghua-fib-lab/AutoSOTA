from quantize.int_linear_real import load_quantized_model, QuantLinear
import torch
from datautils_block import test_ppl
from pathlib import Path

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None,  help="model path of resumed quantized model")
    parser.add_argument("--save_dir", default=None, type=str, help="direction for saving quantization model")
    parser.add_argument("--wbits", type=int, default=4, help="quantization bits")
    parser.add_argument("--group_size", type=int, default=128, help="quantization group size")
    


    args = parser.parse_args()
    model, tokenizer = load_quantized_model(args.model,args.wbits, args.group_size)
    # model.cuda()
    
    for name, module in model.named_modules():
        if isinstance(module, QuantLinear):
            module.cuda()
            module.use_fake_quantization(del_quant=True,transpose=True)
            module.cpu()


    if args.save_dir:
        Path(args.save_dir).mkdir(parents=True, exist_ok=True)
        print("start saving model")
        model.to(torch.float16)
        model.save_pretrained(args.save_dir)  
        tokenizer.save_pretrained(args.save_dir) 
        print(f"save model to {args.save_dir} success")

if __name__ =='__main__':
    main()