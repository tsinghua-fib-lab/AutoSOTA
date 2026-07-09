import torch
from data_utils import get_eval_dataset
import numpy as np


def get_score(args, tokenizer, x, pred, mask, i):
    x = x[i]
    pred = pred[i]
    mask = mask[i]

    # str_acc = int(torch.equal(x * mask, pred * mask)) # I don't think I know what equal does
    str_acc = int(sum(torch.eq(x * mask, pred * mask)) == sum(mask))
    
    if sum(mask) == 0:
        char_acc = 1
    else:
        # char_acc = sum([m * int(c1 == c2) for (c1, c2, m) in zip(x, pred, mask)]) / sum(mask)
        char_acc = sum(mask * torch.eq(x, pred)) / sum(mask)

    return str_acc, char_acc


def evaluation(args, model, tokenizer, do_print=False):
    lengths = np.arange(args.min_eval_length, args.max_eval_length, 5)
    # lengths = np.arange(args.min_eval_length, args.max_eval_length)

    attention_mask = torch.ones((args.sequence_length, args.sequence_length))
    attention_mask = (torch.triu(attention_mask, diagonal=0) - torch.triu(attention_mask, diagonal=args.window)).T.to('cuda' if torch.cuda.is_available() else 'cpu')
    
    str_acc_mean_list = []
    str_acc_std_list = []
    char_accuracy_list = []
    if do_print:
        print("\n")
    
    for length in lengths:
        str_acc_batch = np.zeros(args.eval_num_batches)
        char_acc_mean = 0

        for j in range(args.eval_num_batches):
            long_dataset = get_eval_dataset(args, tokenizer, length, length)     
            batch = next(iter(long_dataset))
            
            x = batch['input_ids'].to('cuda' if torch.cuda.is_available() else 'cpu')
            y = batch['output_ids'].to('cuda' if torch.cuda.is_available() else 'cpu')
            mask = batch['mask'].to('cuda' if torch.cuda.is_available() else 'cpu')
            
            with torch.no_grad():
                # prediction
                # if args.model=="mamba":
                #     logits = model(x)[0]
                # else:
                logits = model(x, attention_mask=attention_mask, return_dict=True)['logits']
                
                # greedy decoding
                pred = torch.argmax(logits, dim=-1)

                # evaluation
                for i in range(len(x)):
                    str_acc, char_acc = get_score(args, tokenizer, y, pred, mask, i) 

                    str_acc_batch[j] += str_acc
                    char_acc_mean    += char_acc

        if do_print:
            print("v"*100)
            print("COMPLETE EXAMPLE: ", tokenizer.to_string(batch['input'][0], pytorch=False))
            # print("EXAMPLE:", batch['input'][0])
            print("-"*100)
            print("INPUT EXAMPLE: ", tokenizer.to_string(batch['input_ids'][0][batch['mask'][0]==1]))
            print("OUTPUT EXAMPLE:", tokenizer.to_string(batch['output_ids'][0][batch['mask'][0]==1]))
            # print("TOKENIZED:", batch['input_ids'][0][batch['mask'][0]==1])
            print("-"*100)
            print("PREDICTION:    ", tokenizer.to_string(pred[0][batch['mask'][0]==1]))
            # print("PREDICTION:", pred[-1][batch['mask'][0]==1])
            print("^"*100)
        

        str_acc_batch = str_acc_batch/args.eval_batch_size
        # str_acc_batch = str_acc_batch/len(x)
        mean_str_acc = float(np.mean(str_acc_batch))
        std_str_acc = float(np.std(str_acc_batch))

        str_acc_mean_list.append(mean_str_acc)
        str_acc_std_list.append(std_str_acc)

        mean_char_acc = char_acc_mean/(args.eval_batch_size*args.eval_num_batches)
        # mean_char_acc = char_acc_mean/(len(x)*args.eval_num_batches)
        char_accuracy_list.append(mean_char_acc.item())
        
        if do_print:
            print(f"{args.eval_task}; len {length}: {mean_str_acc} +- {std_str_acc}; char: {mean_char_acc}")
    
    if do_print:
        print("\n")        
    
    return str_acc_mean_list, str_acc_std_list, char_accuracy_list



