# Sequence Generation
import numpy as np
import math
from collections import defaultdict


task_choices = ["var-copy", "var-copy-rep", "decode-recall", "decode-recall-last", "assoc-recall", "assoc-recall-mk"]


def force_args(args):
    if args.train_task in ["var-copy", "var-copy-rep"]:
        pass

    if args.train_task in ["decode-recall", "decode-recall-last"]:
        args.num_numbers = 2
        args.num_vocab = int(2 ** math.floor(math.log(args.num_vocab) / math.log(2)))

    if args.train_task == "assoc-recall":
        args.num_numbers = 0

    if args.train_task == "assoc-recall-mk":
        args.num_numbers = 0
        # args.num_vocab = 1 + int(args.num_vocab ** (1./size_key))


def generate_seq(tokenizer, length, task, p=0.2):
    num_vocab = tokenizer.num_vocab
    num_numbers = tokenizer.num_numbers
    
    if task == "var-copy":
        # Start with num_numbers vocab tokens
        input_seq = rand_seq(tokenizer, length, num_vocab, num_numbers, p_numbers=p) 
        # input_seq = rand_seq(tokenizer, length, num_vocab, num_numbers) 
        
        nums = [(i, int(c[1:])) for (i, c) in enumerate(input_seq) if c in tokenizer.number_tokens]
        
        # The real task, if not degenerate
        if len(nums) > 0:
            output_seq = ["<null>"] * nums[0][0]

            for i in range(len(nums)-1):
                if nums[i][0]-nums[i][1] < 0:
                    if nums[i+1][0]-nums[i][1] < 0:
                        output_seq += ["<null>"] * (nums[i+1][0]-nums[i][0])
                    else:
                        output_seq += ["<null>"] * (nums[i][1]-nums[i][0])
                        output_seq += input_seq[:nums[i+1][0]-nums[i][1]]
                else:
                    output_seq += input_seq[nums[i][0]-nums[i][1]:nums[i+1][0]-nums[i][1]]

            if nums[-1][0]-nums[-1][1] < 0:
                output_seq += ["<null>"] * (nums[-1][1]-nums[-1][0])
                output_seq += input_seq[:-nums[-1][1]]
            else:
                output_seq += input_seq[nums[-1][0]-nums[-1][1]:-nums[-1][1]]
        else:
            output_seq = ["<null>"] * length

        # output_seq = ["<bos>"] + output_seq[:length] + ["<eos>"]

    elif task == "var-copy-rep":
        # Start with num_numbers vocab tokens
        # input_seq = rand_seq(tokenizer, length, num_vocab, num_numbers, p_numbers=p) 
        input_seq = rand_seq_special(tokenizer, length, num_vocab, num_numbers, p_numbers=p, special_type="repetitive_vocab") 
        
        nums = [(i, int(c[1:])) for (i, c) in enumerate(input_seq) if c in tokenizer.number_tokens]
        
        # The real task, if not degenerate
        if len(nums) > 0:
            output_seq = ["<null>"] * nums[0][0]

            for i in range(len(nums)-1):
                if nums[i][0]-nums[i][1] < 0:
                    if nums[i+1][0]-nums[i][1] < 0:
                        output_seq += ["<null>"] * (nums[i+1][0]-nums[i][0])
                    else:
                        output_seq += ["<null>"] * (nums[i][1]-nums[i][0])
                        output_seq += input_seq[:nums[i+1][0]-nums[i][1]]
                else:
                    output_seq += input_seq[nums[i][0]-nums[i][1]:nums[i+1][0]-nums[i][1]]

            if nums[-1][0]-nums[-1][1] < 0:
                output_seq += ["<null>"] * (nums[-1][1]-nums[-1][0])
                output_seq += input_seq[:-nums[-1][1]]
            else:
                output_seq += input_seq[nums[-1][0]-nums[-1][1]:-nums[-1][1]]
        else:
            output_seq = ["<null>"] * length

        input_seq = ["<bos>"] + input_seq + ["<eos>"]
        output_seq = ["<bos>"] + output_seq + ["<eos>"]
        # output_seq = ["<bos>"] + output_seq[:length] + ["<eos>"]

    elif task == "decode-recall":
        input_seq = rand_seq(tokenizer, length, num_vocab, num_numbers, p_numbers=p) 
        output_seq = [None for _ in range(len(input_seq))]

        assoc = {v: "<null>" for v in tokenizer.vocab}
        s = 0
        for i in range(len(output_seq)):
            if i != 0:
                assoc[input_seq[i-1]] = input_seq[i]
            
            if input_seq[i][0] == '#':
                # s = (2 * s + int(input_seq[i][1:])) % num_numbers
                s = (2 * s + int(input_seq[i][1:])) % num_vocab

            # if i-s < 0:
            #     output_seq[i] = "<null>"
            # else:
            #     output_seq[i] = input_seq[i-s]

            output_seq[i] = assoc["V%d" % s]

    elif task == "decode-recall-last":
        input_seq = rand_seq(tokenizer, length, num_vocab, num_numbers, p_numbers=0) 
        output_seq = ["<null>" for _ in range(len(input_seq))]

        n_bits = int(math.log(num_vocab)/math.log(2))

        target = np.random.randint(0, num_vocab)
        temp = target
        for i in range(length-1, length-1-n_bits, -1):
            input_seq[i] = "#%d" % (temp % 2)
            temp = temp // 2

        try:
            i = length-2-n_bits - input_seq[-2-n_bits::-1].index("V%d" % target)
            output_seq[-1] = input_seq[i+1]
        except ValueError:
            pass

    elif task == "assoc-recall":
        input_seq = rand_seq(tokenizer, length, num_vocab, num_numbers, p_numbers=0.2) 
        output_seq = [None for _ in range(len(input_seq))]

        assoc = {v: "<null>" for v in tokenizer.vocab}

        for i in range(len(output_seq)):
            if i != 0:
                assoc[input_seq[i-1]] = input_seq[i]

            output_seq[i] = assoc[input_seq[i]]

    elif task == "assoc-recall-mk":
        size_key = 2
        
        input_seq = rand_seq(tokenizer, length, num_vocab, 0, p_numbers=0.0) 
        output_seq = ["<null>" for _ in range(len(input_seq))]

        assoc = defaultdict(lambda: "<null>")

        for i in range(len(output_seq)):
            if i > size_key:
                key = tuple(input_seq[i-size_key:i])
                assoc[key] = input_seq[i]

            if i+1 > size_key:
                key = tuple(input_seq[i-size_key+1:i+1])
                output_seq[i] = assoc[key]

    else:
        print("Task name:", task)
        assert False # Not implemented

    return input_seq, output_seq

################################################################################################

# SEQUENCE GENERATION HELPERS

def rand_seq(tokenizer, length, num_vocab, num_numbers, p_numbers=-1):
    if p_numbers == -1:
        p_numbers = num_numbers / (num_vocab + num_numbers)

    if num_numbers != 0:
        props = {"V": (1-p_numbers)/num_vocab, "#": p_numbers/num_numbers, "<": 0}  
    else:
        props = {"V": 1/num_vocab, "#": 0, "<": 0}  
    props = np.array([props[i[0]] for i in tokenizer.vocab])

    return np.random.choice(tokenizer.vocab, size=length, p=props).tolist()


# For other special generations
def rand_seq_special(tokenizer, length, num_vocab, num_numbers, p_numbers=-1, special_type=None):
    if special_type == "repetitive_vocab":
        if p_numbers == -1:
            p_numbers = num_numbers / (num_vocab + num_numbers)
    
        if num_numbers != 0:
            props = {"V": 0, "#": p_numbers/num_numbers, "<": 0} 
        else:
            props = {"V": 0, "#": 0, "<": 0}  
        if num_numbers != 0:
            props_V0 = (1-p_numbers)
        else:
            props_V0 = 1
        props = np.array([props[i[0]] if i != "V0" else props_V0 for i in tokenizer.vocab])
        
        tile_length = 3

        props_tile = {"V": 1./num_vocab, "#": 0, "<": 0} 
        props_tile = np.array([props_tile[i[0]] for i in tokenizer.vocab])

        ret_seq = np.random.choice(tokenizer.vocab, size=length, p=props)
        ret_seq2 = np.tile(np.random.choice(tokenizer.vocab, size=tile_length, p=props_tile), (length // tile_length + 1))[:length]

        return np.where(ret_seq == "V0", ret_seq2, ret_seq).tolist()

    else:
        assert False, "Not implemented"