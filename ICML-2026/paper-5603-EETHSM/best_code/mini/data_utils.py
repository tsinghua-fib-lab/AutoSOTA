import numpy as np
import torch
import string
import torch.nn.functional as F
import random 
import math

from collections import defaultdict
from transformers import AutoTokenizer

#################################################################

# Tokenizer

class Tokenizer:
    def __init__(self, TO_TOKEN, vocab_tokens, number_tokens):
        
        self.TO_TOKEN = TO_TOKEN
        self.TO_STR = {v:k for k, v in TO_TOKEN.items()}

        self.vocab = np.array(list(TO_TOKEN.keys()))
        
        self.vocab_tokens = vocab_tokens
        self.number_tokens = number_tokens
        self.num_vocab = len(self.vocab_tokens)
        self.num_numbers = len(self.number_tokens)
        
        self.bos_token = self.TO_TOKEN['<bos>']
        self.eos_token = self.TO_TOKEN['<eos>']
        self.null = '<null>'

        # Human readible printing
        vocab_part = {self.TO_TOKEN[k]: v for (k, v) in zip(self.vocab_tokens, string.ascii_lowercase[:self.num_vocab])}
        number_part = {self.TO_TOKEN[t]: t[1:] for t in self.number_tokens}
        
        self.TO_STRING = {**vocab_part, **number_part}
        self.TO_STRING[self.bos_token] = "$"
        self.TO_STRING[self.eos_token] = "."
        self.TO_STRING[self.TO_TOKEN[self.null]] = "_"

    def __call__(self, x):
        encoded = [self.TO_TOKEN[c] for c in x]
        return torch.tensor(encoded, dtype=torch.int64)

    def decode(self, x):
        x = x.detach().cpu().numpy()
        decoded = [str(t) if t not in self.TO_STR else self.TO_STR[t] for t in x]
        return decoded

    def __len__(self):
        return len(self.TO_TOKEN)

    def to_string(self, x, pytorch=True):
        if pytorch:
            return "".join([self.TO_STRING[t.item()] for t in x])
        else:
            return "".join([self.TO_STRING[self.TO_TOKEN[t]] for t in x])


def get_tokenizer(args):
    if args.model == "pretrained":
        tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)
        return tokenizer

    vocab_tokens = ["V%d" % i for i in range(args.num_vocab)]
    
    if args.train_task in ["var-copy", "var-copy-rep"]:
        number_tokens = ["#%d" % (5+i) for i in range(args.num_numbers)]
    else:
        number_tokens = ["#%d" % i for i in range(args.num_numbers)]

    vocab = vocab_tokens + number_tokens + ["<bos>", "<eos>", "<null>"]

    TO_TOKEN = dict(zip(vocab, range(len(vocab))))

    tokenizer = Tokenizer(TO_TOKEN, vocab_tokens, number_tokens)
    
    return tokenizer

#################################################################

# Sequence Generation

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


def force_args(args):
    if args.train_task == "var-copy":
        pass

    if args.train_task == "var-copy-rep":
        pass
        
    if args.train_task in ["decode-recall", "decode-recall-last"]:
        args.num_numbers = 2
        args.num_vocab = int(2 ** math.floor(math.log(args.num_vocab) / math.log(2)))

    if args.train_task == "assoc-recall":
        args.num_numbers = 0

    if args.train_task == "assoc-recall-mk":
        size_key = 2
        
        args.num_numbers = 0
        args.num_vocab = 1 + int(args.num_vocab ** (1./size_key))

    if args.train_task == "addition":
        args.num_numbers = 10 # Decimal addition
        args.num_vocab = 2 # For +, =

        args.min_train_length = 3*args.min_train_length+3
        args.max_train_length = 3*args.max_train_length+3
        args.min_eval_length = 3*args.min_eval_length+3
        args.max_eval_length = 3*args.max_eval_length+3


task_choices = ["var-copy", "var-copy-rep", "decode-recall", "decode-recall-last", "assoc-recall", "assoc-recall-mk", "addition"]


def generate_seq_and_mask(tokenizer, length, task, p=0.2):
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

        input_seq = ["<bos>"] + input_seq + ["<eos>"]
        output_seq = ["<bos>"] + output_seq + ["<eos>"]
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
        
        input_seq = rand_seq(tokenizer, length, num_vocab, 0, p_numbers=0.2) 
        output_seq = ["<null>" for _ in range(len(input_seq))]

        assoc = defaultdict(lambda: "<null>")

        for i in range(len(output_seq)):
            if i > size_key:
                key = tuple(input_seq[i-size_key:i])
                assoc[key] = input_seq[i]

            if i+1 > size_key:
                key = tuple(input_seq[i-size_key+1:i+1])
                output_seq[i] = assoc[key]

    elif task == "addition":
        len_number = (length+1) // 3 - 1
        max_num = num_numbers ** len_number

        num1 = np.random.randint(0, max_num)
        num2 = np.random.randint(0, max_num)
        num3 = (num1 + num2) % max_num
        
        input_seq = ["<null>" for _ in range(length)]
        output_seq = ["<null>" for _ in range(length)]

        for i in range(len_number-1, -1, -1):
            input_seq[i] = "#%d" % (num1 % num_numbers)
            num1 = num1 // num_numbers

        input_seq[len_number] = "V0"

        for i in range(2*len_number, len_number, -1):
            input_seq[i] = "#%d" % (num2 % num_numbers)
            num2 = num2 // num_numbers

        input_seq[2*len_number+1] = "V1"

        for i in range(3*len_number+1, 2*len_number+1, -1):
            input_seq[i] = "#%d" % (num3 % num_numbers)
            output_seq[i-1] = "#%d" % (num3 % num_numbers)
            
            num3 = num3 // num_numbers

    else:
        print("Task name:", task)
        assert False # Not implemented

    # Create the mask
    mask = [0 if i in ["<bos>", "<eos>", "<null>"] else 1 for i in output_seq]

    return input_seq, output_seq, mask


#################################################################

# Datasets

class Dataset:
    def __init__(self, 
                 tokenizer, 
                 train_task="var_copy", 
                 sequence_length=220, 
                 min_subseq_length=20, 
                 max_subseq_length=50, 
                 num_examples=1000, 
                 batch_size=8,
                 p=0.2): 
        
        self.tokenizer = tokenizer
        self.train_task = train_task
        self.num_vocab = self.tokenizer.num_vocab
        self.num_numbers = self.tokenizer.num_numbers

        self.sequence_length = sequence_length
        self.min_subseq_length = min_subseq_length
        self.max_subseq_length = max_subseq_length
        self.num_examples = num_examples
        self.batch_size = batch_size
        self.p = p

    def __len__(self):
        return self.num_examples

    def __getitem__(self, idx):
        batch = {'input': [], 'input_ids': [], 'output': [], 'output_ids': [], 'mask': []}

        for _ in range(self.batch_size):
            
            # Fill the context with subsequences of the desired task
            prospective_len = 0
            input_seq = []
            output_seq = []
            mask = []
            while prospective_len < self.sequence_length:
                # Sample for the task
                length = np.random.randint(self.min_subseq_length, self.max_subseq_length+1)
                input_sample, output_sample, mask_sample = generate_seq_and_mask(self.tokenizer, length, self.train_task, self.p)

                # Add the sample to the context
                if prospective_len + len(input_sample) <= self.sequence_length:
                    prospective_len += len(input_sample)
                    input_seq += input_sample
                    output_seq += output_sample
                    mask += mask_sample
                # Not enough room for another sample
                else:
                    remaining_len = self.sequence_length - prospective_len
                    remaining_mask_len = self.sequence_length - prospective_len
                    input_seq += input_sample[:remaining_len]
                    output_seq += output_sample[:remaining_len]
                    mask += [0] * (remaining_mask_len) # Just mask it
                    break
                    
            # Add the sequence to the sampled dataset
            assert len(input_seq) == len(mask)
            input_ids = self.tokenizer(input_seq)
            output_ids = self.tokenizer(output_seq)
            mask = torch.tensor(mask)
            
            batch['input'].append(input_seq)
            batch['input_ids'].append(input_ids)
            batch['output'].append(output_seq)
            batch['output_ids'].append(output_ids)
            batch['mask'].append(mask)
        
        batch['input_ids'] = torch.stack(batch['input_ids'], dim=0)
        batch['output_ids'] = torch.stack(batch['output_ids'], dim=0)
        batch['mask'] = torch.stack(batch['mask'], dim=0)
        return batch


class EvalDataset:
    def __init__(self, 
                 tokenizer, 
                 train_task="var_copy", 
                 sequence_length=220, 
                 min_subseq_length=20, 
                 max_subseq_length=50, 
                 num_examples=1000, 
                 batch_size=8,
                 p=0.2): 
        
        self.tokenizer = tokenizer
        self.train_task = train_task

        self.sequence_length = sequence_length
        self.min_subseq_length = min_subseq_length
        self.max_subseq_length = max_subseq_length
        self.num_examples = num_examples
        self.batch_size = batch_size
        self.p = p

    def __len__(self):
        return self.num_examples

    def __getitem__(self, idx):
        batch = {'input': [], 'input_ids': [], 'output': [], 'output_ids': [], 'mask': []}

        for _ in range(self.batch_size):
            
            # Fill the context with subsequences of the desired task
            prospective_len = 0
            input_seq = []
            output_seq = []
            mask = []

            # Sample for the task
            length = np.random.randint(self.min_subseq_length, self.max_subseq_length+1)
            input_seq, output_seq, mask = generate_seq_and_mask(self.tokenizer, length, self.train_task, self.p)

            # DO NOT REPLACE
            # Fill the context with null tokens
            input_seq += ["<null>"] * (self.sequence_length - len(input_seq))
            output_seq += ["<null>"] * (self.sequence_length - len(output_seq))
            mask += [0] * (self.sequence_length - len(mask))
                    
            # Add the sequence to the sampled dataset
            assert len(input_seq) == len(mask)
            input_ids = self.tokenizer(input_seq)
            output_ids = self.tokenizer(output_seq)
            mask = torch.tensor(mask)
            
            batch['input'].append(input_seq)
            batch['input_ids'].append(input_ids)
            batch['output'].append(output_seq)
            batch['output_ids'].append(output_ids)
            batch['mask'].append(mask)
        
        batch['input_ids'] = torch.stack(batch['input_ids'], dim=0)
        batch['output_ids'] = torch.stack(batch['output_ids'], dim=0)
        batch['mask'] = torch.stack(batch['mask'], dim=0)
        return batch


#################################################################

# Util functions

def get_train_dataset(args, tokenizer):
    train_dataset = Dataset(
        tokenizer=tokenizer,
        train_task=args.train_task,

        sequence_length=args.sequence_length,
        min_subseq_length=args.min_train_length,
        max_subseq_length=args.max_train_length,
        num_examples=args.num_examples,
        batch_size=args.train_batch_size,
        p=args.p
    )
    
    return train_dataset


def get_eval_dataset(args, tokenizer, min_length, max_length):
    eval_dataset = EvalDataset(
        tokenizer=tokenizer,
        train_task=args.train_task,

        sequence_length=args.sequence_length,
        min_subseq_length=min_length,
        max_subseq_length=max_length,
        num_examples=args.num_examples,
        batch_size=args.eval_batch_size,
        p=args.p
    )
    
    return eval_dataset
