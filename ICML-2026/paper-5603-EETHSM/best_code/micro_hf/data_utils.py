import numpy as np
import torch
from torch.utils.data import Dataset
import string
from generate import generate_seq

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

    # def to_string(self, x, pytorch=True):
    #     if pytorch:
    #         return "".join([self.TO_STRING[t.item()] for t in x])
    #     else:
    #         return "".join([self.TO_STRING[self.TO_TOKEN[t]] for t in x])


def get_tokenizer(args):
    if args.model == "pretrained":
        tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)
        return tokenizer

    # Vocab tokens look like 'V' + number
    vocab_tokens = ["V%d" % i for i in range(args.num_vocab)]
    
    # Create number tokens. Variable copy tasks start with 5, not 0
    if args.train_task.startswith("var-copy"):
        number_tokens = ["#%d" % (5+i) for i in range(args.num_numbers)]
    else:
        number_tokens = ["#%d" % i for i in range(args.num_numbers)]

    vocab = vocab_tokens + number_tokens + ["<bos>", "<eos>", "<null>"]

    TO_TOKEN = dict(zip(vocab, range(len(vocab))))

    tokenizer = Tokenizer(TO_TOKEN, vocab_tokens, number_tokens)
    
    return tokenizer


#################################################################

# Datasets

class TrainDataset(Dataset):
    def __init__(self, 
                 tokenizer, 
                 task="var_copy", 
                 sequence_length=220, 
                 min_subseq_length=20, 
                 max_subseq_length=50, 
                 num_examples=1000, 
                 batch_size=8,
                 p=0.2,
                 pack_examples=False,
                 mixed=False): 
        
        self.tokenizer = tokenizer
        self.task = task
        self.num_vocab = self.tokenizer.num_vocab
        self.num_numbers = self.tokenizer.num_numbers

        self.sequence_length = sequence_length
        self.min_subseq_length = min_subseq_length
        self.max_subseq_length = max_subseq_length
        self.num_examples = num_examples
        self.batch_size = batch_size
        self.p = p
        self.pack_examples = pack_examples
        self.mixed = mixed

    def __len__(self):
        return self.num_examples

    def __getitem__(self, idx):
        if idx >= self.num_examples:
            raise IndexError("Index out of range in dataset")
        batch = {'input': [], 'input_ids': [], 'output': [], 'output_ids': [], 'mask': []}

        for _ in range(self.batch_size):
            
            # Fill the context with subsequences of the desired task
            prospective_len = 0
            input_seq = []
            output_seq = []
            mask = []

            if self.pack_examples:
                while prospective_len < self.sequence_length:
                    # Sample for the task
                    length = np.random.randint(self.min_subseq_length, self.max_subseq_length+1)
                    input_sample, output_sample = generate_seq_and_mask(self.tokenizer, length, self.task, self.p, mixed=self.mixed)

                    input_sample = ["<bos>"] + input_sample + ["<eos>"]
                    output_sample = ["<bos>"] + output_sample + ["<eos>"]
                    mask_sample = [0 if i in ["<bos>", "<eos>", "<null>"] else 1 for i in output_seq]

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
            
            else:
                input_seq, output_seq = generate_seq(self.tokenizer, self.sequence_length, self.task, self.p, mixed=self.mixed)
                mask = [0 if i in ["<bos>", "<eos>", "<null>"] else 1 for i in output_seq]
                    
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


class EvalDataset(Dataset):
    def __init__(self, 
                 tokenizer, 
                 train_task="var_copy", 
                 sequence_length=220, 
                 min_subseq_length=20, 
                 max_subseq_length=50, 
                 num_examples=1000, 
                 batch_size=8,
                 p=0.2,
                 mixed=False): 
        
        self.tokenizer = tokenizer
        self.train_task = train_task

        self.sequence_length = sequence_length
        self.min_subseq_length = min_subseq_length
        self.max_subseq_length = max_subseq_length
        self.num_examples = num_examples
        self.batch_size = batch_size
        self.p = p
        self.mixed = mixed

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
            input_seq, output_seq = generate_seq(self.tokenizer, length, self.train_task, self.p, mixed=self.mixed)
            mask = [0 if i in ["<bos>", "<eos>", "<null>"] else 1 for i in output_seq]

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
    train_dataset = TrainDataset(
        tokenizer=tokenizer,
        task=args.train_task,

        sequence_length=args.sequence_length,
        min_subseq_length=args.min_train_length,
        max_subseq_length=args.max_train_length,
        num_examples=args.num_examples,
        batch_size=args.train_batch_size,
        p=args.p,
        pack_examples=args.pack_examples,
        mixed=args.mixed
    )
    
    return train_dataset


def get_eval_dataset(args, tokenizer, min_length, max_length):
    eval_dataset = EvalDataset(
        tokenizer=tokenizer,
        train_task=args.train_task,

        sequence_length=args.sequence_length,
        min_subseq_length=min_length,
        max_subseq_length=max_length,
        num_examples=args.num_eval_examples,
        batch_size=args.eval_batch_size,
        p=args.eval_p,
        mixed=args.mixed
    )
    
    return eval_dataset
