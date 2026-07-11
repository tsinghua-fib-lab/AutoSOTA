import torch
from copy import deepcopy
import random
from torch.utils.data import Dataset, IterableDataset
from typing import Optional
from collections import Counter
import string
import math
from abc import ABC, abstractmethod
import numpy as np
from itertools import accumulate
import re



class customTokenizer():
    def __init__(self, vocab: list[str]):
        normal_tkn_num = len(vocab) # each element is a token

        self.bos_token = "<bos>"
        self.sep_token = "<sep>"
        self.eos_token = "<eos>"
        self.pad_token = "<pad>"
        self.bos_token_id = normal_tkn_num
        self.sep_token_id = normal_tkn_num + 1
        self.eos_token_id = normal_tkn_num + 2
        self.pad_token_id = normal_tkn_num + 3
        self.special_token_ids = [self.bos_token_id, self.sep_token_id, self.eos_token_id, self.pad_token_id]
        self.special_tokens = [self.bos_token, self.sep_token, self.eos_token, self.pad_token]
        assert all(t not in vocab for t in self.special_tokens)
        
        # self.vocab = {"0": 0, "1": 1}
        self.vocab = {t: i for i, t in enumerate(vocab)}
        self.vocab[self.bos_token] = self.bos_token_id
        self.vocab[self.sep_token] = self.sep_token_id
        self.vocab[self.eos_token] = self.eos_token_id
        self.vocab[self.pad_token] = self.pad_token_id

        self.vocab_inv = {v: k for k, v in self.vocab.items()}
        self.padding_side = "right"

    def __call__(self, strings: list[str] | str, **kwargs):
        # this func is not used, since the data generator does not generate str
        # string is tokenized by white space
        if type(strings) == str:
            strings = [strings]
        ids = []
        strings = [s.split(" ") for s in strings]
        max_len = max(map(lambda x: len(x), strings))
        for s in strings:
            ids.append( list(map(lambda x: self.vocab[x], s)) + [self.pad_token_id] * (max_len-len(s)) )

        return {"input_ids": torch.LongTensor(ids)}

    def convert_ids_to_tokens(self, ids: list[int], rm_special=False):
        if rm_special:
            return [self.vocab_inv[i] for i in ids if i not in self.special_token_ids]
        else:
            return list(map(lambda x: self.vocab_inv[x], ids))

    def __len__(self):
        return len(self.vocab)
   

class UniqueCopyDataset(IterableDataset):
    def __init__(self, tokenizer: customTokenizer, length_range: tuple[int, int], max_test_length: int):
        super().__init__()
        self.tokenizer = tokenizer 
        self.range_min, self.range_max = length_range
        self.range_min = max(1, self.range_min)
        self.max_test_length = max_test_length
        assert len(tokenizer) - 4 >= max_test_length
        assert (max_test_length >= self.range_max) or (max_test_length == -1)    # the pos emb is initialized based on max_test_length

    def __iter__(self):
        while True:
            length = random.randint(self.range_min, self.range_max)     # length of string to be copied
            
            temp = random.sample(range(len(self.tokenizer)-4), length)
            instance = [self.tokenizer.bos_token_id]
            instance.extend(temp)
            instance.append(self.tokenizer.sep_token_id)
            instance.extend(temp)
            instance.append(self.tokenizer.eos_token_id)

            label = deepcopy(instance)
            # setting some tokens to [pad] will make the loss on these tokens (as pred targets) be ignored
            label[:length+2] = [self.tokenizer.pad_token_id,] * (length+2)   # bos + ... + sep 
            
            if self.max_test_length != -1:
                offset = random.randint(0, (self.max_test_length - length) * 2)
            else:
                offset = 0
            pos_ids = list(range(offset, len(instance)+offset))

            yield instance, pos_ids, label


class RepeatCopyDataset(IterableDataset):
    def __init__(self, tokenizer: customTokenizer, length_range: tuple[int, int], max_test_length: int):
        super().__init__()
        self.tokenizer = tokenizer 
        self.range_min, self.range_max = length_range
        self.range_min = max(1, self.range_min)
        self.max_test_length = max_test_length
        assert (max_test_length >= self.range_max) or (max_test_length == -1)    # the pos emb is initialized based on max_test_length

    def __iter__(self):
        while True:
            length = random.randint(self.range_min, self.range_max)     # length of string to be copied
            
            temp = random.choices(range(len(self.tokenizer)-4), k=length)
            instance = [self.tokenizer.bos_token_id]
            instance.extend(temp)
            instance.append(self.tokenizer.sep_token_id)
            instance.extend(temp)
            instance.append(self.tokenizer.eos_token_id)

            label = deepcopy(instance)
            # setting some tokens to [pad] will make the loss on these tokens (as pred targets) be ignored
            label[:length+2] = [self.tokenizer.pad_token_id,] * (length+2)   # bos + ... + sep 
            
            if self.max_test_length != -1:
                offset = random.randint(0, (self.max_test_length - length) * 2)
            else:
                offset = 0
            pos_ids = list(range(offset, len(instance)+offset))

            yield instance, pos_ids, label


class ParityDataset(IterableDataset):
    def __init__(self, tokenizer: customTokenizer, length_range: tuple[int, int], max_test_length: int):
        super().__init__()
        self.tokenizer = tokenizer 
        self.range_min, self.range_max = length_range
        self.max_test_length = max_test_length
        assert (max_test_length >= self.range_max) or (max_test_length == -1)    # the pos emb is initialized based on max_test_length

    def __iter__(self):
        while True:
            length = random.randint(self.range_min, self.range_max) 
            num_ones = random.randint(0, length)
            temp = [self.tokenizer.vocab["1"]] * num_ones + [self.tokenizer.vocab["0"]] * (length - num_ones)
            random.shuffle(temp)
            ans = self.tokenizer.vocab[str(num_ones % 2)]   # TODO

            instance = [self.tokenizer.bos_token_id]
            instance.extend(temp)
            instance.append(self.tokenizer.sep_token_id)
            instance.append(ans)
            # instance.append(self.tokenizer.eos_token_id)

            label = deepcopy(instance)
            # setting some tokens to [pad] will make the loss on these tokens (as pred targets) be ignored
            label[:length+2] = [self.tokenizer.pad_token_id,] * (length+2)   # bos + bits.. + sep 
            
            if self.max_test_length != -1:
                offset = random.randint(0, self.max_test_length - length)
            else:
                offset = 0
            pos_ids = list(range(offset, len(instance)+offset))

            yield instance, pos_ids, label


class AdditionDataset(IterableDataset):
    def __init__(self, tokenizer: customTokenizer, length_range: tuple[int, int], max_test_length: int):
        super().__init__()
        self.tokenizer = tokenizer 
        self.range_min, self.range_max = length_range
        self.range_min = max(4, self.range_min)
        self.max_test_length = max_test_length
        assert (max_test_length >= self.range_max) or (max_test_length == -1)    # the pos emb is initialized based on max_test_length

    def __iter__(self):
        while True:
            length = random.randint(self.range_min, self.range_max)     # length of string to be copied

            len_operand1 = random.randint(1, length-2)
            len_operand2 = length - 1 - len_operand1
            
            if len_operand1 > 1:
                operand1 = ["1"] + random.choices(["0", "1"], k=len_operand1-1)
            else:
                operand1 = random.choices(["0", "1"], k=1)
            if len_operand2 > 1:
                operand2 = ["1"] + random.choices(["0", "1"], k=len_operand2-1)
            else:
                operand2 = random.choices(["0", "1"], k=1)

            ans = int("0b" + "".join(operand1), 2) + int("0b" + "".join(operand2), 2)
            ans = list(bin(ans)[2:])

            instance = [self.tokenizer.bos_token]
            instance.extend(operand1)
            instance.append("+")
            instance.extend(operand2)
            instance.append(self.tokenizer.sep_token)   # TODO
            instance.extend(ans)
            instance.append(self.tokenizer.eos_token)

            instance = list(map(lambda x: self.tokenizer.vocab[x], instance))

            label = deepcopy(instance)
            # setting some tokens to [pad] will make the loss on these tokens (as pred targets) be ignored
            label[:length+2] = [self.tokenizer.pad_token_id,] * (length+2)   # bos + bits.. + sep 
            
            if self.max_test_length != -1:
                offset = random.randint(0, (self.max_test_length+1)*2 - len(instance))
            else:
                offset = 0
            pos_ids = list(range(offset, len(instance)+offset))

            yield instance, pos_ids, label


class SortDataset(IterableDataset):
    def __init__(self, tokenizer: customTokenizer, length_range: tuple[int, int], max_test_length: int):
        super().__init__()
        self.tokenizer = tokenizer 
        self.range_min, self.range_max = length_range
        self.range_min = max(1, self.range_min)
        self.max_test_length = max_test_length
        assert len(tokenizer) - 4 >= max_test_length
        assert (max_test_length >= self.range_max) or (max_test_length == -1)    # the pos emb is initialized based on max_test_length

    def __iter__(self):
        while True:
            length = random.randint(self.range_min, self.range_max)     # length of string to be copied

            temp = random.sample(range(len(self.tokenizer)-4), length)
            instance = [self.tokenizer.bos_token_id]
            instance.extend(temp)
            instance.append(self.tokenizer.sep_token_id)
            instance.extend(sorted(temp))
            instance.append(self.tokenizer.eos_token_id)

            label = deepcopy(instance)
            # setting some tokens to [pad] will make the loss on these tokens (as pred targets) be ignored
            label[:length+2] = [self.tokenizer.pad_token_id,] * (length+2)   # bos + bits.. + sep 
            
            if self.max_test_length != -1:
                offset = random.randint(0, (self.max_test_length - length) * 2)
            else:
                offset = 0
            pos_ids = list(range(offset, len(instance)+offset))

            yield instance, pos_ids, label
    

class BinaryMajorityInterleaveDataset(IterableDataset):
    def __init__(self, tokenizer: customTokenizer, length_range: tuple[int, int], max_test_length: int, period: Optional[int] = None):
        super().__init__()
        self.tokenizer = tokenizer 
        assert len(tokenizer) == 6, len(tokenizer)
        self.range_min, self.range_max = length_range
        self.range_min = max(3, self.range_min)
        self.max_test_length = max_test_length
        assert (max_test_length >= self.range_max) or (max_test_length == -1)    # the pos emb is initialized based on max_test_length
        self.period = period
        if not period:
            self.period = 3

    def __iter__(self):
        while True:
            total_length = random.randint(self.range_min, self.range_max)
            length = round(total_length / self.period)
            if length * self.period > self.range_max:
                length -= 1
            if length * self.period < self.range_min:
                length += 1
            
            instances = []
            answers = []
            for i in range(self.period):
                while True:
                    num_zero = random.randint(0, length)
                    if num_zero != length-num_zero:
                        break
                instance = [0, ] * num_zero + [1, ] * (length - num_zero)
                random.shuffle(instance)
                instances.append(instance)

                ans = 0 if num_zero > length-num_zero else 1
                answers.append(ans)

            whole_instance = [val for tup in zip(*instances) for val in tup]

            whole_instance.insert(0, self.tokenizer.bos_token_id)
            whole_instance.append(self.tokenizer.sep_token_id)
            whole_instance.extend(answers)
            whole_instance.append(self.tokenizer.eos_token_id)

            label = deepcopy(whole_instance)
            # setting some tokens to [pad] will make the loss on these tokens (as pred targets) be ignored
            label[:length*self.period+2] = [self.tokenizer.pad_token_id,] * (length*self.period+2)   # bos + bits.. + sep 
            
            if self.max_test_length != -1:
                offset = random.randint(0, self.max_test_length - length*self.period)
            else:
                offset = 0
            pos_ids = list(range(offset, len(whole_instance)+offset))

            yield whole_instance, pos_ids, label


class MajorityDataset(IterableDataset):
    def __init__(self, tokenizer: customTokenizer, length_range: tuple[int, int], max_test_length: int):
        super().__init__()
        self.tokenizer = tokenizer
        self.range_min, self.range_max = length_range
        self.range_min = max(1, self.range_min)
        self.max_test_length = max_test_length
        assert (max_test_length >= self.range_max) or (max_test_length == -1)   # the pos emb is initialized based on max_test_length

    def __iter__(self):
        while True:
            length = random.randint(self.range_min, self.range_max)
            while True:
                instance = random.choices(range(len(self.tokenizer)-4), k=length)
                most_common = Counter(instance).most_common(2)
                if len(most_common) < 2 or most_common[0][1] > most_common[1][1]:
                    break
            ans = most_common[0][0]

            instance.insert(0, self.tokenizer.bos_token_id)
            instance.append(self.tokenizer.sep_token_id)
            instance.append(ans)
            # instance.append(self.tokenizer.eos_token_id)

            label = deepcopy(instance)
            # setting some tokens to [pad] will make the loss on these tokens (as pred targets) be ignored
            label[:length+2] = [self.tokenizer.pad_token_id,] * (length+2)   # bos + bits.. + sep 
            
            if self.max_test_length != -1:
                offset = random.randint(0, self.max_test_length - length)
            else:
                offset = 0
            pos_ids = list(range(offset, len(instance)+offset))

            yield instance, pos_ids, label


  
class BinaryMajorityDataset(IterableDataset):
    def __init__(self, tokenizer: customTokenizer, length_range: tuple[int, int], max_test_length: int):
        super().__init__()
        self.tokenizer = tokenizer 
        assert len(tokenizer) == 6
        self.range_min, self.range_max = length_range
        self.range_min = max(1, self.range_min)
        self.max_test_length = max_test_length
        assert (max_test_length >= self.range_max) or (max_test_length == -1)   # the pos emb is initialized based on max_test_length

    def __iter__(self):
        while True:
            length = random.randint(self.range_min, self.range_max)
            while True:
                num_zero = random.randint(0, length)
                if num_zero != length-num_zero:
                    break
            instance = [0, ] * num_zero + [1, ] * (length - num_zero)
            random.shuffle(instance)
            ans = 0 if num_zero > length-num_zero else 1

            instance.insert(0, self.tokenizer.bos_token_id)
            instance.append(self.tokenizer.sep_token_id)
            instance.append(ans)
            # instance.append(self.tokenizer.eos_token_id)

            label = deepcopy(instance)
            # setting some tokens to [pad] will make the loss on these tokens (as pred targets) be ignored
            label[:length+2] = [self.tokenizer.pad_token_id,] * (length+2)   # bos + bits.. + sep 
            
            if self.max_test_length != -1:
                offset = random.randint(0, self.max_test_length - length)
            else:
                offset = 0
            pos_ids = list(range(offset, len(instance)+offset))

            yield instance, pos_ids, label


class UniqueReverseDataset(IterableDataset):
    def __init__(self, tokenizer: customTokenizer, length_range: tuple[int, int], max_test_length: int):
        super().__init__()
        self.tokenizer = tokenizer 
        self.range_min, self.range_max = length_range
        self.range_min = max(1, self.range_min)
        self.max_test_length = max_test_length
        assert len(tokenizer) - 4 >= max_test_length
        assert (max_test_length >= self.range_max) or (max_test_length == -1)    # the pos emb is initialized based on max_test_length

    def __iter__(self):
        while True:
            length = random.randint(self.range_min, self.range_max)     # length of string to be copied
            
            temp = random.sample(range(len(self.tokenizer)-4), length)
            instance = [self.tokenizer.bos_token_id]
            instance.extend(temp)
            instance.append(self.tokenizer.sep_token_id)
            instance.extend(temp[::-1])
            instance.append(self.tokenizer.eos_token_id)

            label = deepcopy(instance)
            # setting some tokens to [pad] will make the loss on these tokens (as pred targets) be ignored
            label[:length+2] = [self.tokenizer.pad_token_id,] * (length+2)   # bos + ... + sep 
            
            if self.max_test_length != -1:
                offset = random.randint(0, (self.max_test_length - length) * 2)
            else:
                offset = 0
            pos_ids = list(range(offset, len(instance)+offset))

            yield instance, pos_ids, label


class UniqueBigramCopyDataset(IterableDataset):
    def __init__(self, tokenizer: customTokenizer, length_range: tuple[int, int], max_test_length: int):
        super().__init__()
        self.tokenizer = tokenizer 
        self.range_min, self.range_max = length_range
        self.range_min = max(1, self.range_min)
        self.max_test_length = max_test_length
        assert (len(tokenizer) - 4)**2+1 >= max_test_length
        assert (max_test_length >= self.range_max) or (max_test_length == -1)    # the pos emb is initialized based on max_test_length

    def __iter__(self):
        
        vocab_size = len(self.tokenizer)-4

        def get_all_bigrams(seq):
            bigrams = []
            for i in range(len(seq)-1):
                bigrams.append((seq[i], seq[i+1]))
            return bigrams

        while True:
            length = random.randint(self.range_min, self.range_max)     # length of string to be copied
            
            while True:
                temp = random.choices(range(vocab_size), k=min(vocab_size, length))
                bigrams = get_all_bigrams(temp)
                if len(set(bigrams)) == len(temp)-1:
                    break

            transition = {i: {j: 0 for j in range(vocab_size)} for i in range(vocab_size)}
            for i, j in bigrams:
                transition[i][j] = 1

            # continue generation
            while length > len(temp):
                last = temp[-1]
                candidates = []
                for j in transition[last]:
                    if transition[last][j] == 0:
                        num_next_possibilities = vocab_size - sum(transition[j].values())
                        candidates.append((j, num_next_possibilities))
                max_num = max(c[1] for c in candidates)
                next_item = random.choice([c[0] for c in candidates if c[1] == max_num])
                temp.append(next_item)
                transition[last][next_item] = 1

            instance = [self.tokenizer.bos_token_id]
            instance.extend(temp)
            instance.append(self.tokenizer.sep_token_id)
            instance.extend(temp)
            instance.append(self.tokenizer.eos_token_id)

            label = deepcopy(instance)
            # setting some tokens to [pad] will make the loss on these tokens (as pred targets) be ignored
            label[:length+2] = [self.tokenizer.pad_token_id,] * (length+2)   # bos + ... + sep 
            
            if self.max_test_length != -1:
                offset = random.randint(0, (self.max_test_length - length) * 2)
            else:
                offset = 0
            pos_ids = list(range(offset, len(instance)+offset))

            yield instance, pos_ids, label


class CountDataset(IterableDataset):
    def __init__(self, tokenizer: customTokenizer, length_range: tuple[int, int], max_test_length: int):
        super().__init__()
        self.tokenizer = tokenizer 
        self.range_min, self.range_max = length_range
        self.range_min = max(2, self.range_min)
        self.max_test_length = max_test_length
        assert len(tokenizer) - 4 >= max_test_length
        assert (max_test_length >= self.range_max) or (max_test_length == -1)    # the pos emb is initialized based on max_test_length

    def __iter__(self):
        while True:
            length = random.randint(self.range_min, self.range_max)     # length of string to be copied
            
            vocab_size = len(self.tokenizer) - 4
            start = random.randint(0, vocab_size-length)
            end = start + length - 1

            instance = [self.tokenizer.bos_token]
            instance.append(str(start))
            instance.append(str(end))
            instance.append(self.tokenizer.sep_token)
            instance.extend([str(i) for i in range(start, end+1)])
            instance.append(self.tokenizer.eos_token)
            instance = list(map(lambda x: self.tokenizer.vocab[x], instance))

            label = deepcopy(instance)
            # setting some tokens to [pad] will make the loss on these tokens (as pred targets) be ignored
            label[:4] = [self.tokenizer.pad_token_id,] * 4   # bos + ... + sep 
            
            if self.max_test_length != -1:
                offset = random.randint(0, self.max_test_length - length)
            else:
                offset = 0
            pos_ids = list(range(offset, len(instance)+offset))

            yield instance, pos_ids, label


class EvalDataset(Dataset):
    def __init__(self, d: IterableDataset, num_data: int) -> None:
        super().__init__()
        self.data = []
        for i, item in enumerate(d):
            if i >= num_data:
                break
            self.data.append(item)

    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)

def get_tokenizer_for_task(task, max_test_length):
    if task == "bin_majority":
        tokenizer = customTokenizer(["0", "1"])

    elif task == "majority":
        tokenizer = customTokenizer(list(string.ascii_lowercase))
    
    elif task == "bin_majority_interleave":
        tokenizer = customTokenizer(["0", "1"])

    elif task == "unique_copy":
        tokenizer = customTokenizer([str(i) for i in range(max_test_length)])

    elif task == "sort":
        tokenizer = customTokenizer([str(i) for i in range(max_test_length)])

    elif task == "unique_reverse":
        tokenizer = customTokenizer([str(i) for i in range(max_test_length)])

    elif task == "unique_bigram_copy":
        size = math.ceil(math.sqrt(max_test_length) * 1.25)
        tokenizer = customTokenizer([str(i) for i in range(size)])
    
    elif task == "count":
        tokenizer = customTokenizer([str(i) for i in range(max_test_length)])

    elif task == "repeat_copy":
        tokenizer = customTokenizer(["a", "b"])
    
    elif task == "parity":
        tokenizer = customTokenizer(["0", "1"])

    elif task == "addition":
        tokenizer = customTokenizer(["0", "1", "+"])      

    else:
        tokenizer = FormalLangDataset(None, (0, 0), -1, task).tokenizer
    
    return tokenizer

def get_dataset_for_task(task, tokenizer, train_length_range, max_test_length, other_config):
    if task == "bin_majority":
        train_dataset = BinaryMajorityDataset(tokenizer, train_length_range, max_test_length)

    elif task == "majority":
        train_dataset = MajorityDataset(tokenizer, train_length_range, max_test_length)
    
    elif task == "bin_majority_interleave":
        train_dataset = BinaryMajorityInterleaveDataset(tokenizer, train_length_range, max_test_length, period=other_config["period_for_data"])

    elif task == "unique_copy":
        train_dataset = UniqueCopyDataset(tokenizer, train_length_range, max_test_length)

    elif task == "sort":
        train_dataset = SortDataset(tokenizer, train_length_range, max_test_length)

    elif task == "unique_reverse":
        train_dataset = UniqueReverseDataset(tokenizer, train_length_range, max_test_length)

    elif task == "unique_bigram_copy":
        train_dataset = UniqueBigramCopyDataset(tokenizer, train_length_range, max_test_length)
    
    elif task == "count":
        train_dataset = CountDataset(tokenizer, train_length_range, max_test_length)
    
    elif task == "repeat_copy":
        train_dataset = RepeatCopyDataset(tokenizer, train_length_range, max_test_length)

    elif task == "parity":
        train_dataset = ParityDataset(tokenizer, train_length_range, max_test_length)
    
    elif task == "addition":
        train_dataset = AdditionDataset(tokenizer, train_length_range, max_test_length)

    else:
        train_dataset = FormalLangDataset(tokenizer, train_length_range, max_test_length, task)
    
    return train_dataset

def get_tokenizer_and_dataset_for_task(task, train_length_range, max_test_length, other_config):
    tokenizer = get_tokenizer_for_task(task, max_test_length)
    train_dataset = get_dataset_for_task(task, tokenizer, train_length_range, max_test_length, other_config)
    return tokenizer, train_dataset


class DFA():
    def __init__(self, sigma, Q, delta, q0, F):
        self.sigma = sigma
        self.Q = Q
        self.delta = delta
        self.q0 = q0
        self.F = F

    def __call__(self, string):
        qt = self.q0
        for symbol in string:
            qt = self.delta(qt, symbol)
        if qt in self.F:
            return True
        else:
            return False
    
# Tomita 1, 2, 4, 7
class FormalLanguage(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def belongs_to_lang(self, seq):
        pass
    
    @abstractmethod
    def generate_pos_sample(self, min_length, max_length):
        pass

    @abstractmethod
    def is_valid_length(self, length):
        pass

    @abstractmethod
    def get_next_tokens(self, pos_sample):
        pass

    @abstractmethod
    def get_bos_next_tokens(self):
        pass

class Tomita1Language(FormalLanguage):
    def __init__(self):
        super().__init__()
        self.sigma = ['0', '1']
        self.Q = ['q0', 'q1']
        self.q0 = 'q0'
        self.F = {'q0'}
        self.dfa = DFA(self.sigma, self.Q, self.transition_function, self.q0, self.F)

    def transition_function(self, q, s):
        if q == 'q0':
            if s == '0':
                return 'q1'
            if s == '1':
                return 'q0'
        if q == 'q1':
            return 'q1'

        raise RuntimeError(f"{q}, {s} is invalid")

    def belongs_to_lang(self, seq):
        return self.dfa(seq)
    
    def generate_pos_sample(self, min_length, max_length):
        length = random.randint(min_length, max_length)
        return "1"*length
    
    def is_valid_length(self, length):
        return length >= 0
    
    def get_next_tokens(self, pos_sample):
        output_tokens = []
        for letter in pos_sample:
            assert letter == "1"
            output_tokens.append(set(["1", "<eos>"]))
        return output_tokens

    def get_bos_next_tokens(self):
        return set(["1", "<eos>"])
    

class Tomita2Language(FormalLanguage):

    def __init__(self):
        super().__init__()
        self.sigma = ['0', '1']
        self.Q = ['q0', 'q1', 'q2']
        self.q0 = 'q0'
        self.F = {'q0'}
        self.dfa = DFA(self.sigma, self.Q, self.transition_function, self.q0, self.F)

    def transition_function(self, q, s):
        if q == 'q0':
            if s == '0':
                return 'q2'
            if s == '1':
                return 'q1'
        if q == 'q1':
            if s == '0':
                return 'q0'
            if s == '1':
                return 'q2'
        if q == 'q2':
            return 'q2'

        raise RuntimeError(f"{q}, {s} is invalid")
    
    def belongs_to_lang(self, seq):
        return self.dfa(seq)

    def generate_pos_sample(self, min_length, max_length):
        while not self.is_valid_length(length := random.randint(min_length, max_length)):
            pass
        string = "10"*(length // 2)
        return string
    
    def is_valid_length(self, length):
        return (length >= 0) and (length % 2 == 0)
    
    def get_next_tokens(self, pos_sample):
        output_tokens = []
        for letter in pos_sample:
            if letter == "1":
                output_tokens.append(set(["0"]))
            elif letter == "0":
                output_tokens.append(set(["1", "<eos>"]))
            else:
                raise RuntimeError
        return output_tokens

    def get_bos_next_tokens(self):
        return set(["1", "<eos>"])
    

class Tomita3Language(FormalLanguage):
    def __init__(self):
        super().__init__()
        self.sigma = ['0', '1']
        self.Q = ['q0', 'q1', 'q2', 'q3', 'q4']
        self.q0 = 'q0'
        self.F = {'q0', 'q1', 'q2'}
        self.dfa = DFA(self.sigma, self.Q, self.transition_function, self.q0, self.F)

    def transition_function(self, q, s):
        if q == 'q0':
            if s == '0':
                return 'q0'
            if s == '1':
                return 'q1'
        if q == 'q1':
            if s == '0':
                return 'q3'
            if s == '1':
                return 'q0'
        if q == 'q2':
            if s == '0':
                return 'q3'
            if s == '1':
                return 'q1'
        if q == 'q3':
            if s == '0':
                return 'q2'
            if s == '1':
                return 'q4'

        if q == 'q4':
            return 'q4'
        
        raise RuntimeError(f"{q}, {s} is invalid")

    def belongs_to_lang(self, seq):
        return self.dfa(seq)
    
    def generate_pos_sample(self, min_length, max_length):
        length = random.randint(min_length, max_length)
        string = ''
        last_toss = None
        last_one_count = 0
        while len(string) != length:
            toss = random.choice(['0', '1'])
            up_to = length - len(string) + 1
            probs = up_to - np.arange(1, up_to)
            probs = probs / probs.sum()
            char_count = np.random.choice(np.arange(1, up_to), p=probs)
            if toss == '1':
                # char_count = random.randint(0, length - len(string))  seems too easy
                string += toss * char_count
                if last_toss == '0' and char_count != 0:
                    last_one_count = char_count
                else:
                    last_one_count += char_count
            else:
                if last_toss is None or last_one_count % 2 == 0:
                    # char_count = random.randint(0, length - len(string))
                    string += toss * char_count
                else:
                    if up_to == 2:
                        toss = "1"
                        char_count = 1
                    else:
                        probs = up_to - np.arange(2, up_to, 2)
                        probs = probs / probs.sum()
                        char_count = np.random.choice(np.arange(2, up_to, 2), p=probs)
                    # choices = list(range(0, length - len(string) + 1, 2))
                    # char_count = random.choice(choices)
                    string += toss * char_count
            if char_count != 0:
                last_toss = toss

        if not self.dfa(string):
            raise RuntimeError(f"{string} is not acceptable for Tomita3")

        return string
    
    def is_valid_length(self, length):
        return length >= 0
    
    def get_next_tokens(self, pos_sample):
        prev_ones = 0
        prev_zeros = 0
        output_tokens = []
        for letter in pos_sample:
            if letter == "1" and prev_zeros > 0:
                prev_ones = 1
                prev_zeros = 0
            elif letter == "1":
                prev_ones += 1
            elif letter == "0":
                prev_zeros += 1
            else:
                raise RuntimeError("Unrecognizable letter", letter)
        
            if letter == "0" and (prev_ones % 2 == 1) and (prev_zeros % 2 == 1):
                output_tokens.append(set(["0"]))
            else:
                output_tokens.append(set(["0", "1", "<eos>"]))

        return output_tokens

    def get_bos_next_tokens(self):
        return set(["0", "1", "<eos>"])


class Tomita4Language(FormalLanguage):

    def __init__(self, ):
        super().__init__()
        self.sigma = ['0', '1']
        self.Q = ['q0', 'q1', 'q2', 'q3']
        self.q0 = 'q0'
        self.F = {'q0', 'q1', 'q2'}
        self.dfa = DFA(self.sigma, self.Q, self.transition_function, self.q0, self.F)

    def transition_function(self, q, s):
        if q == 'q0':
            if s == '0':
                return 'q1'
            if s == '1':
                return 'q0'
        if q == 'q1':
            if s == '0':
                return 'q2'
            if s == '1':
                return 'q0'
        if q == 'q2':
            if s == '0':
                return 'q3'
            if s == '1':
                return 'q0'
        if q == 'q3':
            return 'q3'
        
        raise RuntimeError(f"{q}, {s} is invalid")


    def belongs_to_lang(self, seq):
        return self.dfa(seq)

    def generate_pos_sample(self, min_length, max_length):
        length = random.randint(min_length, max_length)
        string = ''
        while len(string) < length:
            toss = random.choice(['0', '1'])
            if toss == '0':
                if len(string) >= 2 and string[-1] == '0' and string[-2] == '0':
                    continue
                else:
                    string += toss
            else:
                string += toss
        if not self.dfa(string):
            raise RuntimeError(f"{string} is not acceptable for Tomita4")
        
        return string

    def is_valid_length(self, length):
        return length >= 0
    
    def get_next_tokens(self, pos_sample):
        prev_zeros = 0
        output_tokens = []
        for letter in pos_sample:
            if letter == "1":
                prev_zeros = 0
            elif letter == "0":
                prev_zeros += 1
            else:
                raise RuntimeError("Unrecognizable letter", letter)
        
            if prev_zeros == 2:
                output_tokens.append(set(["1", "<eos>"]))
            elif prev_zeros > 2:
                raise RuntimeError("invalid input string")
            else:
                output_tokens.append(set(["0", "1", "<eos>"]))

        return output_tokens

    def get_bos_next_tokens(self):
        return set(["0", "1", "<eos>"])

class Tomita5Language(FormalLanguage):

    def __init__(self, ):
        super().__init__()
        self.sigma = ['0', '1']

    def belongs_to_lang(self, seq):
        counts = Counter(seq)
        if counts["0"] % 2 == 0 and counts["1"] % 2 == 0:
            return True
        else:
            return False

    def generate_pos_sample(self, min_length, max_length):
        while not self.is_valid_length(length := random.randint(min_length, max_length)):
            pass
        while not self.is_valid_length(length0 := random.randint(0, length)):
            pass
        string = ["0"] * length0 + ["1"] * (length - length0)
        random.shuffle(string)
        return "".join(string)

    def is_valid_length(self, length):
        return (length >= 0) and (length % 2 == 0)
    
    def get_next_tokens(self, pos_sample):
        num_ones = 0
        output_tokens = []
        for i, letter in enumerate(pos_sample):
            if letter == "1":
                num_ones += 1
        
            if ((i+1) % 2 == 0) and (num_ones % 2 == 0):
                output_tokens.append(set(["0", "1", "<eos>"]))
            else:
                output_tokens.append(set(["0", "1"]))

        return output_tokens

    def get_bos_next_tokens(self):
        return set(["0", "1", "<eos>"])
    
class Tomita6Language(FormalLanguage):

    def __init__(self, ):
        super().__init__()
        self.sigma = ['0', '1']

    def belongs_to_lang(self, seq):
        counts = Counter(seq)
        if (counts["0"] - counts["1"]) % 3 == 0:
            return True
        else:
            return False

    def generate_pos_sample(self, min_length, max_length):
        while not self.is_valid_length(length := random.randint(min_length, max_length)):
            pass
        while True:
            length0 = random.randint(0, length)
            length1 = length - length0
            if (length0 - length1) % 3 == 0:
                break
        string = ["0"] * length0 + ["1"] * (length1)
        random.shuffle(string)
        return "".join(string)

    def is_valid_length(self, length):
        return (length >= 3) and (length % 2 == 1)
    
    def get_next_tokens(self, pos_sample):
        num_ones = 0
        num_zeros = 0
        output_tokens = []
        for letter in pos_sample:
            if letter == "1":
                num_ones += 1
            elif letter == "0":
                num_zeros += 1
            else:
                raise RuntimeError
        
            if (num_zeros - num_ones) % 3 == 0:
                output_tokens.append(set(["0", "1", "<eos>"]))
            else:
                output_tokens.append(set(["0", "1"]))

        return output_tokens

    def get_bos_next_tokens(self):
        return set(["0", "1", "<eos>"])
    

class Tomita7Language(FormalLanguage):
    def __init__(self, ):
        super().__init__()
        self.sigma = ['0', '1']
        self.Q = ['q0', 'q1', 'q2', 'q3', 'q4']
        self.q0 = 'q0'
        self.F = {'q0', 'q1', 'q2', 'q3'}
        self.dfa = DFA(self.sigma, self.Q, self.transition_function, self.q0, self.F)

    def transition_function(self, q, s):
        if q == 'q0':
            if s == '0':
                return 'q0'
            if s == '1':
                return 'q1'
        if q == 'q1':
            if s == '0':
                return 'q2'
            if s == '1':
                return 'q1'
        if q == 'q2':
            if s == '0':
                return 'q2'
            if s == '1':
                return 'q3'
        if q == 'q3':
            if s == '0':
                return 'q4'
            if s == '1':
                return 'q3'
        if q == 'q4':
            return 'q4'
        
        raise RuntimeError(f"{q}, {s} is invalid")

    def belongs_to_lang(self, seq):
        return self.dfa(seq)

    def generate_pos_sample(self, min_length, max_length):
        length = random.randint(min_length, max_length)
        lengths = []
        for i in range(3):
            lengths.append(random.randint(0, length - sum(lengths)))
        lengths.append(length - sum(lengths))

        random.shuffle(lengths)

        return "".join([c*l for l, c in zip(lengths, ["0", "1", "0", "1"])])
    
    def is_valid_length(self, length):
        return length >= 0
    
    def get_next_tokens(self, pos_sample):
        num_blocks = 1
        last_bit = "0"
        output_tokens = []
        for letter in pos_sample:
            if letter != last_bit:
                num_blocks += 1

            if num_blocks == 4:
                output_tokens.append(set(["1", "<eos>"]))
            elif num_blocks > 4:
                raise RuntimeError
            else:
                output_tokens.append(set(["0", "1", "<eos>"]))
            
            last_bit = letter

        return output_tokens

    def get_bos_next_tokens(self):
        return set(["0", "1", "<eos>"])


class Dn(FormalLanguage):
    def __init__(self, n):
        super().__init__()
        self.sigma = ["a", "b"]
        self.n = n
        assert n >= 1

    def belongs_to_lang(self, seq):
        depth = 0
        for letter in seq:
            if letter == "a":
                depth += 1
            elif letter == "b":
                depth -= 1
            else:
                raise RuntimeError("Unrecognizable letter", letter)
            
            if depth < 0 or depth > self.n: # TODO retrain if use recognition
                return False
        
        return depth == 0
            

    def generate_pos_sample(self, min_length, max_length):
        def get_next_action(current_depth, last_direction, max_down_step):
            if last_direction == "b":
                return ["a"] * random.randint(1, min(self.n-current_depth, max_down_step))
            elif last_direction == "a":
                return ["b"] * random.randint(1, current_depth)

        while not self.is_valid_length(length := random.randint(min_length, max_length)):
            pass
    
        depth = 0
        generated = []
        last_direction = "b"
        while len(generated) + depth < length:
            max_down_step = (length - len(generated) - depth) // 2
            new_s = get_next_action(depth, last_direction, max_down_step)
            generated.extend(new_s)
            last_direction = new_s[0]
            depth += (len(new_s) if new_s[0] == "a" else -len(new_s))
        
        generated.extend(["b"] * depth)
        assert len(generated) == length
        return "".join(generated)

    def is_valid_length(self, length):
        return (length >= 0) and (length % 2 == 0)
    
    def get_next_tokens(self, pos_sample):
        depth = 0
        output_tokens = []
        for letter in pos_sample:
            if letter == "a":
                depth += 1
            elif letter == "b":
                depth -= 1
            else:
                raise RuntimeError("Unrecognizable letter", letter)
        
            if depth == 0:
                output_tokens.append(set(["a", "<eos>"]))
            elif depth == self.n:
                output_tokens.append(set(["b"]))
            else:
                output_tokens.append(set(["a", "b"]))

        return output_tokens

    def get_bos_next_tokens(self):
        return set(["a", "<eos>"])


class AAStarLanguage(FormalLanguage):

    def __init__(self) -> None:
        super().__init__()
        self.sigma = ["a"]

    def belongs_to_lang(self, seq):
        if set(seq) != set(self.sigma) or len(seq) % 2 == 1:
            return False        
        else:
            return True

    def generate_pos_sample(self, min_length: int, max_length: int) -> str:
        while (length := random.randint(min_length, max_length)) % 2 == 1:
            pass
        return self.sigma[0] * length
    
    def is_valid_length(self, length):
        return (length >= 0) and (length % 2 == 0)
    
    def get_next_tokens(self, pos_sample):
        output_tokens = []
        for i, letter in enumerate(pos_sample):
            if i % 2 == 0:
                output_tokens.append(set(["a"]))
            else:
                output_tokens.append(set(["a", "<eos>"]))

        return output_tokens

    def get_bos_next_tokens(self):
        return set(["a", "<eos>"])
    

class AAAAStarLanguage(FormalLanguage):

    def __init__(self) -> None:
        super().__init__()
        self.sigma = ["a"]

    def belongs_to_lang(self, seq):
        if set(seq) != set(self.sigma) or len(seq) % 4 != 0:
            return False        
        else:
            return True

    def generate_pos_sample(self, min_length: int, max_length: int) -> str:
        while (length := random.randint(min_length, max_length)) % 4 != 0:
            pass
        return self.sigma[0] * length
    
    def is_valid_length(self, length):
        return (length >= 0) and (length % 4 == 0)
    
    def get_next_tokens(self, pos_sample):
        output_tokens = []
        for i, letter in enumerate(pos_sample):
            if (i+1) % 4 == 0:
                output_tokens.append(set(["a", "<eos>"]))
            else:
                output_tokens.append(set(["a"]))

        return output_tokens

    def get_bos_next_tokens(self):
        return set(["a", "<eos>"])
    

class ABABStarLanguage(FormalLanguage):

    def __init__(self) -> None:
        super().__init__()
        self.sigma = ["a", "b"]

    def belongs_to_lang(self, seq):
        if len(seq) % 4 != 0:
            return False
        if all(seq[i:i+4] == "abab" for i in range(0, len(seq), 4)):
            return True
        return False

    def generate_pos_sample(self, min_length: int, max_length: int) -> str:
        while (length := random.randint(min_length, max_length)) % 4 != 0:
            pass
        return "abab" * (length // 4)
    
    def is_valid_length(self, length):
        return (length >= 0) and (length % 4 == 0)
    
    def get_next_tokens(self, pos_sample):
        output_tokens = []
        for i, letter in enumerate(pos_sample):
            if (i+1) % 4 == 0:
                output_tokens.append(set(["a", "<eos>"]))
            else:
                if letter == "a":
                    output_tokens.append(set(["b"]))
                elif letter == "b":
                    output_tokens.append(set(["a"]))
                else:
                    raise RuntimeError("unrecognized letter", letter)

        return output_tokens

    def get_bos_next_tokens(self):
        return set(["a", "<eos>"])
    

class ABCDELanguage(FormalLanguage):
    def __init__(self) -> None:
        super().__init__()
        self.sigma = ["a", "b", "c", "d", "e"]

    def belongs_to_lang(self, seq):
        if match := re.search(r"a+b+c+d+e+", seq):  # greedy match
            if len(match.group()) == len(seq):
                return True
        return False
    
    def generate_pos_sample(self, min_length: int, max_length: int) -> str:
        # different from prev implementation
        length = random.randint(min_length, max_length)
        lengths = []
        for i in range(4):
            lengths.append(random.randint(0, length-5-sum(lengths)))
        lengths.append(length-5-sum(lengths))
        random.shuffle(lengths)
        s = ""
        for l, c in zip(lengths, self.sigma):
            s += c * (l+1)
        return s
    
    def is_valid_length(self, length):
        return (length >= 5)
    
    def get_next_tokens(self, pos_sample):
        output_tokens = []
        for letter in pos_sample:
            if letter == "a":
                output_tokens.append(set(["a", "b"]))
            elif letter == "b":
                output_tokens.append(set(["b", "c"]))
            elif letter == "c":
                output_tokens.append(set(["c", "d"]))
            elif letter == "d":
                output_tokens.append(set(["d", "e"]))
            elif letter == "e":
                output_tokens.append(set(["e", "<eos>"]))
            else:
                raise RuntimeError("unrecognized letter", letter)

        return output_tokens

    def get_bos_next_tokens(self):
        return set(["a"])
    

class AB_D_BCLanguage(FormalLanguage):  
    def __init__(self) -> None: 
        super().__init__()
        self.sigma = ["a", "b", "c", "d"]

    def belongs_to_lang(self, seq):
        if match := re.search(r"[ab]*d[bc]*", seq):  # greedy match
            if len(match.group()) == len(seq):
                return True
        return False
    
    def generate_pos_sample(self, min_length: int, max_length: int) -> str:
        # different from prev implementation
        length = random.randint(min_length, max_length)
        length0 = random.randint(0, length-1)
        length1 = length - 1 - length0

        # part0
        num_a = random.randint(0, length0)
        num_b = length0 - num_a
        part0 = list("a" * num_a + "b" * num_b)
        random.shuffle(part0)
        part0 = "".join(part0)

        # part1
        num_b = random.randint(0, length1)
        num_c = length1 - num_b
        part1 = list("b" * num_b + "c" * num_c)
        random.shuffle(part1)
        part1 = "".join(part1)
        
        return part0 + "d" + part1
    
    def is_valid_length(self, length):
        return (length >= 1)
    
    def get_next_tokens(self, pos_sample):
        d_occurred = False
        output_tokens = []
        for letter in pos_sample:
            if letter == "d":
                d_occurred = True
        
            if not d_occurred:
                output_tokens.append(set(["a", "b", "d"]))
            else:
                output_tokens.append(set(["b", "c", "<eos>"]))

        return output_tokens

    def get_bos_next_tokens(self):
        return set(["a", "b", "d"])

class ZOT_Z_TLanguage(FormalLanguage):     # {012}*02*
    def __init__(self) -> None: 
        super().__init__()
        self.sigma = ["0", "1", "2"]

    def belongs_to_lang(self, seq):
        if match := re.search(r"[012]*02*", seq):  # greedy match
            if len(match.group()) == len(seq):
                return True
        return False
    
    def generate_pos_sample(self, min_length: int, max_length: int) -> str:
        # different from prev implementation
        length = random.randint(min_length, max_length)
        length0 = random.randint(0, length-1)
        length1 = length - 1 - length0

        part0 = "".join(random.choices(self.sigma, k=length0))
        part1 = "2"*length1
        
        return part0 + "0" + part1
    
    def is_valid_length(self, length):
        return (length >= 1)

    def get_next_tokens(self, pos_sample):
        can_end = False
        output_tokens = []
        for c in pos_sample:
            if c == "0":
                can_end = True
            elif c == "1":
                can_end = False
            
            if can_end:
                output_tokens.append(set(["0", "1", "2", "<eos>"]))    # means EOS
            else:
                output_tokens.append(set(["0", "1", "2"]))
        return output_tokens

    def get_bos_next_tokens(self):
        return set(["0", "1", "2"])

def generate_neg_sample(language: FormalLanguage, min_length, max_length): # min_length already >= 1
    perturb_p = 0.5
    match language:
        # case Tomita1Language() | Tomita2Language() | ABABStarLanguage():
        #     length = random.randint(max(min_length, 1), max_length)
        case Tomita3Language(): # not used
            length = random.randint(max(min_length, 2), max_length)
        case Tomita4Language():
            length = random.randint(max(min_length, 3), max_length)
        case Tomita7Language():
            length = random.randint(max(min_length, 5), max_length)
        case AAStarLanguage():
            length = random.choice([l for l in range(max(min_length, 1), max_length+1) if l % 2 == 1])
            perturb_p = 0
        case AAAAStarLanguage():
            length = random.choice([l for l in range(max(min_length, 1), max_length+1) if l % 4 != 0])
            perturb_p = 0
        case Dn() | Tomita2Language() | Tomita5Language(): 
            # not for min invalid length, but for less trivial neg examples
            while True:
                length = random.randint(min_length, max_length)
                if length % 2 == 0:
                    break
                elif random.random() < 0.2:
                    break
        case Tomita6Language():
            while True:
                length = random.randint(min_length, max_length)
                if length % 2 == 1:
                    break
                elif random.random() < 0.2:
                    break
        case ABABStarLanguage():
            while True:
                length = random.randint(min_length, max_length)
                if length % 4 == 0:
                    break
                elif random.random() < 0.2:
                    break
        case _:
            length = random.randint(min_length, max_length)

    if random.random() < perturb_p:
        # perturbation sampling
        while True:
            edit_options = {"insert": 1, "replace": 0, "delete": -1}
            for _ in range(1000):
                k = np.random.geometric(0.5, 1).item()
                perturbs = random.choices(list(edit_options.keys()), k=k)
                length_increment = sum([edit_options[p] for p in perturbs])
                pos_sample_len = length - length_increment
                if language.is_valid_length(pos_sample_len) and all(i >= 0 for i in accumulate([edit_options[p] for p in perturbs], initial=pos_sample_len)):
                    break
            else:
                print(f"target length={length}, k={k}")
                raise RuntimeError("hard to find perturbs")
            candidate = language.generate_pos_sample(min_length=pos_sample_len, max_length=pos_sample_len)
            candidate = list(candidate)
            for p in perturbs:
                match p:
                    case "insert":
                        candidate.insert(random.randint(0, len(candidate)), random.choice(language.sigma))
                    case "replace":
                        if len(candidate) == 0 or len(language.sigma) <= 1:
                            continue
                        idx = random.randint(0, len(candidate)-1)
                        sigma = language.sigma.copy()
                        sigma.remove(candidate[idx])
                        candidate[idx] = random.choice(sigma)
                    case "delete":
                        del candidate[random.randint(0, len(candidate)-1)]
            candidate = "".join(candidate)
            if not language.belongs_to_lang(candidate):
                return candidate
    else:
        # uniform sampling
        while True:
            if not language.belongs_to_lang(candidate := "".join(random.choices(language.sigma, k=length))):
                return candidate



class FormalLangDataset(IterableDataset):
    def __init__(self, tokenizer: customTokenizer, length_range: tuple[int, int], max_test_length: int, language: str):
        super().__init__()
        self.range_min, self.range_max = length_range
        self.range_min = max(1, self.range_min)
        self.max_test_length = max_test_length
        assert (max_test_length >= self.range_max) or (max_test_length == -1)    # the pos emb is initialized based on max_test_length
        if match := re.search(r"[Tt]omita(\d)", language):
            self.language: FormalLanguage = globals()[f"Tomita{match.group(1)}Language"]()
        elif re.search(r"aaaastar|AAAAstar", language):
            self.language = AAAAStarLanguage()
        elif re.search(r"aastar|AAstar", language):
            self.language = AAStarLanguage()
        elif re.search(r"ababstar|ABABstar", language):
            self.language = ABABStarLanguage()
        elif re.search(r"abcde|ABCDE", language):
            self.language = ABCDELanguage()
            self.range_min = max(5, self.range_min)
        elif re.search(r"ab_d_bc|AB_D_BC", language):
            self.language = AB_D_BCLanguage()
        elif re.search(r"012_0_2|zot_z_t|ZOT_Z_T", language):
            self.language = ZOT_Z_TLanguage()
        elif match := re.search(r"[Dd]_?(\d+)", language):
            self.language = Dn(int(match.group(1)))
        else:
            raise NotImplementedError(f"{language} not recognized")
        
        self.BCE = language.startswith("bce") or language.startswith("BCE")

        if tokenizer is None:
            if self.BCE:
                tokenizer = customTokenizer(sorted(self.language.sigma))
            else:
                tokenizer = customTokenizer(sorted(list(set(["0", "1"] + self.language.sigma))))
        self.tokenizer = tokenizer

    def __iter__(self):
        while True:
            if self.BCE:
                instance, pos_ids, label = self.get_modeling_instance()
            else:
                instance, pos_ids, label = self.get_recognition_instance()
            yield instance, pos_ids, label

    def get_recognition_instance(self):
        if random.random() < 0.5:
            s = self.language.generate_pos_sample(self.range_min, self.range_max)
            label = "1"
        else:
            if callable(getattr(self.language, "generate_neg_sample", None)):
                s = self.language.generate_neg_sample(self.range_min, self.range_max)
            else:
                s = generate_neg_sample(self.language, self.range_min, self.range_max)
            label = "0"

        instance = [self.tokenizer.bos_token_id]
        instance.extend(list(map(lambda x: self.tokenizer.vocab[x], s)))
        instance.append(self.tokenizer.sep_token_id)
        instance.append(self.tokenizer.vocab[label])
        # instance.append(self.tokenizer.eos_token_id)

        label = [self.tokenizer.pad_token_id,] * len(instance)
        # setting some tokens to [pad] will make the loss on these tokens (as pred targets) be ignored
        # label[-2:] = instance[-2:]
        label[-1] = instance[-1]
        
        if self.max_test_length != -1:
            offset = random.randint(0, self.max_test_length - len(s))
        else:
            offset = 0
        pos_ids = list(range(offset, len(instance)+offset))

        return instance, pos_ids, label

    def get_modeling_instance(self):
        s = self.language.generate_pos_sample(self.range_min, self.range_max)
        labels = [self.language.get_bos_next_tokens()]
        labels.extend(self.language.get_next_tokens(s))
        labels.append(set())

        instance = [self.tokenizer.bos_token_id]
        instance.extend(list(map(lambda x: self.tokenizer.vocab[x], s)))
        instance.append(self.tokenizer.eos_token_id)

        label_ids = []
        for label in labels:
            label_ids.append([1 if self.tokenizer.vocab_inv[i] in label else 0 for i in range(len(self.tokenizer))])
        
        if self.max_test_length != -1:
            offset = random.randint(0, self.max_test_length - len(s))
        else:
            offset = 0
        pos_ids = list(range(offset, len(instance)+offset))

        return instance, pos_ids, label_ids


if __name__ == "__main__":
    d = ZOT_Z_TLanguage()
    print(d.generate_pos_sample(5, 10))
    print(d.generate_pos_sample(10, 20))
    print()
    print(generate_neg_sample(d, 5, 10))
    print(generate_neg_sample(d, 10, 20))

    # tokenizer, dataset = get_tokenizer_and_dataset_for_task("012_0_2", (0, 50), 150, None)
    # dataset = iter(dataset)
    # print(next(dataset))
    # print(next(dataset))
    # print(next(dataset))
    # print(next(dataset))
    # print(next(dataset))
    # print(next(dataset))