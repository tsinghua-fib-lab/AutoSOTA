from typing import Type, Optional, List
from random import Random
import torch
from abc import ABC, abstractmethod 

ATTACK_LIST = []

class Attack(ABC):

    @abstractmethod
    def __init__(self, name):
        self.name = name

    @abstractmethod
    def warp(self, text, prompt=None):
        pass

    @staticmethod
    @abstractmethod
    def get_param_list():
        pass

    def name_with_params(self):
        return self.name

    def score(self, model, input_encodings, target_encodings):
        start=0

        with torch.no_grad():
            scores = []
            if "is_encoder_decoder" not in model.config.__dict__ or not model.config.is_encoder_decoder:
                full_encodings = torch.cat((input_encodings, target_encodings), dim=1).to(input_encodings.device)
                target_ids = full_encodings.clone()
                target_ids[:, :input_encodings.shape[1]] = -100
                output = model(full_encodings, labels=target_ids)
            else:
                output = model(input_encodings, labels=target_encodings)
            batch_size, tgt_len = target_encodings.shape
            logits = torch.nn.functional.softmax(output.logits, dim=2)
            logits[:,:,0] = 1.0
            logprobs = logits[torch.arange(batch_size).unsqueeze(1).expand(-1, tgt_len), \
                            torch.arange(tgt_len).unsqueeze(0).expand(batch_size, -1), \
                            target_encodings].sum(dim=1)
            counts = torch.count_nonzero(target_encodings, dim=1)
        return torch.div(logprobs,counts)


