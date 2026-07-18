from logdiff.score.sampling_compositional import Atom

import torch
from torch.nn import functional as F


class AtomCMNIST(Atom):
    def __init__(self, condition):
        super().__init__(condition)
    
    def log_probability(self, classifier, x, t):
        logits = classifier(x, t)
        atom_logits =  logits[self.condition_idx]
        log_p_all = F.log_softmax(atom_logits, dim=1)
        log_p = log_p_all[:, self.condition[self.condition_idx]]
        return log_p
    
    def get_classifier_guidance(self, classifier, xt, t, neg_guiding=False):
        with torch.enable_grad():
            xt_req = xt.detach().requires_grad_(True)
            logits = classifier(xt_req, t)           # [B, C]
            logp = torch.log_softmax(logits[self.condition_idx], dim=-1)
            target = logp[:, self.condition[self.condition_idx]].sum()
            grad = torch.autograd.grad(target, xt_req)[0]
        return grad
    
    def get_neg_guiding_cond_prob(self, classifier, x, t):
        logits = classifier(x, t)
        digit_logits = logits[self.condition_idx]

        log_p_all = F.log_softmax(digit_logits, dim=1)

        # exclude current condition -> we want (next) highest prob and condition
        forbidden_idx = self.condition[self.condition_idx]
        log_p_all[:, forbidden_idx] = -float('inf')

        most_probable_atom = torch.argmax(log_p_all, dim=1)
        max_log_prob = torch.max(log_p_all, dim=1).values
        if self.condition_idx == 0: # Digit
            conds = [[c.item(), 10] for c in most_probable_atom]
        else: # Color
            conds = [[10, c.item()] for c in most_probable_atom]

        return conds, max_log_prob
    

class Color(AtomCMNIST):
    colors_name_number = {"darkblue": 0, "green": 1, "red": 2, "yellow": 3, "pink": 4, "neongreen": 5, 
          "purple": 6, "brightblue": 7, "blue": 8, "beige": 9}
    colors_number_name = {v: k for k, v in colors_name_number.items()}
    
    def __init__(self, color):
        self.condition_idx = 1
        self.condition = self.__get_color(color)
        self.value = self.condition[self.condition_idx ]
        super().__init__(self.condition)
    
    def __str__(self):
        return f"{Color.colors_number_name[self.condition[1].item()]}"

    @staticmethod
    def __get_color(x):
        if isinstance(x, str) and x in Color.colors_name_number.keys():
            x = Color.colors_name_number[x]
        return torch.tensor([10, x])


class Digit(AtomCMNIST):
    def __init__(self, digit):
        self.condition_idx = 0
        self.condition = torch.tensor([digit, 10])
        self.value = self.condition[self.condition_idx]
        super().__init__(self.condition)

    def __str__(self):
        return f"{self.condition[self.condition_idx]}"
    
    