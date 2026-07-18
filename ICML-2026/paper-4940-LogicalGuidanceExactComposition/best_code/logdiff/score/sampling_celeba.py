from logdiff.score.sampling_compositional import Atom
import torch
from torch.nn import functional as F

class CelebaAtom(Atom):
    def __init__(self, condition):
        super().__init__(condition)
        self.null_token = torch.tensor([2, 5]) # Male (yes/no), HairColor (0-4 or 5 for null)
    
    def log_probability(self, classifier, x, t):
        logits = classifier(x, t)
        attr_logits = logits[self.condition_idx]
        log_p_all = F.log_softmax(attr_logits, dim=1)
        log_p = log_p_all[:, self.condition[self.condition_idx]]
        return log_p
    
    def get_neg_guiding_cond_prob(self, classifier, x, t):
        logits = classifier(x, t)
        digit_logits = logits[self.condition_idx]

        log_p_all = F.log_softmax(digit_logits, dim=1)

        # exclude current condition -> we want (next) highest prob and condition
        forbidden_idx = self.condition[self.condition_idx]
        log_p_all[:, forbidden_idx] = -float('inf')

        most_probable_atom = torch.argmax(log_p_all, dim=1)
        max_log_prob = torch.max(log_p_all, dim=1).values
        
        B = most_probable_atom.shape[0]
        device = most_probable_atom.device
        device = device[0] if isinstance(device, tuple) else device

        # null / base condition 
        conds = (self.null_token.to(device=device, dtype=torch.long).unsqueeze(0).repeat(B, 1))
        conds[:, self.condition_idx] = most_probable_atom

        return conds, max_log_prob

class Gender(CelebaAtom): 
    # Male attr_idx 20
    genders_name_number = {"female": 0, "male": 1}
    genders_number_name = {v: k for k, v in genders_name_number.items()}
    
    def __init__(self, gender):
        self.condition = self.__get_gender(gender)
        self.condition_idx = 0
        self.value = self.condition[self.condition_idx]
        super().__init__(self.condition)
    
    def __str__(self):
        return f"{Gender.genders_number_name[self.condition[self.condition_idx].item()]}"

    @staticmethod
    def __get_gender(x):
        if isinstance(x, str) and x in Gender.genders_name_number.keys():
            x = Gender.genders_name_number[x]
        return torch.tensor([x, 5])


class Hair(CelebaAtom):  
    # Hair color attribute follows the dataset label order:
    # 0=Black, 1=Blond, 2=Brown, 3=Gray, 4=Bald.
    hair_name_number = {"black": 0, "blond": 1, "brown": 2, "gray": 3, "bald": 4}
    hair_number_name = {0: "black", 1: "blond", 2: "brown", 3: "gray", 4: "bald"}
    
    def __init__(self, hair):
        self.condition = self.__get_hair(hair)
        self.condition_idx = 1
        self.value = self.condition[self.condition_idx]
        super().__init__(self.condition)
    
    def __str__(self):
        return f"{Hair.hair_number_name[self.condition[self.condition_idx].item()]}"

    @staticmethod
    def __get_hair(x):
        if isinstance(x, str) and x in Hair.hair_name_number.keys():
            x = Hair.hair_name_number[x]
        return torch.tensor([2, x])
