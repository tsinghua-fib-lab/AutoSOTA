from logdiff.score.sampling_compositional import Atom
import torch
from torch.nn import functional as F

class Atom3DShape(Atom):
    def __init__(self, condition):
        super().__init__(condition)
        self.null_token = torch.tensor([10, 10, 10,  8,  4, 15])
    
    def log_probability(self, classifier, x, t):
        logits = classifier(x, t)
        color_logits = logits[self.condition_idx]
        log_p_all = F.log_softmax(color_logits, dim=1)
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


class Hue(Atom3DShape):
    colors_numbers = {"red": 0, "orange": 1, "yellow": 2, "neongreen": 3, "lightgreen": 4, 
          "brightblue": 5, "blue": 6, "darkblue": 7, "purple": 8, "pink": 9}
    number_colors = {v: k for k, v in colors_numbers.items()}
    
    def __init__(self, color):
        super().__init__(color)
    
    def __str__(self):
        return f"{Hue.number_colors[self.condition[self.condition_idx].item()]}"
    
    def get_color(self, x):
        if isinstance(x, str) and x in Hue.colors_numbers.keys():
            x = Hue.colors_numbers[x]
        condition = torch.tensor([10, 10, 10,  8,  4, 15])
        condition[self.condition_idx] = x
        return condition


class FloorHue(Hue):
    
    def __init__(self, color):
        self.condition_idx = 0
        self.condition = self.get_color(color)
        self.value = self.condition[self.condition_idx]
        super().__init__(self.condition)

    def __str__(self):
        return f"{Hue.number_colors[self.condition[self.condition_idx].item()]} floor"


class WallHue(Hue):
  
    def __init__(self, color):
        self.condition_idx = 1
        self.condition = self.get_color(color)
        self.value = self.condition[self.condition_idx]
        super().__init__(self.condition)

    def __str__(self):
        return f"{Hue.number_colors[self.condition[self.condition_idx].item()]} wall"


class ObjectHue(Hue):
  
    def __init__(self, color):
        self.condition_idx = 2
        self.condition = self.get_color(color)
        self.value = self.condition[self.condition_idx]
        super().__init__(self.condition)

    def __str__(self):
        return f"{Hue.number_colors[self.condition[self.condition_idx].item()]} object"


class Scale(Atom3DShape):
    
    def __init__(self, scale):
        self.condition_idx = 3
        self.condition = torch.tensor([10, 10, 10, scale, 4, 15])
        self.value = self.condition[self.condition_idx]
        super().__init__(self.condition)
    
    def __str__(self):
        return f"{self.condition[self.condition_idx].item()} scale"
    

class Shape(Atom3DShape):
    shapes_numbers = {"cube": 0, "cylinder": 1, "sphere": 2, "capsule": 3}
    numbers_shapes = {v: k for k, v in shapes_numbers.items()}

    def __init__(self, shape):
        self.condition_idx = 4
        self.condition = self.__get_shape(shape)
        self.value = self.condition[self.condition_idx]
        super().__init__(self.condition)
    
    def __str__(self):
        return f"{Shape.numbers_shapes[self.condition[self.condition_idx].item()]}"
        
    def __get_shape(self, x):
        if isinstance(x, str) and x in Shape.shapes_numbers.keys():
            x = self.shapes_numbers[x]
        condition = torch.tensor([10, 10, 10,  8,  4, 15])
        condition[self.condition_idx] = x
        return condition


class Orientation(Atom3DShape):
    # orientation: 15 values linearly spaced in [-30, 30]
    def __init__(self, azimuth):
        self.condition_idx = 5
        self.condition = torch.tensor([10, 10, 10, 8, 4, azimuth])
        self.value = self.condition[self.condition_idx]
        super().__init__(self.condition)
    
    def __str__(self):
        return f"{self.condition[self.condition_idx].item() * 4 - 30}°"
    