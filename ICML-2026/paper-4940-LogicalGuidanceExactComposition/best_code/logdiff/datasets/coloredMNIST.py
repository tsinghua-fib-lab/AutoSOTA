# -*- coding: utf-8 -*-
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision import datasets
import numpy as np
import random

normalized_rgbs = [
    (0.18, 0.31, 0.31),  # Dark Slate Gray
    (0.13, 0.55, 0.13),  # Forest Green
    (1.0, 0.27, 0.0),    # Orange Red
    (1.0, 1.0, 0.0),     # Yellow
    (0.78, 0.08, 0.52),  # Medium Violet Red
    (0.0, 1.0, 0.0),     # Lime
    (0.25, 0.41, 0.88),  # Royal Blue
    (0.0, 1.0, 1.0),     # Aqua
    (0.0, 0.0, 1.0),     # Blue
    (1.0, 0.87, 0.68)    # Navajo White
]
COLOURS =  torch.tensor(normalized_rgbs)

class Uniform:
    def __init__(self,train=True):
        self.train = train
    def make_color_from_label(self, y):
        return random.randint(0,9)
    def impart_color_to_mnist(self, mnist):
        colors = np.vectorize(self.make_color_from_label)(mnist.targets.numpy())
        return mnist, colors

class Partial:
    def __init__(self,train=True):
        self.train = train
    def make_color_from_label(self, y):
        if self.train:
            return random.randint(y,min(y+1,9))
        else:
            return random.choice([i for i in range(10) if i not in [y,min(y+1,9)]])
    def impart_color_to_mnist(self, mnist):
        colors = np.vectorize(self.make_color_from_label)(mnist.targets.numpy())
        return mnist, colors

class NonUniform:
    def __init__(self,train=True):
        self.train = train
    def make_color_from_label(self, y):
        if self.train:
            if random.uniform(0,1) < 0.5:
                return random.randint(y,min(y+1,9))
            else:
                return random.randint(0,9)
        else:
            return random.randint(0,9) 
    def impart_color_to_mnist(self, mnist):
        colors = np.vectorize(self.make_color_from_label)(mnist.targets.numpy())
        return mnist, colors

class Gaussian:
    def __init__(self,train=True):
        self.train = train
        if train:
            categorical_probs = [0.0010, 0.0076, 0.0360, 0.1095, 0.2132, 0.2663, 0.2132, 0.1095, 0.0360, 0.0076]
        else:
            categorical_probs = [0.1]*10
        categorical_probs = np.array(categorical_probs)
        self.categorical_probs /= categorical_probs.sum()

    def impart_color_to_mnist(self, mnist):
        colors = np.random.choice(len(self.categorical_probs), size=len(mnist), p=self.categorical_probs)
        count_samples = self.categorical_probs*len(mnist)
        count_samples = count_samples.astype(int)
        new_target_idx = []
        for i in range(10):
            idx = np.where(mnist.targets == i)[0]
            idx = np.random.choice(idx,count_samples[i],replace=True)
            new_target_idx.extend(idx.tolist())
            
        mnist.targets = mnist.targets[new_target_idx]
        mnist.data = mnist.data[new_target_idx]
        return mnist, colors




class ColoredMNIST(Dataset):
    def __init__(self, root_dir,train=True, download=False,support='uniform'):
        """
        Different support options:
        - uniform
        - non-uniform
        - partial
        - gaussian
        """
        mnist = datasets.MNIST(root=root_dir, train=train, transform=transforms.ToTensor(), download=download)
        supportfunctions = {
            'uniform': Uniform,
            'non-uniform': NonUniform,
            'partial': Partial,
            'gaussian': Gaussian
        }
        self.support = supportfunctions[support](train)
        self.colors = COLOURS
        self.mnist,self.color_labels = self.support.impart_color_to_mnist(mnist)
        self.transform = None
        self.null_color = torch.tensor([10,10])
        
    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, idx):
        img, label = self.mnist[idx]
        colour_idx= self.color_labels[idx]
        colour = self.colors[colour_idx]
        coloured_img = self.colour_image(img, colour)
        if self.transform:
            coloured_img = self.transform(coloured_img)
        label = torch.tensor([label,colour_idx]).long()
        return {'X': coloured_img, 'label':label, 'label_null': self.null_color}
    
    def colour_image(self, img, colour):
        # color all the pixels that are not black
        img = img.squeeze(0)
        color_indices = img > 0
        img = img.repeat(3, 1, 1) # convert to 3 channel image
        img[:, color_indices] = colour[:, None]
        return img
