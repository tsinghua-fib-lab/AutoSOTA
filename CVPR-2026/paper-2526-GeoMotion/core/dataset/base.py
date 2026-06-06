from torch.utils.data import Dataset
import random

class ProbabilisticDataset(Dataset):
    def __init__(self, dataset1, dataset2, dataset3, dataset4, prob1=0.25, prob2=0.25, prob3=0.25, prob4=0.25):
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.dataset3 = dataset3
        self.dataset4 = dataset4
        self.prob1 = prob1
        self.prob2 = prob2
        self.prob3 = prob3
        self.prob4 = prob4

        total_prob = self.prob1 + self.prob2 + self.prob3 + self.prob4
        if total_prob != 1:
            raise ValueError("The probabilities must sum to 1.")

    def __len__(self):
        return max(len(self.dataset1), len(self.dataset2), len(self.dataset3), len(self.dataset4))

    def __getitem__(self, idx):
        rand_value = random.random()
        if rand_value < self.prob1:
            sample = self.dataset1[idx % len(self.dataset1)]
        elif rand_value < self.prob1 + self.prob2:
            sample = self.dataset2[idx % len(self.dataset2)]
        elif rand_value < self.prob1 + self.prob2 + self.prob3:
            sample = self.dataset3[idx % len(self.dataset3)]
        else:
            sample = self.dataset4[idx % len(self.dataset4)]
        return sample
