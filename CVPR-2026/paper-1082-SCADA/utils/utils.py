import sys

import torch.utils
import torch.utils.data
sys.path.append('.')

import matplotlib.pyplot as plt

import numpy as np

import os

import pickle

from sklearn.linear_model import LogisticRegression

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam 

from torchvision import transforms
from torchvision.transforms import ToTensor, Normalize, CenterCrop
from torchvision.datasets import ImageFolder
from torch.utils.data import Subset, DataLoader, random_split

from tllib.vision.transforms import ResizeImage




def log(out: str):
    """
    Dumps output to log.txt file
    """


    print(out)
    with open('log.txt', 'a') as f:
        f.write(out + '\n')


def dump(obj, path):
    """
    Similar to pickle dump, instead takes path and creates dir if not exists
    """
    
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pickle.dump(obj, open(path, 'wb'))
    

def load(path):
    """
    Similar to pickle dump, instead takes path
    """


    obj = pickle.load(open(path, 'rb'))
    return obj


def num_correct(logits, labels):
    """
    Computes number of correctly predicted labels per batch
    """


    is_correct = (
        torch.argmax(logits, dim=1) == labels
    ).float()
    return is_correct.sum().item()


def validate(model, dl, config):
    """
    Computes accuracy of the model on the data in dl
    """


    model.eval()
    epoch_test_acc = 0
    with torch.no_grad():
        for (images, labels) in dl:
            images, labels = images.to(config['device']), labels.to(config['device'])
            logits = model(images)
            epoch_test_acc += num_correct(logits, labels)
        epoch_test_acc /= len(dl.dataset)

    return 100 * epoch_test_acc


class Sample(nn.Module):
    def __init__(self, *dim):
        super().__init__()
        self.sample = nn.Parameter(torch.randn(*dim), requires_grad=True)

    def forward(self):
        return self.sample
    

class AdversarialSample:
    """
    Sample that maximizes probability of belonging to adv_class
    """
    
    
    def __init__(self, adv_classes, config):
        
        self.num_classes = config['num_classes']

        self.adv_classes = adv_classes
        self.num_adv = config['m_samples']
        self.sample = Sample(self.num_adv, config['channels'], config['size'], config['size'])

        self.hist = [0]

    
    def learn_init(self, classifier):

        optim = Adam(self.sample.parameters(), lr=0.1)
        adv_labels = torch.zeros(self.num_adv, self.num_classes).cuda()
        adv_labels[:, self.adv_classes] = 1. / len(self.adv_classes)

        for epoch in range(5):
            classifier.eval()
            loss_val = 0
            for step in range(10):
                sample = self.sample().cuda()
                logits = classifier(sample)
                loss = F.cross_entropy(logits, adv_labels)

                optim.zero_grad()
                loss.backward()
                optim.step()

                loss_val += loss.item()

            loss_val /= 10

            print(f"Epoch {epoch+1} Adv Loss: {loss_val:.5f}")
        
        classifier.train()

        return loss_val
    
    def update(self, classifier):

        adv_labels = torch.zeros(self.num_adv, self.num_classes).cuda()
        adv_labels[:, self.adv_classes] = 1. / len(self.adv_classes)

        optim = Adam(self.sample.parameters(), lr=1e-3)

        classifier.eval()
        sample = self.sample().cuda()
        logits = classifier(sample)
        loss = F.cross_entropy(logits, adv_labels)

        loss_val = loss.item()

        optim.zero_grad()
        loss.backward()
        optim.step()

        classifier.train()

        return loss_val


def predict_forget_class(classifier, target_dl, config, exclude=[], N=5):
    

    mean_output = torch.zeros(config['num_classes'])
    num_samples = 0

    for (images, _) in tqdm(target_dl, 'Predicting Forget Classes'):
        classifier.eval()
        with torch.no_grad():
            images = images.to(config['device'])
            logits = classifier(images)
            outputs = F.softmax(logits, dim=1)
            mean_output += outputs.sum(dim=0).cpu()
            num_samples += outputs.size(0)

    mean_output /= mean_output.max()

    mean_output[exclude] = 1.0

    sorted_indices = mean_output.argsort()[:N]

    predicted_classes = [idx.item() for idx in sorted_indices]
    
    print(mean_output)
    
    return predicted_classes


def create_ood_data(config, size):

    torch.manual_seed(config['seed'])

    RESIZE = 256
    SIZE = 224
    MEANS = [0.485, 0.456, 0.406]
    STDS = [0.229, 0.224, 0.225]

    transform = transforms.Compose([
        ResizeImage(RESIZE),
        CenterCrop(SIZE),
        ToTensor(),
        Normalize(mean=MEANS, std=STDS)
    ])

    if config['dataset'] == 'OfficeHome':
        path = os.path.join(config['data_path'], 'DomainNet', 'sketch')        
        dataset = ImageFolder(root=path, transform=transform)
        ood_classes = ["camel", "giraffe", "rhinoceros", "whale", "anvil", "dragon", "cannon", "The_Eiffel_Tower", "umbrella", "streetlight"]
        
    elif config['dataset'] == 'DomainNet':
        path = os.path.join(config['data_path'], 'OfficeHome', 'Product')        
        dataset = ImageFolder(root=path, transform=transform)
        ood_classes = ["Calendar", "Curtains", "Exit_Sign", "File_Cabinet", "Mop", "Postit_Notes", "Push_Pin", "Refrigerator", "ToothBrush", "Trash_Can"]
    elif config['dataset'] == 'Office31':
        path = os.path.join(config['data_path'], 'OfficeHome', 'Clipart')
        dataset = ImageFolder(root=path, transform=transform)
        ood_classes = ["Alarm_Clock" ,"Bed" ,"Candles" ,"Drill" ,"Flowers" ,"Hammer" ,"Refrigerator" ,"ToothBrush" ,"Toys" ,"TV"]

    else:
        raise NotImplementedError()

    ood_labels = {dataset.class_to_idx[c] for c in ood_classes if c in dataset.class_to_idx}

    valid_indices = [i for i, (_, lbl) in enumerate(dataset.samples) if lbl in ood_labels]

    chosen_indices = torch.randperm(len(valid_indices))[:size].tolist()

    subset_ood = Subset(dataset, [valid_indices[i] for i in chosen_indices])

    return DataLoader(
        subset_ood,
        batch_size=config['batch'],
        shuffle=False,
        num_workers=config['workers']
    )


class ClassLevelMembershipInferenceAttack:

    def __init__(self):
        pass

    
    def collect_prob(self, dl, classifier):
        
        dl = torch.utils.data.DataLoader(
            dl.dataset, batch_size=1, shuffle=False
        )
        
        prob = []
        classifier.eval()
        with torch.no_grad():
            for images, labels in dl:
                images = images.cuda()
                logits = classifier(images)
                outputs = F.softmax(logits, dim=-1).cpu()
                prob.append(outputs)
        
        return torch.cat(prob)


    def entropy(self, p, dim=-1, keepdim=False):
        return -torch.where(p > 0, p * p.log(), p.new([0.0])).sum(dim=dim, keepdim=keepdim)


    def get_mia(self, classifier, retain_dl, forget_dl, config):

        self.ood_dl = create_ood_data(config, size=len(retain_dl.dataset))

        retain_prob = self.collect_prob(retain_dl, classifier)
        ood_prob = self.collect_prob(self.ood_dl, classifier)
        forget_prob = self.collect_prob(forget_dl, classifier)

        
        X_r = (
            torch.cat([self.entropy(retain_prob), self.entropy(ood_prob)])
            .cpu()
            .numpy()
            .reshape(-1, 1)
        )

        y_r = np.concatenate([np.ones(len(retain_prob)), np.zeros(len(ood_prob))])

        X_f = self.entropy(forget_prob).cpu().numpy().reshape(-1, 1)
        y_f = np.ones(len(forget_prob))

        model = LogisticRegression(class_weight='balanced', solver='lbfgs', multi_class='multinomial')

        model.fit(X_r, y_r)

        results = model.predict(X_f)

        return results.mean()

def create_pseudo_dl(classifier, dl, size=100):
    """
    Creates a pseudo dataloader with dl inputs and model predictions as outputs (hard labels)
    """

    classifier.eval()
    pseudo_inputs, pseudo_labels = [], []
    
    with torch.no_grad():
        for i, (inputs, _) in enumerate(dl):
            if i > size:
                break
            inputs = inputs.cuda()
            outputs = classifier(inputs)
            _, preds = torch.max(outputs, 1)
            pseudo_inputs.append(inputs.cpu())
            pseudo_labels.append(preds.cpu())
    
    pseudo_inputs = torch.cat(pseudo_inputs)
    pseudo_labels = torch.cat(pseudo_labels)
    
    pseudo_dataset = TensorDataset(pseudo_inputs, pseudo_labels)
    pseudo_dl = DataLoader(pseudo_dataset, batch_size=dl.batch_size, shuffle=True)
    
    return pseudo_dl


def create_noisy_dl(classifier, reg_dl, config, num_reg=float('inf'), forget_classes=None):
    """
    Creates a mixed dataloader consisting of noisy samples and regular samples
    noisy samples are generated by maximizing error (UNSIR)
    """
    
    
    EPOCHS = 5
    STEPS = 8
    LR = 0.1
    ALPHA = 0.002
    BATCH_SIZE = 32
    IDK_WHAT_THIS_IS = [1, 2, 3]
    
    if forget_classes is None:
        forget_classes = config['forget_classes']

    noise = Sample(BATCH_SIZE, config['channels'], config['size'], config['size']).cuda()
    optim = torch.optim.Adam(noise.parameters(), lr=LR)

    forget_labels = torch.zeros(BATCH_SIZE, config['num_classes']).cuda()
    forget_labels[:, forget_classes] = 1.0 / len(forget_classes)

    for epoch in range(EPOCHS):
        classifier.eval()

        loss_val = 0
        for i in range(STEPS):

            images = noise()

            logits = classifier(images)
            loss = -F.cross_entropy(logits, forget_labels) + ALPHA * torch.mean(torch.sum(torch.square(images), IDK_WHAT_THIS_IS))

            optim.zero_grad()
            loss.backward()
            optim.step()

            loss_val += loss.item()

        loss_val /= STEPS

        print(f'Noise Epoch {epoch+1} Loss: {loss_val}')
    
    reg_samples = []
    n = 0
    for (images, labels) in reg_dl:
        for i in range(labels.shape[0]):
            reg_samples.append([images[i].cpu(), torch.nn.functional.one_hot(labels[i], num_classes=config['num_classes']).cpu().float()])
            n += 1
            if n >= num_reg:
                break
        if n >= num_reg:
            break
    
    noisy_samples = []
    for i in range(noise.sample.shape[0]):
        noisy_samples.append([noise().detach().cpu()[i], torch.zeros_like(forget_labels[i]).cpu().float()])

    samples = noisy_samples + reg_samples
    
    print('Noisy:',len(noisy_samples), 'Reg:', len(reg_samples), 'Total:', len(samples))
    noisy_dl = DataLoader(samples, config['batch'], shuffle=True, num_workers=8, drop_last=True)

    return noisy_dl


def detect_forget_classes(classifier, dl, config):
    num_present = config['num_classes'] - len(config['forget_classes'])
    features_list = []

    with torch.no_grad():
        for x, _ in dl:
            x = x.cuda()
            _, features = classifier(x)
            features_list.append(features.cpu())

    features = torch.cat(features_list)
    oracle_features = []

    for class_num in range(config['num_classes']):
        oracle_features.append(gen_adv(classifier, class_num, config).detach().cpu())

    oracle_features = torch.stack(oracle_features)
    distances = torch.cdist(features, oracle_features, p=2)
    assigned_classes = torch.argmin(distances, dim=1)

    class_counts = torch.bincount(assigned_classes, minlength=config['num_classes'])
    print("Samples per class:", class_counts.tolist())

    forget_classes = set(range(config['num_classes'])) - set(torch.nonzero(class_counts).squeeze().tolist())
    print("Forget classes:", forget_classes)

    del features_list, features, oracle_features, x
    torch.cuda.empty_cache()

    return assigned_classes, forget_classes

def gen_adv(classifier, class_num, config):
    sample = torch.randn((2, config['channels'], config['size'], config['size']), requires_grad=True, device='cuda')
    optim = Adam([sample], lr=0.1)
    adv_labels = torch.tensor([2, class_num], device='cuda')

    for epoch in range(5):
        classifier.eval()
        loss_val = 0
        for step in range(10):
            logits = classifier(sample)
            loss = F.cross_entropy(logits, adv_labels)
            
            optim.zero_grad()
            loss.backward()
            optim.step()
            loss_val += loss.item()
        
        loss_val /= 10
        print(f"Epoch {epoch+1} Adv Loss: {loss_val:.5f}")
    
    classifier.train()
    _, oracle_feature = classifier(sample)
    return oracle_feature.squeeze()

def predict_forget_classes_advanced(classifier, target_dl, config, exclude=[], N=5):
    

    mean_output = torch.zeros(config['num_classes'])
    num_samples = 0

    for (images, _) in tqdm(target_dl, 'Predicting Forget Classes'):
        classifier.eval()
        with torch.no_grad():
            images = images.to(config['device'])
            logits = classifier(images)
            outputs = F.softmax(logits, dim=1)
            mean_output += outputs.sum(dim=0).cpu()
            num_samples += outputs.size(0)

    mean_output /= mean_output.max()

    mean_output[exclude] = 1.0

    sorted_indices = mean_output.argsort()[:N]

    predicted_classes = [idx.item() for idx in sorted_indices]
    
    print(mean_output)
    
    return predicted_classes
