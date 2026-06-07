import os

from tqdm import tqdm

import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
import torch.nn.functional as F

from tllib.utils.data import ForeverDataIterator

from utils.utils import AdversarialSample, predict_forget_class, validate, create_noisy_dl, create_pseudo_dl
from utils.sfda import SFDA2




def minimax(classifier, target_train_dl, target_test_dl, target_forget_dl, source_test_dl, config, save=False):
    """
    Adversarial Minimax for DASEC Unlearning    
    """


    sfda2 = SFDA2(classifier, config)
    indexed_target_train_dl = sfda2.label_dataset(target_train_dl)
    sfda2.create_banks(indexed_target_train_dl)

    indexed_target_train_dl = ForeverDataIterator(indexed_target_train_dl)
    optimizer = SGD(classifier.get_parameters(), lr=1e-2, momentum=0.9, weight_decay=1e-3, nesterov=True)
    lr_scheduler = LambdaLR(optimizer, lambda x: 1e-2 * (1. + 1e-3 * float(x)) ** -0.90)

    num_forget_classes = len(config['forget_classes'])
    iter_per_class = config['iter_per_epoch'] // num_forget_classes
    adversarial_samples = [AdversarialSample([adv_class], config) for adv_class in config['forget_classes']]
    
    sfda_loss_hist = []
    mu_loss_hist = []
    adv_loss_hist = []

    for epoch in range(config['epochs']):
        classifier.train()
        loss_val = 0
        mu_loss_val = 0
        
        for i in tqdm(range(config['iter_per_epoch']), desc=f"Epoch {epoch+1}"):
            
            class_idx = min(i // iter_per_class, num_forget_classes - 1)

            if i % iter_per_class == 0 and i // iter_per_class < num_forget_classes:
                print(f"Learning Adv Sample Class {adversarial_samples[class_idx].adv_classes}")
                forget_acc = validate(classifier, target_forget_dl, config)
                print(f"Forget Accuracy: {forget_acc:.2f}") 

                if config['m_init']:
                    adv_loss_val = adversarial_samples[class_idx].learn_init(classifier)
                    adv_loss_hist.append(adv_loss_val)

            else:

                if i % config['m_update'] == 0:
                    adv_loss_val = adversarial_samples[class_idx].update(classifier)
                    adv_loss_hist.append(adv_loss_val)


            classifier.train()
            images, indices = next(indexed_target_train_dl)
            images = images.cuda()

            sample = adversarial_samples[class_idx].sample().detach().cuda()
            logits = classifier(sample)[0]

            if config['m_label'] == 'rescaled':
                labels = logits.detach().clone().cuda()
                labels[:, adversarial_samples[class_idx].adv_classes] = -float('inf')
                labels = F.softmax(labels, dim=1)
            
            elif config['m_label'] == 'uniform':
                labels = torch.ones_like(logits).cuda()
                labels[:, adversarial_samples[class_idx].adv_classes] = 0
                labels /= labels.sum(dim=1, keepdim=True)
                labels = F.softmax(labels, dim=1)

            elif config['m_label'] == 'random':
                labels = torch.zeros_like(logits).cuda()
                retain_classes = list(set(range(logits.size(1))) - set(adversarial_samples[class_idx].adv_classes))
                random_indices = torch.randint(0, len(retain_classes), (config['m_samples'],), device=config['device'])
                random_classes = [retain_classes[i] for i in random_indices]
                for i in range(logits.size(0)):
                    labels[i, random_classes[i]] = 1.0

            else:
                raise NotImplementedError()
            
            mu_loss = F.cross_entropy(logits, labels)

            sfda_loss = sfda2.loss(images, indices)
            
            loss = sfda_loss + mu_loss * config['m_alpha']
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), max_norm=1.0)
            optimizer.step()
            lr_scheduler.step()
            
            loss_val += loss.item()
            mu_loss_val += mu_loss.item()

            sfda_loss_hist.append(sfda_loss.item())
            mu_loss_hist.append(mu_loss.item())


        if (not config['fast_train']) or (epoch == config['epochs'] - 1):
            
            if save:
                path = os.path.join(config['save_path'], config['dataset'], save, f"{config['source']}_{config['target']}.pt")
                os.makedirs(os.path.dirname(path), exist_ok=True)
                torch.save(classifier.state_dict(), path)
        
            loss_val /= config['iter_per_epoch']
            source_acc = validate(classifier, source_test_dl, config)
            target_acc = validate(classifier, target_test_dl, config)
            forget_acc = validate(classifier, target_forget_dl, config)

            print(f"----------\n"
                  f"Epoch                  : {epoch+1}\n"
                  f"Train Loss             : {loss_val:.2f}\n"
                  f"MU Loss                : {mu_loss_val:.2f}\n"
                  f"Source Accuracy        : {source_acc:.2f}\n" 
                  f"Target Accuracy        : {target_acc:.2f}\n" 
                  f"Forget Accuracy        : {forget_acc:.2f}\n" 
                  f"----------\n")
            

    return classifier


def uc_minimax(classifier, target_train_dl, target_test_dl, target_forget_dl, source_test_dl, config, save=False):
    """
    Unknown Class Minimax ()
    """


    sfda2 = SFDA2(classifier, config)
    indexed_target_train_dl = sfda2.label_dataset(target_train_dl)
    sfda2.create_banks(indexed_target_train_dl)

    indexed_target_train_dl = ForeverDataIterator(indexed_target_train_dl)
    optimizer = SGD(classifier.get_parameters(), lr=1e-2, momentum=0.9, weight_decay=1e-3, nesterov=True)
    lr_scheduler = LambdaLR(optimizer, lambda x: 1e-2 * (1. + 1e-3 * float(x)) ** -0.90)

    num_forget_classes = len(config['forget_classes'])
    iter_per_class = config['iter_per_epoch'] // num_forget_classes
    forget_classes = []
    adversarial_samples = []
    
    for epoch in range(config['epochs']):
        classifier.train()
        loss_val = 0
        mu_loss_val = 0
        
        for i in tqdm(range(config['iter_per_epoch']), desc=f"Epoch {epoch+1}"):
            
            class_idx = min(i // iter_per_class, num_forget_classes - 1)
                
            if i % iter_per_class == 0 and i // iter_per_class < num_forget_classes:
                
                if epoch == 0:
                    predicted_classes = predict_forget_class(classifier, target_train_dl, config, exclude=sum(forget_classes, []), N=3) # sum(x, []) will unpack x somehow
                    print(f"Predicted Classes {predicted_classes}")
                    adversarial_samples.append(AdversarialSample(predicted_classes, config))
                    forget_classes.append(predicted_classes)

                print(f"Learning Sample for Classes {adversarial_samples[class_idx].adv_classes}")
                forget_acc = validate(classifier, target_forget_dl, config)
                print(f"Forget Accuracy: {forget_acc:.2f}") 

                adversarial_samples[class_idx].learn_init(classifier)
            else:

                adversarial_samples[class_idx].update(classifier)

            classifier.train()
            images, indices = next(indexed_target_train_dl)
            images = images.cuda()

            sample = adversarial_samples[class_idx].sample().detach().cuda()
            logits = classifier(sample)[0]
            labels = logits.detach().clone().cuda()
            labels[:, adversarial_samples[class_idx].adv_classes] = -float('inf')
            labels = F.softmax(labels, dim=1)

            mu_loss = F.cross_entropy(logits, labels)

            loss = sfda2.loss(images, indices) + mu_loss * 10.0
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            
            loss_val += loss.item()
            mu_loss_val += mu_loss.item()

        if (not config['fast_train']) or (epoch == config['epochs'] - 1):
            
            if save:
                path = os.path.join(config['save_path'], config['dataset'], save, f"{config['source']}_{config['target']}.pt")
                os.makedirs(os.path.dirname(path), exist_ok=True)
                torch.save(classifier.state_dict(), path)

            loss_val /= config['iter_per_epoch']
            source_acc = validate(classifier, source_test_dl, config)
            target_acc = validate(classifier, target_test_dl, config)
            forget_acc = validate(classifier, target_forget_dl, config)

            print(f"----------\n"
                  f"Epoch                  : {epoch+1}\n"
                  f"Train Loss             : {loss_val:.2f}\n"
                  f"MU Loss                : {mu_loss_val:.2f}\n"
                  f"Source Accuracy        : {source_acc:.2f}\n" 
                  f"Target Accuracy        : {target_acc:.2f}\n" 
                  f"Forget Accuracy        : {forget_acc:.2f}\n" 
                  f"----------\n")

    return classifier



def unsir(classifier, target_train_dl, target_test_dl, target_forget_dl, source_test_dl, config):
    """
    UNSIR Unlearning Method
    """


    pseudo_target_dl = create_pseudo_dl(classifier, target_train_dl)
    noisy_dl = create_noisy_dl(classifier, pseudo_target_dl, config, num_reg=256)

    optimizer = SGD(classifier.get_parameters(),
                    lr=0.1, momentum=0.9, weight_decay=1e-3, nesterov=True)
    lr_scheduler = LambdaLR(optimizer, lambda x: 1e-1 * (1. + 1e-3 * float(x)) ** -0.75)
    
    for epoch in range(1):
        classifier.train()
        for i, (images, labels) in enumerate(noisy_dl):
            images, labels = images.cuda(), labels.cuda()
            logits = classifier(images)[0]
            
            loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            
            if i == 8:
                break

    optimizer = SGD(classifier.get_parameters(),
                    lr=0.1, momentum=0.9, weight_decay=1e-3, nesterov=True)
    lr_scheduler = LambdaLR(optimizer, lambda x: 1e-1 * (1. + 1e-3 * float(x)) ** -0.75)

    for epoch in range(1):
        classifier.train()
        for i, (images, labels) in enumerate(pseudo_target_dl):
            images, labels = images.cuda(), labels.cuda()
            logits = classifier(images)[0]
            
            loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

    source_acc = validate(classifier, source_test_dl, config)
    target_acc = validate(classifier, target_test_dl, config)
    forget_acc = validate(classifier, target_forget_dl, config)

    print(f"----------\n"
            f"Source Accuracy        : {source_acc:.2f}\n" 
            f"Target Accuracy        : {target_acc:.2f}\n" 
            f"Forget Accuracy        : {forget_acc:.2f}\n" 
            f"----------\n")

    return classifier
