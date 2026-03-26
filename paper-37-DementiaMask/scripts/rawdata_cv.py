# check model performance w/o distribution shift
import sys
sys.path.insert(0, '.')
from src.utils import load_data, EarlyStopping

from transformers import set_seed
import random
import os
import pickle
from tqdm import trange
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from scipy.special import softmax
from datasets import Dataset

import pandas as pd
from collections import defaultdict
import argparse
import logging
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, f1_score, average_precision_score, roc_auc_score, recall_score, precision_score


from transformers import (
    AutoModelForSequenceClassification, 
    AutoTokenizer,
    AutoConfig,
    set_seed,
    get_scheduler,
    )
import logging
from collections import defaultdict
import pickle


# configuration
seed = 2024
device='cuda:0'
max_length = 256
batch_size = 4
num_epochs = 20
hidden_dropout_prob = 0.2
#num_warmup_steps = 50
lr = {'wls':1e-7, 'pitts': 1e-5, 'ccc': 3e-5}
N=5



def compute_metrics(logits, labels):
    probs = softmax(logits, axis=1)
    prob, pred = probs[:,1], np.argmax(probs, axis = 1) # probability for predicting dementia and its label
    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    aps = average_precision_score(y_true=labels, y_score=prob)
    try:
        roc_score = roc_auc_score(y_true=labels, y_score=prob)
    except ValueError:
        roc_score = 0.
    f1 = f1_score(y_true=labels, y_pred=pred)
    eval_output = {"accuracy": accuracy, "roc": roc_score, 
                   "aps": aps, "f1": f1}
    return eval_output

def eval_model(model, testloader):
    """Evaluate model performance on group level or all instances"""
    model.eval()
    val_loss = 0
    all_labels = []
    all_logits = []
    with torch.no_grad():
        for batch in testloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            labels = batch['labels'].cpu().numpy()
            logits = outputs.logits.cpu().numpy()
            loss = outputs.loss.cpu().numpy()
            all_labels.extend(labels)
            all_logits.extend(logits)
            val_loss += loss

    avg_loss = val_loss / len(testloader)
    ## group with different metrics
    metrics = compute_metrics(all_logits, all_labels)
    metrics['eval_loss'] = avg_loss

    return metrics

def model_training(model, trainloader, testloader, epochs, learning_rate):
    # set optimizers and scheduler
    optimizer = AdamW(model.parameters(), lr= learning_rate)
    lr_scheduler = get_scheduler(name="linear", optimizer=optimizer, 
                                num_warmup_steps=50, 
                                num_training_steps=epochs * len(trainloader))
    

    early_stopping = EarlyStopping(patience=5, verbose=True, device=device, path=f'/home/sheng136/DeconDTN/code/checkpoints/rawdata.model')

    # iterate batches
    for e in trange(epochs):
        model.train()
        for batch in trainloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        # evaluation
        metrics = eval_model(model = model, testloader=testloader)
        logging.info(f"""Epoch {e+1} - Eval loss: {metrics['eval_loss']:.3f}, Accuracy: {metrics['accuracy']:.3f}, 
                     AUROC: {metrics['roc']:.3f},
                     APS: {metrics['aps']:.3f}, 
                     F1: {metrics['f1']:.3f}""")
        # Early stopping call find model with best aps
        early_stopping(metrics['aps'], model)
        if early_stopping.early_stop: # end training if early stopping is triggered
            break
    
    model = early_stopping.load_checkpoint()

    return model

def setseed(SEED):
    np.random.seed(SEED)
    set_seed(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)


def cross_validation(df, rand_seed, num_folds):
    kf = StratifiedGroupKFold(n_splits=num_folds, shuffle=True,random_state=rand_seed)
    groups = df['id'].values
    return kf.split(df[['gender','text']], df['label'], groups=groups)

def down_sample(df, ds_seed = 2024):
    # do not need to downsample based on id
    # gender: 0 - male, 1 - female
    df1_m = df[(df['label'] == 1) & (df['gender'] == 0)]
    df1_f = df[(df['label'] == 1) & (df['gender'] == 1)]
    df0_m = df[(df['label'] == 0) & (df['gender'] == 0)]
    df0_f = df[(df['label'] == 0) & (df['gender'] == 1)]
    if len(df1_m) > len(df1_f):
        df1_m = df1_m.sample(n = len(df1_f), random_state = ds_seed)
    else:
        df1_f = df1_f.sample(n = len(df1_m), random_state = ds_seed)
    
    if len(df0_m) > len(df0_f):
        df0_m = df0_m.sample(n = len(df0_f), random_state = ds_seed)
    else:
        df0_f = df0_f.sample(n = len(df0_m), random_state = ds_seed)
    
    df = pd.concat([df1_m, df1_f, df0_m, df0_f])
    return df


def main():

    parser = argparse.ArgumentParser(description='Train a binary classification model use bert-base')
    parser.add_argument('-d', '--data', default='pitts')
    parser.add_argument('-s', '--subsample', default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument('-m', '--model_name', default='bert-base-uncased')
    parser.add_argument('-f', '--folds', default=5, type=int)
    args = parser.parse_args()


    model_name = args.model_name
    log_file = f'output/logs/{args.data}_{model_name}_sample{int(args.subsample)}_cv.log'
    if os.path.exists(log_file):
        os.remove(log_file)

    logging.basicConfig(filename=log_file,
                    filemode='w',
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    level=logging.DEBUG)


    logging.info("Run cross validation with 5 folds")
    dat_male, dat_female = load_data(args.data)
    logging.info(f"Loading the data from {args.data.upper()} dataset")
    dat = pd.concat([dat_male, dat_female], ignore_index = True)

    if args.subsample:
        logging.info("Downsample data to make it balanced...")
        dat = down_sample(dat)
    
    results = []
    
    # define data preprocessing functions
    def tokenize_function(examples):
        return tokenizer(examples["text"], max_length = 256, padding="max_length", truncation=True)

    def process_data(df, batch_size, shuffle):
        """ Tokenizer text to torch format from pandas"""
        data = Dataset.from_pandas(df[['text','label']])
        tokenized_data = data.map(tokenize_function, batched=True).remove_columns(["text"]).rename_column("label", "labels")
        tokenized_data.set_format("torch")
        tokenized_dataloader = DataLoader(tokenized_data, batch_size=batch_size, shuffle=shuffle)
        return tokenized_dataloader


    for _ in trange(N):
        rand_seed = random.randint(0, 1e5)
        logging.info(f"repeat: {_}, seed: {rand_seed}")
        
        eval_metrics = {'overall': defaultdict(list),
                        'male': defaultdict(list),
                        'female': defaultdict(list)}
        i = 1
        for train_idx, test_idx in cross_validation(dat, rand_seed, num_folds = args.folds):
            
            logging.info(f'Fold {i}')
            i+=1

            # same initialization for model
            set_seed(seed)

            configuration = AutoConfig.from_pretrained(model_name, num_labels=2)
            configuration.hidden_dropout_prob = hidden_dropout_prob

            tokenizer = AutoTokenizer.from_pretrained(model_name, device_map = 'auto')
            model = AutoModelForSequenceClassification.from_pretrained(model_name, 
                                                                       config=configuration)
            model.to(device)

            # train_test split
            df_train, df_test = dat.iloc[train_idx].reset_index(), dat.iloc[test_idx].reset_index()

            logging.info(f"Number of training samples: {len(df_train)}, Number of test samples: {len(df_test)}")

            logging.info('train and test data distribution:')
            logging.info(f"train:\n{pd.crosstab(df_train['label'], df_train['gender'], margins = True)}")
            logging.info(f"test:\n{pd.crosstab(df_test['label'], df_test['gender'], margins = True)}")
        
            set_seed(rand_seed)
            # NOTE:checkpoint: NO SAMPLES IN BOTH TRAIN AND TEST
            assert ~df_train.id.isin(df_test.id).any(), 'data leakage'

            male_test = df_test[df_test['gender']== 0].reset_index()
            female_test = df_test[df_test['gender']== 1].reset_index()
            
            # make sure there are samples in subgroup
            if len(male_test) == 0 or len(female_test) == 0:
                logging.info("Skip this fold due to no samples in subgroup")
                continue

            processed_train = process_data(df_train, shuffle = True, batch_size = batch_size)
            processed_test = process_data(df_test, shuffle = False, batch_size = batch_size)
            processed_maletest = process_data(male_test, shuffle = False, batch_size = batch_size)
            processed_femaletest = process_data(female_test, shuffle = False, batch_size = batch_size)

            config = {'epochs': num_epochs, 'learning_rate': lr[args.data]}
            # train model
            model = model_training(model, processed_train, processed_test, **config)

            # eval model
            overall_results = eval_model(model = model, testloader=processed_test)
            male_results = eval_model(model = model, testloader=processed_maletest)
            female_results = eval_model(model = model, testloader=processed_femaletest)

            for key in overall_results:
                eval_metrics['overall'][key].append(overall_results[key])
                eval_metrics['male'][key].append(male_results[key])
                eval_metrics['female'][key].append(female_results[key])
            
        # save results for repeats
        results.append(eval_metrics)

        logging.info("Results from 5-fold cross validation:")
        logging.info(f"Average Overall Accuracy: {np.mean(eval_metrics['overall']['accuracy'])}, \nAverage AUROC: {np.mean(eval_metrics['overall']['roc'])}, \nAverage F1: {np.mean(eval_metrics['overall']['f1'])}, \nAverage AUPRC: {np.mean(eval_metrics['overall']['aps'])}")
        logging.info(f"Average Male Accuracy: {np.mean(eval_metrics['male']['accuracy'])}, \nAverage AUROC: {np.mean(eval_metrics['male']['roc'])}, \nAverage F1: {np.mean(eval_metrics['male']['f1'])}, \nAverage AUPRC: {np.mean(eval_metrics['male']['aps'])}")
        logging.info(f"Average Female Accuracy: {np.mean(eval_metrics['female']['accuracy'])}, \nAverage AUROC: {np.mean(eval_metrics['female']['roc'])}, \nAverage F1: {np.mean(eval_metrics['female']['f1'])}, \nAverage AUPRC: {np.mean(eval_metrics['female']['aps'])}")
    
    model_tag = os.path.basename(model_name)
    sample = 1 if args.subsample else 0
    with open(f'output/raw_cv/results_{args.data}_{model_tag}_sample{sample}.pkl', 'wb') as f:
        pickle.dump(results, f)

if __name__ == '__main__':
    main()

