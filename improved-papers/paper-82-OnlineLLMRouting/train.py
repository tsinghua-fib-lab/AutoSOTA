import numpy as np
import argparse, random
import torch, pickle
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset, random_split
from datasets import Dataset
from datasets import load_dataset, load_from_disk, concatenate_datasets
from sklearn.model_selection import cross_val_score, train_test_split
from tqdm.auto import tqdm
import transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, EvalPrediction, EarlyStoppingCallback, get_scheduler
from sentence_transformers import SentenceTransformer

RANDOM_STATE = 42
TEST_SIZE = .2
VAL_SIZE = .1
MAX_LENGTH = 512
MODEL_NAME = "roberta-base"

# 11 
models = [
    'WizardLM/WizardLM-13B-V1.2', 'claude-instant-v1', 'claude-v1', 'claude-v2', 'gpt-3.5-turbo-1106', 'gpt-4-1106-preview', 'meta/code-llama-instruct-34b-chat', 'meta/llama-2-70b-chat', 'mistralai/mistral-7b-chat', 'mistralai/mixtral-8x7b-chat', 'zero-one-ai/Yi-34B-Chat'
]

# 13
models_4_sprout = [
    "aws-claude-3-5-sonnet-v1", "aws-titan-text-premier-v1", "openai-gpt-4o", "openai-gpt-4o-mini", "wxai-granite-3-2b-instruct-8k-max-tokens",
    "wxai-granite-3-8b-instruct-8k-max-tokens", "wxai-llama-3-1-70b-instruct", "wxai-llama-3-1-8b-instruct", "wxai-llama-3-2-1b-instruct", 
    "wxai-llama-3-2-3b-instruct", "wxai-llama-3-3-70b-instruct", "wxai-llama-3-405b-instruct", "wxai-mixtral-8x7b-instruct-v01"
]

# 18
models_4_leaderboard = ['01-ai/Yi-34B-Chat', 'NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO', 'Qwen/QwQ-32B-Preview', 'Qwen/Qwen2-72B-Instruct', 'Qwen/Qwen2.5-72B-Instruct', 
      'Qwen/Qwen2.5-7B-Instruct', 'alpindale/WizardLM-2-8x22B', 'deepseek-ai/deepseek-llm-67b-chat', 'google/gemma-2-27b-it', 'google/gemma-2-9b-it', 
      'google/gemma-2b-it', 'meta-llama/Llama-2-13b-chat-hf', 'meta-llama/Meta-Llama-3.1-70B-Instruct', 'mistralai/Mistral-7B-Instruct-v0.1', 
      'mistralai/Mistral-7B-Instruct-v0.2', 'mistralai/Mistral-7B-Instruct-v0.3', 'mistralai/Mixtral-8x7B-Instruct-v0.1', 'nvidia/Llama-3.1-Nemotron-70B-Instruct-HF']

input_costs = [3.0, 0.5, 2.5, 0.15, 0.1, 0.2, 0.9, 0.2, 0.06, 0.06, 0.9, 3.5,0.6]
output_costs = [15.0, 1.5, 10.0, 0.6, 0.1, 0.2, 0.9, 0.2, 0.06, 0.06, 0.9, 3.5, 0.6] 
input_costs = [x / 1000000 for x in input_costs]
output_costs = [x / 1000000 for x in output_costs]


def prepare_datasets_prediction(test_texts,
                                tokenizer,
                                max_length=MAX_LENGTH):
    test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=max_length)
    test_dataset = Dataset.from_dict({"input_ids": test_encodings["input_ids"]})
    return test_dataset
    
def prepare_datasets(train_texts,
                     val_texts,
                     train_labels,
                     val_labels,
                     tokenizer,
                     max_length=MAX_LENGTH):
    
    train_texts = list(train_texts)
    val_texts = list(val_texts)
    

    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=max_length)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=max_length)

    train_dataset = Dataset.from_dict({"input_ids": train_encodings["input_ids"], "labels": train_labels})
    val_dataset = Dataset.from_dict({"input_ids": val_encodings["input_ids"], "labels": val_labels})

    pad_token = tokenizer.pad_token_id
    return train_dataset, val_dataset

def compute_metrics_for_regression(eval_pred):
    logits, labels = eval_pred
    mse = np.mean(np.square(labels-logits))
    return {"mse": mse}

def compute_metrics_for_classification(eval_pred, threshold=.5):
    sigmoid = torch.nn.Sigmoid()
    logits, labels = eval_pred
    probs = sigmoid(torch.Tensor(logits))
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1
    y_true = np.zeros(labels.shape)
    y_true[np.where(labels >= threshold)] = 1
    return {'all_accuracy': (y_true == y_pred).mean()}

def get_compute_metrics(task):
    if task=='classification':
        return compute_metrics_for_classification
    else:
        return compute_metrics_for_regression

def train_roberta(X, #list of texts
                  Y, #response tensor
                  task, #'classification' vs 'regression'
                  output_dir = "./models/roberta/checkpoints",
                  model_name = MODEL_NAME,
                  learning_rate = 2e-5,
                  weight_decay = 0.01,
                  batch_size = 16,
                  gradient_accumulation_steps = 1,
                  early_stopping_patience = 50,
                  val_size = VAL_SIZE,
                  max_length = MAX_LENGTH,
                  warmup_steps = 1000,
                  eval_steps = 200,
                  max_steps = 500000,
                  random_state = RANDOM_STATE,
                  device = 'cuda'):

    assert task in ['classification', 'regression']


    # Loading model and tokenizer
    if Y.squeeze().shape==Y.shape: # assuming labels are at most bi-dimensional
        num_labels = Y.shape[1]
    else:
        num_labels = 1
 
    if task=='classification':
        model = AutoModelForSequenceClassification.from_pretrained(model_name, problem_type="multi_label_classification", num_labels=num_labels)
        model.to(device)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(model_name, problem_type="regression", num_labels=num_labels)
        model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Prepare datasets
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        X, Y, test_size=val_size, random_state=random_state
    )

    train_dataset, val_dataset = prepare_datasets(
        train_texts, val_texts, train_labels, val_labels, tokenizer, max_length
    )


    # Generate training_args
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=eval_steps,
        save_total_limit=1,
        metric_for_best_model="eval_loss",  
        greater_is_better=False,
        load_best_model_at_end=True,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        max_steps=max_steps,
        num_train_epochs=9999,
        weight_decay=weight_decay,
        learning_rate=learning_rate,
        logging_dir="./logs",
        logging_steps=eval_steps,
        fp16=True if device.startswith("cuda") else False,
        report_to = "wandb",
        gradient_accumulation_steps=gradient_accumulation_steps,  
        lr_scheduler_type="linear",
        warmup_steps=warmup_steps, 
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=get_compute_metrics(task), 
        callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)])
    

    trainer.train()
    return trainer


def sigmoid(z):
    return 1/(1+np.exp(-z))


if __name__=="__main__":
    ### Inputs
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--dataset", type=str)
    args = parser.parse_args()
    dataset = args.dataset

    assert dataset in ['routerbench','sprout', 'leaderboard']

    ### Load data
    if dataset == "routerbench":
        with open("data/routerbench_0shot.pkl", "rb") as f:
            routerbench = pickle.load(f)
        routerbench = Dataset.from_pandas(routerbench)

        random.seed(42)
        all_indices = list(range(len(routerbench)))
        sample_indices = random.sample(all_indices, k=10000)
        rest_indices = list(set(all_indices) - set(sample_indices))
        sampled_dataset = routerbench.select(sample_indices)
        rest_dataset = routerbench.select(rest_indices)

        data = {
            'Q_train': rest_dataset['prompt'],
            'Q_test': sampled_dataset['prompt'],
            'Y_train':np.array([[d for d in rest_dataset[m]] for m in models]).T,
            'Y_test':np.array([[d for d in sampled_dataset[m]] for m in models]).T,
            'OT_train':np.array([[d for d in rest_dataset[f"{m}|total_cost"]] for m in models]).T,
            'OT_test':np.array([[d for d in sampled_dataset[f"{m}|total_cost"]] for m in models]).T,
            'models':models
        }   

    elif dataset == "sprout":
        sprout = load_dataset("CARROT-LLM-Routing/SPROUT")
        data = {
            'Q_train': sprout['train']['prompt'],
            'Q_test': sprout['validation']['prompt'] + sprout['test']['prompt'],
            'Y_train':np.array([[d['score'] for d in sprout['train'][m]] for m in models_4_sprout]).T,
            'Y_test':np.vstack((np.array([[d['score'] for d in sprout['validation'][m]] for m in models_4_sprout]).T, np.array([[d['score'] for d in sprout['test'][m]] for m in models_4_sprout]).T)),
            'IT_train':np.array([[d['num_input_tokens'] for d in sprout['train'][m]] for m in models_4_sprout]).T,
            'IT_test':np.vstack((np.array([[d['num_input_tokens'] for d in sprout['validation'][m]] for m in models_4_sprout]).T,np.array([[d['num_input_tokens'] for d in sprout['test'][m]] for m in models_4_sprout]).T)),
            'OT_train':np.array([[(d['num_output_tokens'] * output_costs[ind] + d['num_input_tokens'] * input_costs[ind]) for d in sprout['train'][m]] for ind, m in enumerate(models_4_sprout)]).T,
            'OT_test':np.vstack((np.array([[(d['num_output_tokens'] * output_costs[ind] + d['num_input_tokens'] * input_costs[ind]) for d in sprout['validation'][m]] for ind, m in enumerate(models_4_sprout)]).T,np.array([[(d['num_output_tokens'] * output_costs[ind] + d['num_input_tokens'] * input_costs[ind]) for d in sprout['test'][m]] for ind, m in enumerate(models_4_sprout)]).T)),
            'models':models_4_sprout
            }

    elif dataset == "leaderboard":
        leaderboard = load_from_disk("leaderboard")

        random.seed(42)
        all_indices = list(range(len(leaderboard)))
        sample_indices = random.sample(all_indices, k=10000)
        rest_indices = list(set(all_indices) - set(sample_indices))
        sampled_dataset = leaderboard.select(sample_indices)
        rest_dataset = leaderboard.select(rest_indices)

        data = {
            'Q_train': rest_dataset['prompt'],
            'Q_test': sampled_dataset['prompt'],
            'Y_train':np.array([[d for d in rest_dataset[m]] for m in models_4_leaderboard]).T,
            'Y_test':np.array([[d for d in sampled_dataset[m]] for m in models_4_leaderboard]).T,
            'OT_train':np.array([[d for d in rest_dataset[f"{m}|total_cost"]] for m in models_4_leaderboard]).T,
            'OT_test':np.array([[d for d in sampled_dataset[f"{m}|total_cost"]] for m in models_4_leaderboard]).T,
            'models':models_4_leaderboard
        }  

    ### Train 
    for s in ['Y_train']:
        data[s] = np.array(data[s]).astype(float)
    if dataset == 'sprout':
        data['OT_train'] = np.array(data['OT_train']).astype(float)


    model_dir =  f"roberta_{dataset}" / f"perf"
    roberta = train_roberta(data['Q_train'], data['Y_train'], task='classification', output_dir = model_dir)
    
    model_dir =  f"roberta_{dataset}" / f"cost"
    roberta = train_roberta(data['Q_train'], data['OT_train'], task='regression', output_dir = model_dir)
