# author: Zhecheng
# class for training and masking
import pandas as pd
import os
import numpy as np
import torch
import copy
import logging
from tqdm import trange
from torch.optim import AdamW
from torch.utils.data import DataLoader
import torch.nn as nn
from datasets import Dataset
from scipy.special import softmax
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, average_precision_score, roc_auc_score, recall_score, precision_score
from transformers import (
    AutoModelForSequenceClassification, 
    AutoTokenizer,
    set_seed,
    get_scheduler,
    )
from .utils import (create_mix, 
                    generate_theoretical_settings, 
                    load_data, EarlyStopping, 
                    get_masks, iid_split)


# batch size
batch_size = 8

def get_duplicates(df):
    is_duplicate = df.duplicated(keep='first')  # mark duplicates except for the first
    number_of_duplicates = is_duplicate.sum()
    total_rows = len(df)
    per_duplicates = (number_of_duplicates / total_rows)

    return per_duplicates



class ConfoundM(object):
    max_length=256 # max length for bert 
    epoch_primary=20 # training epoch for primary goal
    epoch_confounder=20 # training epoch for confounding variable

    def __init__(self, model_name, data_name, pz0, pz1, 
                 alpha, label='label', confounder = 'gender',
                 n_test = 150, data_seed = 123, device='cuda'):
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name,
                                                       device_map = 'auto')
        self.pz0 = pz0 
        self.pz1 = pz1
        self.alpha = alpha
        self.model_name = model_name
        self.data_name = data_name
        self.dementia_finetuned = False
        self.confounder = confounder
        self.label = label
        self.device = device
        self.data_seed = data_seed
        

        self._load_data(data_name = self.data_name, n_test = n_test, data_seed = data_seed)


    def _load_data(self, data_name, sample = True, n_test = 150, data_seed = 123):
        df0, df1 = load_data(data_name)
        setts = generate_theoretical_settings(p_pos_train_z0 = [self.pz0], p_pos_train_z1 = [self.pz1], 
                                              alpha_test=[self.alpha], n_test = n_test)
        # get one setting
        assert setts, 'no available settings'
        sett = np.random.choice(setts)
        
        # create random data split
        logging.debug(f'random seed for data split is {data_seed}')
        dfs = create_mix(df0=df0, df1=df1, target=self.label, setting=sett, sample=sample, 
                         seed=data_seed)
        # check if the data meets the intended distribution
        train_df = dfs['train']
        test_df = dfs['test']
        
        self.check_dist(train_df, test_df)
        logging.info('data distribution is consistent with configuration...')
        # check duplicate rates in train and test
        dp_train = get_duplicates(train_df)
        dp_test = get_duplicates(test_df)
        
        self.duplicates = {'train': dp_train, 'test': dp_test}
        self.data = dfs

        # get data for confounder model training
        # train gender model using only healthy patients
        healthy_dfs = self.data.copy()
        healthy_dfs['train'] = healthy_dfs['train'][healthy_dfs['train'][self.label]==0].reset_index()
        healthy_dfs['test'] = healthy_dfs['test'][healthy_dfs['test'][self.label]==0].reset_index()

        self.confounder_data = healthy_dfs


    def _process_data(self, df, label):
        """ Tokenizer text to torch format from pandas"""
        df = df.reset_index(drop=True)
        data = Dataset.from_pandas(df[['text',label]])
        tokenized_data = data.map(self._tokenize_function, batched=True).remove_columns(["text"]).rename_column(label, "labels")
        tokenized_data.set_format("torch")
        return tokenized_data

    def _compute_metrics(self, logits, labels, group = False, parity = False):
            probs = softmax(logits, axis=1)
            prob, pred = probs[:,1], np.argmax(probs, axis = 1) # probability for predicting dementia and its label
            if parity:
                eval_output = {'dementia_rate': np.mean(prob)}
                return eval_output
            accuracy = accuracy_score(y_true=labels, y_pred=pred)
            aps = average_precision_score(y_true=labels, y_score=prob)
            roc_score = roc_auc_score(y_true=labels, y_score=prob)
            f1 = f1_score(y_true=labels, y_pred=pred)
            eval_output = {"accuracy": accuracy, "roc": roc_score, "aps": aps, "f1": f1}
            if group:
                # precision for each class
                precision_dementia = precision_score(labels, pred, pos_label=1)
                precision_healthy = precision_score(labels, pred, pos_label=0)
                recall_dementia = recall_score(labels, pred, pos_label=1)
                recall_healthy = recall_score(labels, pred, pos_label=0)

                eval_output['precision_dementia'] = precision_dementia
                eval_output['precision_healthy'] = precision_healthy
                eval_output['recall_dementia'] = recall_dementia
                eval_output['recall_healthy'] = recall_healthy
                eval_output['dementia_rate'] = np.mean(prob)

            return eval_output
    
    def _eval_model(self, model, testloader, group = False, parity = False):
        """Evaluate model performance on group level or all instances"""
        model.eval()
        val_loss = 0
        all_labels = []
        all_logits = []
        with torch.no_grad():
            for batch in testloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = model(**batch)
                labels = batch['labels'].cpu().numpy()
                logits = outputs.logits.cpu().numpy()
                loss = outputs.loss.cpu().numpy()
                all_labels.extend(labels)
                all_logits.extend(logits)
                val_loss += loss
        avg_loss = val_loss / len(testloader)
        ## group with different metrics
        metrics = self._compute_metrics(all_logits, all_labels, group=group, parity=parity)
        metrics['eval_loss'] = avg_loss

        return metrics

    def track_layers(self, n = -1, classifier = True, emb = True):
        clf_layer = "classifier.weight"
        tracked_layers = []
        if classifier:
            tracked_layers.append(clf_layer)
        # track token embedding, ffn & attention weights for each layer 
        bert_attn_layers = np.array([[
                f"bert.encoder.layer.{i}.output.dense.weight",
                f"bert.encoder.layer.{i}.intermediate.dense.weight",
                f"bert.encoder.layer.{i}.attention.output.dense.weight",
                f'bert.encoder.layer.{i}.attention.self.value.weight',
                f'bert.encoder.layer.{i}.attention.self.key.weight',                    
                f'bert.encoder.layer.{i}.attention.self.query.weight',
                ] for i in range(11,n,-1)]).flatten()
        tracked_layers = np.append(tracked_layers, bert_attn_layers)
        
        if emb:
            tracked_layers = np.append(tracked_layers,'bert.embeddings.word_embeddings.weight')
        # return layers from bottom to top (emb to layer 12)
        return tracked_layers[::-1]

    # Training model
    def model_training(self, model, trainloader, testloader, tracked_layers, label, epochs, savedir):
        # track parameters on the tracked layers
        config = 'atrain-{:.2f}'.format(self.pz1/self.pz0)
        pre_weights, changes = {}, {}
        for name, param in model.named_parameters():
            if name in tracked_layers:
                pre_weights[name] = param.data.clone().detach()
                changes[name] = torch.zeros_like(pre_weights[name])
        # set optimizers and scheduler
        optimizer = AdamW(model.parameters(), lr= 1e-5)
        lr_scheduler = get_scheduler(name="linear", optimizer=optimizer, 
                                     num_warmup_steps=50, num_training_steps=epochs * len(trainloader))
        

        early_stopping = EarlyStopping(patience=7, verbose=True, device=self.device, path=os.path.join(savedir,f'{label}_{config}.model'))
        print(f'train BERT model for recognizing {label}')

        # iterate batches
        for e in trange(epochs):
            model.train()
            for batch in trainloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss
                loss.sum().backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                # get parameter changes after each batch of training (as in confounding filter paper)
                for name, param in model.named_parameters():
                    if name in pre_weights.keys():
                        # trained weights
                        weight = param.data.clone().detach()
                        diff = weight - pre_weights[name]
                        if torch.max(torch.abs(diff)) == 0: # if no difference
                            continue
                        # normalized absolute change for each step
                        changes[name] += torch.abs(diff)/(torch.max(torch.abs(diff)))
                        pre_weights[name] = weight

            # evaluation
            metrics = self._eval_model(model = model, testloader=testloader)
            logging.info(f"Epoch {e+1} - Eval loss: {metrics['eval_loss']:.3f}, Accuracy: {metrics['accuracy']:.3f}, APS: {metrics['aps']:.3f}, ROC: {metrics['roc']:.3f}, F1: {metrics['f1']:.3f}")
            # Early stopping call find model with best aps
            early_stopping(metrics['aps'], model)
            if early_stopping.early_stop: # end training if early stopping is triggered
                break
        
        # load best checkpoint
        if label == self.confounder: # load confounder model
            self.confounder_model = early_stopping.load_checkpoint()
        else: # load primary model
            self.model = early_stopping.load_checkpoint()

        return changes

    def train_primary(self, num_labels, filter = 'local', savedir='/tmp/checkpoints/'):
        """
        Train towards dementia (primary target) without any constraint
        keep track of the changes
        """
        # set seed to ensure same classifier weight instantiation
        set_seed(42)
        self.model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path= self.model_name, 
                                                                       num_labels=num_labels)


        self.model.to(self.device)

        # track all attention weights, embedding matrix, classification head for primary training
        if filter == 'local':
            tracked_layers = self.track_layers(n = -1, emb=True)
        elif filter == 'global':
            tracked_layers = self.track_layers(n = -1, emb=True, classifier = False)

        dfs = self.data
        df_train = dfs['train']
        df_test = dfs['test']

        # split train and evaluation
        df_train, df_val = iid_split(df_train, self.pz0, self.pz1, self.label, self.data_seed)
        self.check_dist(df_train, df_val, val_set = True)
        token_train = self._process_data(df_train, self.label)
        token_test = self._process_data(df_test, self.label)
        token_val = self._process_data(df_val, self.label)
        # batched data
        train_dataloader = DataLoader(token_train, shuffle=True, batch_size=batch_size)
        val_dataloader = DataLoader(token_val, shuffle=False, batch_size=batch_size) 
        test_dataloader = DataLoader(token_test, shuffle=False, batch_size=batch_size)
        
        # group data for evaluation
        female_test = df_test[df_test['gender']==1].reset_index()
        male_test = df_test[df_test['gender']==0].reset_index()
        female_token_test = self._process_data(female_test, self.label)
        male_token_test = self._process_data(male_test, self.label)
        female_test_dataloader = DataLoader(female_token_test, shuffle=False, batch_size=batch_size)   
        male_test_dataloader = DataLoader(male_token_test, shuffle=False, batch_size=batch_size)   
  
        # for calculating parity metrics in health patients only
        female_ctrl_test = female_test[female_test['label']==0].reset_index()
        male_ctrl_test = male_test[male_test['label']==0].reset_index()
        female_ctrl_token_test = self._process_data(female_ctrl_test, self.label)
        male_ctrl_token_test = self._process_data(male_ctrl_test, self.label)
        female_ctrl_test_dataloader = DataLoader(female_ctrl_token_test, shuffle=False, batch_size=batch_size)   
        male_ctrl_test_dataloader = DataLoader(male_ctrl_token_test, shuffle=False, batch_size=batch_size)  


        # early stopping on evaluate dataset
        changes = self.model_training(trainloader=train_dataloader, testloader=val_dataloader,
                                      model=self.model, tracked_layers=tracked_layers, 
                                      label = self.label, epochs=self.epoch_primary,
                                      savedir=savedir)
        
        self.dementia_finetuned = True
        
        # get original performance
        print('evaluating original dementia model...')
        self.evaluation = {}
        self.evaluation['orig_dementia'] = self._eval_model(self.model, test_dataloader)
        self.evaluation['orig_dementia_female'] = self._eval_model(self.model, female_test_dataloader, group = True)
        self.evaluation['orig_dementia_male'] = self._eval_model(self.model, male_test_dataloader, group = True)
        self.evaluation['orig_dementia_female_ctrl'] = self._eval_model(self.model, female_ctrl_test_dataloader, parity = True)
        self.evaluation['orig_dementia_male_ctrl'] = self._eval_model(self.model, male_ctrl_test_dataloader, parity = True)

        return changes
    
    def filter_global(self, num_labels, reference = 'direct', savedir='/tmp/checkpoints/global'):
        
        assert self.dementia_finetuned, 'Train the Dementia model first!'
        
        label = self.confounder
        dfs = self.confounder_data
        df_train = dfs['train'].drop_duplicates().reset_index()
        df_test = dfs['test'].drop_duplicates().reset_index()

        df_train, df_val = train_test_split(df_train, test_size=0.2, 
                                            stratify=df_train[self.confounder],random_state=self.data_seed)


        token_train = self._process_data(df_train, label)
        token_test = self._process_data(df_test, label)
        token_val = self._process_data(df_val, label)

        # not including classification layer because it is assumed to be changed most
        tracked_layers = self.track_layers(n = -1, classifier = False, emb = True)

        # batched data
        train_dataloader = DataLoader(token_train, shuffle=True, batch_size=batch_size)
        test_dataloader = DataLoader(token_test, shuffle=False, batch_size=batch_size)
        val_dataloader = DataLoader(token_val, shuffle=False, batch_size=batch_size)  
        
        if reference == 'direct':
            logging.info('train from an pretrained model instance...')
            # make sure models are initialized the same
            set_seed(42)
            self.confounder_model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path = self.model_name, 
                                                                                    num_labels=num_labels).to(self.device)
        elif reference == 'phase2':
            logging.info('train from the finetuned dementia model...')
            self.confounder_model = copy.deepcopy(self.model)
        
        # record the number of trainable parameters
        self.num_model_params = np.sum([p.numel() for p in self.confounder_model.parameters() if p.requires_grad])
        
        # early stopping on validation set
        changes = self.model_training(trainloader=train_dataloader, testloader=val_dataloader,
                                      model = self.confounder_model, tracked_layers=tracked_layers,
                                      label = label, epochs=self.epoch_confounder,
                                      savedir=savedir)
        
        print('evaluating original gender model...')
        self.evaluation['orig_gender'] = self._eval_model(self.confounder_model, test_dataloader)
        return changes

    def filter_local(self, n_layer, emb, savedir='/tmp/checkpoints/local'):
        label = self.confounder
        dfs = self.confounder_data
        df_train = dfs['train'].drop_duplicates().reset_index()
        df_test = dfs['test'].drop_duplicates().reset_index()

        # split train and evaluation, retain the distribution of label
        df_train, df_val = train_test_split(df_train, test_size=0.2, 
                                            stratify=df_train[self.confounder],random_state=self.data_seed)


        token_train = self._process_data(df_train, label)
        token_test = self._process_data(df_test, label)
        token_val = self._process_data(df_val, label)
        
        # batched data
        train_dataloader = DataLoader(token_train, shuffle=True, batch_size=batch_size)
        val_dataloader = DataLoader(token_val, shuffle=False, batch_size=batch_size)
        test_dataloader = DataLoader(token_test, shuffle=False, batch_size=batch_size)

        tracked_layers = self.track_layers(n = n_layer, emb = emb, classifier = True)
        # logging.info('trainable layers are: {}'.format(tracked_layers))
        # inherit the weights from trained dementia model
        self.confounder_model = copy.deepcopy(self.model)
        
        # froze trained parameters(BERT layer) for phase 2
        for name, param in self.confounder_model.named_parameters():
            if name in tracked_layers:
                param.requires_grad = True
            else:
                param.requires_grad = False
        
        # record the number of trainable parameters
        self.num_model_params = np.sum([p.numel() for p in self.confounder_model.parameters() if p.requires_grad])
        
        # early stopping on validation set
        # second phase of training
        changes = self.model_training(trainloader=train_dataloader, testloader=val_dataloader,
                                      model = self.confounder_model, tracked_layers=tracked_layers,
                                      label = label, epochs=self.epoch_confounder,
                                      savedir=savedir)
        
        print('evaluating original gender model...')
        self.evaluation['orig_gender'] = self._eval_model(self.confounder_model, test_dataloader)

        return changes
        
    def apply_local_mask(self, changes, ratio, alpha = 0):
        
        self.masked_dmodel = copy.deepcopy(self.model)
        self.masked_gmodel = copy.deepcopy(self.confounder_model)
        mask_dict = {}

        # get masks and assign weights
        for name in changes.keys():
            #normalize change
            # eliminate top ratio% changed weights for each layer
            threshold = np.quantile(changes[name].cpu(), 1 - ratio)
            # Mask out weights
            # phase1_weights[name][changes[name] > threshold] = 0.
            mask_dict[name] = (changes[name] < threshold)
            mask_dict[name][mask_dict[name]== 0.] = alpha

        num_neurons = np.sum([np.sum(1 - c.cpu().numpy()) for c in mask_dict.values()])
        logging.info(f'size of the mask is {num_neurons:,}, which is {num_neurons/self.num_model_params:.2%} of the trainable parameters ({self.num_model_params:,}) in dementia model')
        self.masked_params = num_neurons

        # replacing phase 1 weights
        for (named, paramd), (nameg, paramg) in zip(self.masked_dmodel.named_parameters(), self.masked_gmodel.named_parameters()):
            assert named == nameg, 'not the same weight'
            if named in mask_dict.keys():
                paramd.data = torch.mul(paramd.data, mask_dict[named])
                paramg.data = torch.mul(paramg.data, mask_dict[named])

        return
    
    def apply_global_mask(self, changes1, changes2, ratio, type='intersection', alpha = 0.):
        # get a masked_model
        # deep copy
        self.masked_dmodel = copy.deepcopy(self.model)
        self.masked_gmodel = copy.deepcopy(self.confounder_model)

        layer_of_mask = get_masks(changes1, changes2, ratio, type, alpha = alpha)
        num_neurons = np.sum([np.sum(1 - c.cpu().numpy()) for c in layer_of_mask.values()])
        logging.info(f'size of the {type} mask is {num_neurons}, which is {num_neurons/self.num_model_params:.2%} of the trainable parameters in dementia model')
        self.masked_params = num_neurons

        for (named, paramd), (nameg, paramg) in zip(self.masked_dmodel.named_parameters(), self.masked_gmodel.named_parameters()):
            if named in layer_of_mask:
                assert named == nameg, 'not same layer'
                # Retrieve the corresponding mask for the layer and reverse 0 and 1
                mask_layer = layer_of_mask[named]

                # mask the location from dementia model and gender model
                paramd.data = torch.mul(paramd.data, mask_layer)
                paramg.data = torch.mul(paramg.data, mask_layer)
                #paramg.data[mask_layer] = 0.
        return

    def eval_masked_model(self):
        try:
            all_test = self.data['test']
            all_healthy_test = self.confounder_data['test'] 


            token_test = self._process_data(all_test, self.label)
            gender_token_test = self._process_data(all_healthy_test, self.confounder)
            # batched data
            gender_test_dataloader = DataLoader(gender_token_test, shuffle=False, batch_size=batch_size)   
            test_dataloader = DataLoader(token_test, shuffle=False, batch_size=batch_size)   

            # group data for evaluation
            female_test = all_test[all_test['gender']==1].reset_index()
            male_test = all_test[all_test['gender']==0].reset_index()
            female_token_test = self._process_data(female_test, self.label)
            male_token_test = self._process_data(male_test, self.label)
            female_test_dataloader = DataLoader(female_token_test, shuffle=False, batch_size=batch_size)   
            male_test_dataloader = DataLoader(male_token_test, shuffle=False, batch_size=batch_size)   

            # NOTE: healthy_ctrl 06/04
            # for calculating parity metrics in health patients only
            female_ctrl_test = all_healthy_test[all_healthy_test['gender']==1].reset_index()
            male_ctrl_test = all_healthy_test[all_healthy_test['gender']==0].reset_index()
            female_ctrl_token_test = self._process_data(female_ctrl_test, self.label)
            male_ctrl_token_test = self._process_data(male_ctrl_test, self.label)
            female_ctrl_test_dataloader = DataLoader(female_ctrl_token_test, shuffle=False, batch_size=batch_size)   
            male_ctrl_test_dataloader = DataLoader(male_ctrl_token_test, shuffle=False, batch_size=batch_size) 

            if self.evaluation:
                print('evaluating masked model...')
                print('masked dementia model...')
                masked_d = self._eval_model(self.masked_dmodel, test_dataloader)
                masked_d_female = self._eval_model(self.masked_dmodel, female_test_dataloader, group = True)
                masked_d_male = self._eval_model(self.masked_dmodel, male_test_dataloader, group = True)
                masked_d_female_ctrl = self._eval_model(self.masked_dmodel, female_ctrl_test_dataloader, parity=True)
                masked_d_male_ctrl = self._eval_model(self.masked_dmodel, male_ctrl_test_dataloader, parity=True)

                print('masked gender model...')
                masked_g = self._eval_model(self.masked_gmodel, gender_test_dataloader)
                
                self.evaluation['masked_dementia'] = masked_d
                self.evaluation['masked_dementia_female'] = masked_d_female
                self.evaluation['masked_dementia_male'] = masked_d_male
                self.evaluation['orig_dementia_female_ctrl'] = masked_d_female_ctrl
                self.evaluation['orig_dementia_male_ctrl'] = masked_d_male_ctrl
                self.evaluation['masked_gender'] = masked_g


        except Exception as e:
            logging.error(f'Can not evaluate because of {e}')

        return 

   
    # helper function
    def _tokenize_function(self, examples):
            return self.tokenizer(examples["text"], padding="max_length", truncation=True, max_length=self.max_length)


    def check_dist(self, train_df, test_df, atol = 0.05, rtol=0.05, val_set = False):
        logger = logging.getLogger(__name__)

        pz1_train = len(train_df[(train_df[self.confounder]==1) & (train_df[self.label]==1)])/len(train_df[train_df[self.confounder]==1])
        pz0_train = len(train_df[(train_df[self.confounder]==0) & (train_df[self.label]==1)])/len(train_df[train_df[self.confounder]==0])

        pz1_test = len(test_df[(test_df[self.confounder]==1) & (test_df[self.label]==1)])/len(test_df[test_df[self.confounder]==1])
        pz0_test = len(test_df[(test_df[self.confounder]==0) & (test_df[self.label]==1)])/len(test_df[test_df[self.confounder]==0])

        alpha_test = pz1_test/pz0_test
        try:
            assert np.isclose(pz1_train, self.pz1, atol=atol, rtol=rtol), f"P(y=1|z=1)={pz1_train} not consistent with configuration {self.pz1}"
            assert np.isclose(pz0_train, self.pz0, atol=atol, rtol=rtol), f"P(y=1|z=1)={pz0_train} not consistent with configuration {self.pz0}"
            if val_set:
                assert np.isclose(alpha_test, 1/self.alpha, atol=atol, rtol=rtol), f"alpha_test={alpha_test} not consistent with configuration {1/self.alpha}"
            else:
                assert np.isclose(alpha_test, self.alpha, atol=atol, rtol=rtol),f"alpha_test={alpha_test} not consistent with configuration {self.alpha}"
        except AssertionError as e:
            logger.error(f"Assertion failed: {e}")
            raise AssertionError(e)